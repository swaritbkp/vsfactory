"""Regression pinning and the `knot gate` CI quality gate.

A *regression suite* (`regression_suite.yaml`) is a short list of certified items that a model once
failed (or that a person chose to keep). Each pinned case carries everything needed to grade it
again: the question, the certified answer and the generator inputs (domain, seed, preset, block and
item index) that reproduce its image deterministically.

    suite_name: rack-monochrome-audit regressions
    version: 1
    min_accuracy: 0.85
    cases:
      - case_id: kw-...            # the certified bundle id
        domain: rack_patch
        seed: 12
        preset_or_config: {preset: High stress, block_id: b1, item_index: 12}
        question_family: T1_TRACE
        expected_answer: {type: label, value: "07", canonical: "07", tolerance: 0.0}
        ...

`pin_case_to_suite` deduplicates by `(domain, seed, question_family)`: pinning the same item again
updates its row. `run_gate` grades a model on every case with the kernel's grader
(`knotcore.grading` through the arena's `grade`) and returns the CI exit code:

* 0 -- accuracy >= min_accuracy;  1 -- accuracy below it;  2 -- the suite is missing, invalid or empty.

Offline `mock` mode needs no image and no network. `live` mode re-renders each case from its seed
into a cache folder, checks that the re-rendered item still has the pinned bundle id and answer
(an irreproducible case fails), then asks the model through the arena's OpenRouter client.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from knotwork.arena.schemas import Domain, ItemRecord, find_secret

SUITE_FILE = "regression_suite.yaml"
SUITE_VERSION = 1
DEFAULT_MIN_ACCURACY = 0.85
EXIT_PASS, EXIT_FAIL, EXIT_INVALID = 0, 1, 2
GateMode = Literal["mock", "live"]


class RegressionCase(BaseModel):
    """One pinned item: enough to reproduce it and to grade any model's reply to it."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=80)
    domain: Domain
    seed: int = Field(ge=0, lt=2**31)
    preset_or_config: dict[str, Any] = Field(default_factory=dict)
    question_family: str = Field(min_length=1)
    expected_answer: dict[str, Any]
    question: str = ""
    options: list[str] | None = None
    source_session: str = ""
    pinned_from_model: str = ""
    pinned_at_utc: str = ""
    note: str = Field(default="", max_length=280)

    @property
    def dedup_key(self) -> tuple[str, int, str]:
        return (self.domain, self.seed, self.question_family)

    @property
    def truth(self) -> str:
        return str(self.expected_answer.get("canonical", self.expected_answer.get("value", "")))

    def as_item(self) -> ItemRecord:
        """The case as an arena item (no image directory), for the arena's grader and mock model."""
        cfg = self.preset_or_config
        return ItemRecord(
            block_id=str(cfg.get("block_id", "gate")), item_index=int(cfg.get("item_index", 0)),
            seed=self.seed, domain=self.domain, preset=str(cfg.get("preset", "Clean")),
            family=self.question_family, certified=True, bundle_id=self.case_id,
            question=self.question, options=self.options, answer=self.expected_answer,
            truth=self.truth,
        )  # fmt: skip


class RegressionSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite_name: str = Field(min_length=1, max_length=160)
    version: int = SUITE_VERSION
    min_accuracy: float = Field(default=DEFAULT_MIN_ACCURACY, ge=0.0, le=1.0)
    cases: list[RegressionCase] = Field(default_factory=list)


def case_from_item(
    item: ItemRecord, source_session: str, model: str, pinned_at_utc: str, note: str = ""
) -> RegressionCase:
    """A pinned case for an arena item (the item's generator inputs, question and answer)."""
    return RegressionCase(
        case_id=item.bundle_id or f"{item.block_id}-{item.item_index}", domain=item.domain,  # type: ignore[arg-type]
        seed=item.seed, question_family=item.family, expected_answer=dict(item.answer),
        preset_or_config={"preset": item.preset, "block_id": item.block_id,
                          "item_index": item.item_index},
        question=item.question, options=item.options, source_session=source_session,
        pinned_from_model=model, pinned_at_utc=pinned_at_utc, note=note.strip()[:280],
    )  # fmt: skip


# ------------------------------------------------------------------ storage
def dump_suite(suite: RegressionSuite) -> str:
    data = suite.model_dump(mode="json")
    if find_secret(data):
        raise ValueError("refusing to write a secret-shaped value to a regression suite")
    header = (
        "# Knotwork regression suite: run it with `knot gate --suite <this file> --model <id>`\n"
    )
    return header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)


def load_suite(path: Path) -> RegressionSuite:
    """Parse and validate a suite file. Raises ValueError with a readable reason."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc.strerror or exc}") from None
    except yaml.YAMLError as exc:
        raise ValueError(f"{path} is not valid YAML: {exc}") from None
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must hold a mapping with suite_name, version and cases")
    try:
        suite = RegressionSuite.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        raise ValueError(f"{path}: {where}: {first['msg']}") from None
    if suite.version != SUITE_VERSION:
        raise ValueError(f"{path}: unsupported suite version {suite.version}")
    return suite


def write_suite(path: Path, suite: RegressionSuite) -> None:
    """Atomic write: readers never see half a suite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(dump_suite(suite), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def pin_case_to_suite(
    suite_path: Path, case: RegressionCase, suite_name: str | None = None
) -> tuple[RegressionSuite, bool]:
    """Add `case` to the suite at `suite_path` (created if missing). A case with the same
    `(domain, seed, question_family)` is replaced in place. Returns (suite, added)."""
    path = Path(suite_path)
    if path.is_file():
        suite = load_suite(path)
    else:
        suite = RegressionSuite(
            suite_name=suite_name or f"{path.parent.name or 'arena'} regressions"
        )
    cases = list(suite.cases)
    at = next((i for i, c in enumerate(cases) if c.dedup_key == case.dedup_key), None)
    if at is None:
        cases.append(case)
    else:
        cases[at] = case
    suite = suite.model_copy(update={"cases": cases})
    write_suite(path, suite)
    return suite, at is None


def pinned_keys(suite_path: Path) -> set[tuple[str, int, str]]:
    """Dedup keys already in a suite (empty when the file is missing or invalid)."""
    try:
        return {c.dedup_key for c in load_suite(suite_path).cases}
    except ValueError:
        return set()


# ------------------------------------------------------------------ the gate
def _reproduce(case: RegressionCase, cache: Path) -> tuple[ItemRecord | None, str]:
    """Re-render a case from its seed; (item, "") or (None, reason) when it no longer matches."""
    from knotwork.arena.generation import make_item

    cfg = case.preset_or_config
    item = make_item(str(cfg.get("block_id", "gate")), case.domain, str(cfg.get("preset", "Clean")),
                     int(cfg.get("item_index", 0)), case.seed, cache)  # fmt: skip
    if not item.certified:
        return None, "re-rendered item is no longer certified"
    if item.truth != case.truth:
        return None, f"re-rendered answer {item.truth!r} != pinned {case.truth!r}"
    return item, ""


def _live_asker(model: str, cache: Path, max_tokens: int) -> Any:
    from knotwork.arena.runner import Runner
    from knotwork.arena.schemas import BlockSpec, SessionConfig

    cfg = SessionConfig(session_name="knot gate", slug="knot-gate", mode="live",
                        data_dir=str(cache), results_dir=str(cache / "_results"),
                        default_models=[model], max_tokens=max_tokens,
                        blocks=[BlockSpec(block_id="gate", domain="tangle", preset="Clean", n=1,
                                          seed=0)])  # fmt: skip
    (cache / "_results").mkdir(parents=True, exist_ok=True)
    return Runner(cfg, cache / "_results")


def run_gate(
    suite: RegressionSuite,
    model: str,
    mode: GateMode = "mock",
    min_accuracy: float | None = None,
    cache_dir: Path | None = None,
    max_tokens: int = 512,
) -> dict[str, Any]:
    """Grade `model` on every case. The report's `exit_code` is the CI verdict."""
    from knotwork.arena.runner import ApiError, grade, mock_reply

    threshold = suite.min_accuracy if min_accuracy is None else float(min_accuracy)
    rows: list[dict[str, Any]] = []
    tmp = None
    if mode == "live" and cache_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="knot-gate-")
    cache = Path(tmp.name) if tmp is not None else Path(cache_dir or ".")
    try:
        asker = _live_asker(model, cache, max_tokens) if mode == "live" else None
        for case in suite.cases:
            row: dict[str, Any] = {"case_id": case.case_id, "domain": case.domain,
                                   "seed": case.seed, "family": case.question_family,
                                   "expected": case.truth, "extracted": "", "passed": False,
                                   "error": ""}  # fmt: skip
            item = case.as_item()
            if asker is not None:
                reproduced, why = _reproduce(case, cache)
                if reproduced is None:
                    rows.append({**row, "error": why})
                    continue
                item = reproduced
                asker.data = cache
                try:
                    reply = asker._ask_live(model, item).text
                except ApiError as exc:
                    rows.append({**row, "error": f"api error: {exc}"[:200]})
                    continue
            else:
                reply = mock_reply(model, item, {}).text
            correct, extracted, category = grade(reply, item)
            rows.append({**row, "extracted": extracted, "passed": bool(correct),
                         "error": "" if correct else (category or "wrong")})  # fmt: skip
    finally:
        if tmp is not None:
            tmp.cleanup()
    n = len(rows)
    passed = sum(r["passed"] for r in rows)
    accuracy = passed / n if n else 0.0
    ok = n > 0 and accuracy >= threshold
    return {"suite_name": suite.suite_name, "model": model, "mode": mode, "n": n,
            "passed": passed, "failed": n - passed, "accuracy": accuracy,
            "min_accuracy": threshold, "verdict": "PASS" if ok else "FAIL",
            "exit_code": EXIT_PASS if ok else EXIT_FAIL, "cases": rows}  # fmt: skip


def ascii_table(headers: list[str], rows: list[list[str]]) -> str:
    """A plain `+---+` table: ASCII only, so it prints on any CI log or Windows console."""
    clean = [[str(c).encode("ascii", "replace").decode("ascii") for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in clean)) if clean else len(h)
              for i, h in enumerate(headers)]  # fmt: skip
    rule = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)) + " |"

    return "\n".join([rule, line(headers), rule, *(line(r) for r in clean), rule])


def format_report(report: dict[str, Any]) -> str:
    rows = [[r["case_id"][:24], r["domain"], str(r["seed"]), r["family"], r["expected"][:16],
             (r["extracted"] or "-")[:16], "PASS" if r["passed"] else "FAIL",
             (r["error"] or "")[:40]] for r in report["cases"]]  # fmt: skip
    table = ascii_table(
        ["case", "domain", "seed", "family", "expected", "got", "result", "note"], rows
    )
    summary = (f"knot gate: {report['verdict']}  model={report['model']}  mode={report['mode']}  "
               f"accuracy={report['accuracy']:.1%} ({report['passed']}/{report['n']})  "
               f"min={report['min_accuracy']:.1%}")  # fmt: skip
    return f"{table}\n{summary}"


def gate_command(suite_path: Path | str, model: str = "<model-id>") -> str:
    target = str(suite_path).replace("\\", "/")
    return f'uv run knot gate --suite "{target}" --model {model} --mode mock'


def main_gate(
    suite_path: Path, model: str, mode: GateMode, min_accuracy: float | None, json_out: Path | None
) -> tuple[int, str]:
    """CLI body: (exit code, text to print)."""
    try:
        suite = load_suite(suite_path)
    except ValueError as exc:
        return EXIT_INVALID, f"knot gate: invalid suite: {exc}"
    if not suite.cases:
        return EXIT_INVALID, f"knot gate: {suite_path} has no pinned cases"
    report = run_gate(suite, model, mode, min_accuracy)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return int(report["exit_code"]), format_report(report)


__all__ = [
    "SUITE_FILE", "RegressionCase", "RegressionSuite", "ascii_table", "case_from_item",
    "dump_suite", "format_report", "gate_command", "load_suite", "main_gate",
    "pin_case_to_suite", "pinned_keys", "run_gate", "write_suite",
]  # fmt: skip
