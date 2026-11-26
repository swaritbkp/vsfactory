"""Factory telemetry: yield, rejection Pareto, false quarantine, class balance and throughput.

`knot generate` writes `telemetry.json` beside every dataset; `knot telemetry --compare A B` diffs
two of them and exits 1 when yield falls or false quarantine rises by more than `REGRESSION`.
Generated scenes carry no injected faults, so every quarantined candidate counts as false.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from knotwork.pipeline.io import write_json

SCHEMA = "knotwork.telemetry/1"
FILENAME = "telemetry.json"
REGRESSION = 0.02


def _rate(part: float, whole: float) -> float:
    return round(part / whole, 6) if whole else 0.0


def pareto(codes: Counter[str]) -> list[dict[str, Any]]:
    """Rejection codes, most frequent first, with their share and the running share."""
    total, running, out = sum(codes.values()), 0, []
    for code, count in sorted(codes.items(), key=lambda kv: (-kv[1], kv[0])):
        running += count
        out.append(
            {"code": code, "count": count, "share": _rate(count, total),
             "cumulative_share": _rate(running, total)}
        )  # fmt: skip
    return out


class Telemetry:
    """Accumulates one run slot by slot (constant memory in the number of slots)."""

    def __init__(self) -> None:
        self.families: dict[str, Counter[str]] = defaultdict(Counter)
        self.answers: dict[str, Counter[str]] = defaultdict(Counter)
        self.codes: Counter[str] = Counter()

    def add(self, family: str, result: dict[str, Any]) -> None:
        """Count one `run_slot` result of a slot scheduled for `family`."""
        row = self.families[family]
        row["slots"] += 1
        row["certified"] += int(result["ok"])
        row["candidates"] += int(result["n_candidates"])
        row["bundles"] += len(result["bundles"])
        for q in result["quarantine"]:
            if "codes" in q:
                row["quarantined"] += 1
                self.codes.update(q["codes"])
            else:
                row["sampling_skips"] += 1
        for role, _, task in result["bundles"]:
            if role == "base":
                self.answers[family][str(task["answer"]["canonical"])] += 1

    def _summary(self, row: Counter[str]) -> dict[str, Any]:
        passed = row["candidates"] - row["quarantined"]
        return {
            "slots": row["slots"],
            "certified": row["certified"],
            "candidates": row["candidates"],
            "quarantined": row["quarantined"],
            "sampling_skips": row["sampling_skips"],
            "yield_rate": _rate(passed, row["candidates"]),
            "slot_yield": _rate(row["certified"], row["slots"]),
            "false_quarantine_rate": _rate(row["quarantined"], row["candidates"]),
        }

    def report(self, seconds: float, **context: Any) -> dict[str, Any]:
        total: Counter[str] = Counter()
        families: dict[str, Any] = {}
        for name in sorted(self.families):
            total.update(self.families[name])
            answers = self.answers[name]
            n = sum(answers.values())
            families[name] = self._summary(self.families[name]) | {
                "class_balance": {
                    "n": n,
                    "n_classes": len(answers),
                    "majority_share": _rate(max(answers.values(), default=0), n),
                    "counts": dict(sorted(answers.items())),
                }
            }
        return {
            "schema": SCHEMA,
            **context,
            "aggregate": self._summary(total),
            "families": families,
            "rejection_pareto": pareto(self.codes),
            "throughput": {
                "seconds": round(seconds, 3),
                "items_per_sec": _rate(total["certified"], seconds),
                "bundles_per_sec": _rate(total["bundles"], seconds),
            },
        }


def write(out: Path, report: dict[str, Any]) -> Path:
    write_json(out / FILENAME, report)
    return out / FILENAME


def load(path: Path) -> dict[str, Any]:
    path = Path(path)
    path = path / FILENAME if path.is_dir() else path
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a telemetry report")
    return data


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Metric-by-metric difference b - a, and the list of regressions."""
    rows: list[dict[str, Any]] = []
    regressions: list[str] = []

    def put(scope: str, metric: str, x: float | None, y: float | None) -> None:
        delta = None if x is None or y is None else round(y - x, 6)
        rows.append({"scope": scope, "metric": metric, "a": x, "b": y, "delta": delta})
        if delta is None:
            return
        worse = (metric.endswith("yield_rate") or metric == "slot_yield") and delta < -REGRESSION
        worse = worse or (metric == "false_quarantine_rate" and delta > REGRESSION)
        if worse:
            regressions.append(f"{scope} {metric}: {x} -> {y}")

    scopes = [("aggregate", a["aggregate"], b["aggregate"])]
    for name in sorted(set(a["families"]) | set(b["families"])):
        scopes.append((name, a["families"].get(name, {}), b["families"].get(name, {})))
    for scope, left, right in scopes:
        for metric in ("yield_rate", "slot_yield", "false_quarantine_rate"):
            put(scope, metric, left.get(metric), right.get(metric))
    for metric in ("items_per_sec", "bundles_per_sec"):
        put("throughput", metric, a["throughput"][metric], b["throughput"][metric])
    before = {r["code"]: r["count"] for r in a["rejection_pareto"]}
    after = {r["code"]: r["count"] for r in b["rejection_pareto"]}
    for code in sorted(set(before) | set(after)):
        put("rejections", code, before.get(code, 0), after.get(code, 0))
    return {"rows": rows, "regressions": regressions}


def run_compare(a: Path, b: Path) -> int:
    diff = compare(load(a), load(b))
    print(f"{'scope':<22}{'metric':<26}{'a':>12}{'b':>12}{'delta':>12}")
    for r in diff["rows"]:
        cells = "".join(
            f"{'-' if r[k] is None else format(r[k], 'g'):>12}" for k in ("a", "b", "delta")
        )
        print(f"{r['scope']:<22}{r['metric']:<26}{cells}")
    for line in diff["regressions"]:
        print(f"REGRESSION {line}")
    return 1 if diff["regressions"] else 0
