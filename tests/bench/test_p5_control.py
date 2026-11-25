"""SPEC D-37: the matched true-premise control of the false-premise probe (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from knotwork.bench import p5_control, stats
from knotwork.bench.run import PROFILE, load_pool

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "bench" / "fixtures" / "data"


def test_control_is_certified_and_differs_only_in_the_claimed_zone(tmp_path: Path) -> None:
    source = load_pool(FIX).gaslight[:1]
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    items, records = p5_control.build_control(
        FIX, tmp_path, [source[0].key], profile, workers=1
    )  # fmt: skip
    rec = records[0]
    assert rec["checks"]["control_certified"] and rec["checks"]["kernel_zone_is_claimed"]
    assert rec["checks"]["kernel_premise_holds"] and rec["checks"]["false_premise_was_false"]
    assert items[0].beauty == source[0].beauty and items[0].meta["holds"] is True
    assert rec["claimed_zone_true"] != rec["claimed_zone_false"]
    swapped = source[0].prompt.replace(
        f"in the {rec['claimed_zone_false']} zone", f"in the {rec['claimed_zone_true']} zone"
    )
    assert items[0].prompt == swapped and items[0].prompt != source[0].prompt


def test_premise_control_statistics() -> None:
    split = stats.premise_control([("yes", "no")] * 20 + [("no", "no")] * 4)
    assert (split["true_yes"], split["false_yes"], split["verdict"]) == (20, 0, "discriminates")
    assert split["mcnemar_p"] == pytest.approx(2 * 0.5**20, abs=1e-6)
    always = stats.premise_control([("yes", "yes")] * 24)
    assert always["verdict"] == "accepts regardless" and always["mcnemar_p"] == 1.0
    never = stats.premise_control([("no", "no")] * 24)
    assert never["verdict"] == "rejects regardless" and never["balanced_accuracy"] == 0.5
    mixed = stats.premise_control([("yes", "no"), ("no", "yes"), (None, "no"), ("yes", "yes")])
    assert mixed["verdict"] == "no detectable discrimination" and mixed["unparsed"] == 1


def test_analyze_pairs_by_item_and_grades_the_reading() -> None:
    def row(block: str, key: str, reply: str, correct: bool = False) -> dict[str, object]:
        return {"slot": "vlm_a", "block": block, "key": key, "reply": reply, "correct": correct,
                "missing": False}  # fmt: skip

    items = [{"source_bundle_id": "x", "answer": {"value": {"value": 40.0}, "tolerance": 1.6}}]
    res = p5_control.analyze(
        [row("gaslight", "gas:x", "no; 41")], [row("p5_control", "p5:x", "Yes; 70", False)], items
    )["vlm_a"]
    assert (res["n"], res["true_yes"], res["false_yes"]) == (1, 1, 0)
    assert (res["true_reading_correct"], res["false_reading_correct"]) == (0, 1)


def test_source_keys_require_identical_blocks() -> None:
    rows = [
        {"block": "gaslight", "slot": s, "key": k} for s in ("a", "b") for k in ("gas:1", "gas:2")
    ]
    assert p5_control.source_keys(rows) == ["gas:1", "gas:2"]
    with pytest.raises(ValueError):
        p5_control.source_keys(rows[:-1])


def test_live_budget_above_the_cap_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert p5_control.main(["--live", "--budget-usd", "0.51"]) == 4
    assert "E_BUDGET_OVER_CAP" in capsys.readouterr().out


def test_mock_run_writes_separate_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key = load_pool(FIX).gaslight[0].key
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(json.dumps({"slot": s, "block": "gaslight", "key": key, "reply": "no; 1",
                            "correct": False, "missing": False, "cost_usd": 0.001}) + "\n"
                for s in ("mock_a", "mock_b", "mock_c")), encoding="utf-8")  # fmt: skip
    monkeypatch.setattr(p5_control, "N_ITEMS", 1)

    def run() -> int:
        return p5_control.run_control(FIX, tmp_path / "out", REPO / "configs" / "models.yaml",
                                      tmp_path / "replay", "mock", None, False, source, workers=1)  # fmt: skip

    # the committed fixtures drop masks and id buffers, so the source cannot be re-verified
    assert run() == 1 and not (tmp_path / "out" / p5_control.TRIALS_FILE).exists()
    monkeypatch.setattr("knotwork.pipeline.verify.verify_bundle", lambda *_: None)
    assert run() == 0
    rep = json.loads((tmp_path / "out" / p5_control.REPORT_FILE).read_text(encoding="utf-8"))
    assert rep["network_calls"] == 0 and rep["trials"] == 3 and set(rep["models"]) == {
        "mock_a", "mock_b", "mock_c"}  # fmt: skip
    assert all(m["true_yes"] == 1 for m in rep["models"].values())
    # a non-live run never replaces a live record
    record = tmp_path / "out" / p5_control.REPORT_FILE
    record.write_text(json.dumps(dict(rep, mode="live")), encoding="utf-8")
    assert run() == 4 and json.loads(record.read_text(encoding="utf-8"))["mode"] == "live"
