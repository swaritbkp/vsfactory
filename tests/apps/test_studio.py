"""The Studio starts with four tabs, and its pure helpers behave."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from knotwork.pipeline.io import append_jsonl, write_json
from knotwork.studio import calibration, factory

REPO = Path(__file__).resolve().parents[2]


def test_app_renders_four_tabs_without_errors() -> None:
    app = AppTest.from_file(str(REPO / "apps" / "inspector.py"), default_timeout=12).run()
    assert not app.exception, app.exception
    labels = [t.label for t in app.tabs]
    for name in ("Factory Studio", "Forensics & Counterfactuals", "Human Calibration Lab",
                 "Export & Lineage"):  # fmt: skip
        assert name in labels


def test_calibration_queue_is_blind_and_records_are_graded(tmp_path: Path) -> None:
    answer = {"type": "int", "value": 3, "canonical": "3", "tolerance": 0.0}
    write_json(
        tmp_path / "bundles/b/task.json", {"answer": answer, "question": {"text": "How many?"}}
    )
    append_jsonl(
        tmp_path / "index.jsonl",
        [{"bundle_id": "kw-00000000000000000001", "family": "T2_CROSS", "plugin": "tangle",
          "role": "base", "dir": "bundles/b", "answer_canonical": "3"}],
    )  # fmt: skip
    (item,) = calibration.queue(tmp_path, "p1")
    assert not {"answer", "answer_canonical", "truth"} & set(item)
    log = tmp_path / "log.jsonl"
    assert calibration.record(item, "4", 2.5, "p1", log)["correct"] is False
    result = calibration.summary(log, tmp_path / "none.jsonl")
    assert result["human_overall"] == 0.0 and result["confusion"][0]["response"] == "4"


def test_batch_command_matches_the_cli() -> None:
    args = factory.batch_command(
        "tangle", 5, Path("x"), split="ood", holdout_params=["strands=9:10"],
        difficulty=factory.difficulty_from({"reasoning_load": 0.4, "visual_stress": None}),
    )  # fmt: skip
    assert args[:3] == ["generate", "--plugin", "tangle"] and "--reasoning-load" in args
    assert "--visual-stress" not in args and factory.command_line(args).startswith("uv run knot ")
