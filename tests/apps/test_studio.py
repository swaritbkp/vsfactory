"""The Studio's pages render on the live API, and its pure helpers behave."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from knotwork.pipeline.io import append_jsonl, write_json
from knotwork.studio import calibration, factory, present

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def app() -> AppTest:
    return AppTest.from_file(str(REPO / "apps" / "inspector.py"), default_timeout=60).run()


def test_studio_opens_with_a_certified_sample(app: AppTest) -> None:
    assert not app.exception, app.exception
    item = app.session_state["item"]
    assert item["certificate"]["verdict"] == "CERTIFIED" and item["task"] is not None
    assert [m.label for m in app.metric][:2] == ["Certified answer", "Plugin / family"]


@pytest.mark.parametrize("page", ["forensics", "human_eval", "export"])
def test_other_pages_render(app: AppTest, page: str) -> None:
    app.switch_page(f"app_pages/{page}.py").run()
    assert not app.exception, app.exception
    if page == "forensics":  # the committed sample shows both diagnoses on open, labelled as sample
        shown = " ".join(m.value for m in app.markdown)
        assert "Vision failure" in shown and "Reasoning failure" in shown
        assert any("Sample data" in w.value for w in app.warning)


def test_presets_cover_the_knobs_and_scores_name_their_source() -> None:
    for name, knobs in present.PRESETS.items():
        assert set(knobs) == set(present.KNOB_DEFAULTS), name
    plain = present.difficulty_scores("tangle", {}, {"n": 10, "stroke": 8, "surface": {}})
    assert plain["reasoning"] == (100, "estimated from 10 cables")
    dialled = present.difficulty_scores("panel", {}, {"difficulty_spec": {"visual_stress": 0.8}})
    assert dialled["visual"] == (80, "set by dial")


def test_diagnosis_separates_seeing_from_reasoning() -> None:
    assert present.diagnose({})[0] == "Awaiting twin results"
    assert present.diagnose({"perceptual_clean": True})[0] == "Vision failure"
    assert present.diagnose({"text_ir_twin": False, "perceptual_clean": False})[0] == (
        "Reasoning failure"
    )
    assert present.diagnose({"palette_swap": False})[0] == "Inconclusive"
    suite = {"twins": {"text_ir_twin": {"status": "text", "answer": {"type": "int", "value": 3}}}}
    assert present.grade_twins(suite, {"text_ir_twin": "3"}) == {"text_ir_twin": True}


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
    assert present.scoreboard(result, {})[0] == {
        "who": "You",
        "accuracy": 0.0,
        "n": 1,
        "shared": None,
    }


def test_batch_command_matches_the_cli() -> None:
    args = factory.batch_command(
        "tangle", 5, Path("x"), split="ood", holdout_params=["strands=9:10"],
        difficulty=factory.difficulty_from({"reasoning_load": 0.4, "visual_stress": None}),
    )  # fmt: skip
    assert args[:3] == ["generate", "--plugin", "tangle"] and "--reasoning-load" in args
    assert "--visual-stress" not in args and factory.command_line(args).startswith("uv run knot ")


def test_committed_sample_opens_with_two_concrete_diagnoses() -> None:
    sample = REPO / "samples" / "demo_forensics"
    suites = json.loads((sample / "suites.json").read_text(encoding="utf-8"))["suites"]
    replies = present.load_twin_results(sample)
    verdicts = [
        present.diagnose(present.grade_twins(s, replies[s["bundle_id"]]))[0] for s in suites
    ]
    assert verdicts == ["Vision failure", "Reasoning failure"]
    for suite in suites:
        assert suite["sample"] and "not a model output" in " ".join(suite["replies"])
        images = [suite["image"], *(t.get("image") for t in suite["twins"].values())]
        assert all((REPO / i).is_file() for i in images if i)
