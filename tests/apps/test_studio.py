"""The Studio's pages render on the live API, and its pure helpers behave."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from knotwork.pipeline.io import append_jsonl, write_json
from knotwork.studio import calibration, factory, guide, present

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def app() -> AppTest:
    return AppTest.from_file(str(REPO / "apps" / "inspector.py"), default_timeout=60).run()


def _footer_shown(app: AppTest) -> bool:
    return any("BilotaAI" in c.value and "Team Bilota" in c.value for c in app.caption)


def test_studio_opens_on_a_certified_tangle(app: AppTest) -> None:
    assert not app.exception, app.exception
    item = app.session_state["item"]
    assert item["certificate"]["verdict"] == "CERTIFIED" and item["task"] is not None
    assert (item["plugin"], item["family"]) == ("tangle", "T1_TRACE")
    assert [m.label for m in app.metric][:2] == ["Certified answer", "Plugin / family"]
    assert _footer_shown(app)


def test_presets_have_short_single_row_labels() -> None:
    assert list(present.PRESETS) == ["Clean", "Hard logic", "High stress", "CVD"]


@pytest.mark.parametrize("page", ["forensics", "human_eval", "export"])
def test_other_pages_render(app: AppTest, page: str) -> None:
    app.switch_page(f"app_pages/{page}.py").run()
    assert not app.exception, app.exception
    assert _footer_shown(app)
    if (
        page == "forensics"
    ):  # the committed sample offers both diagnoses on open, labelled as sample
        options = " ".join(o for group in app.button_group for o in group.options)
        assert "Sample 1: Vision failure" in options and "Sample 2: Reasoning failure" in options
        assert any("Sample data" in w.value for w in app.warning)
    if page == "human_eval":  # model baselines show before anyone answers
        assert app.metric[0].label == "Your session" and app.metric[0].value == "0 answered"
        board = app.dataframe[0].value
        assert {"GPT-5.4", "Gemini 2.5 Flash", "Qwen3-VL-235B"} <= set(board["Model"])
        assert abs(board.set_index("Model").loc["GPT-5.4", "Accuracy"] - 25.2) < 0.1


def test_human_eval_rejects_an_empty_answer(app: AppTest) -> None:
    app.switch_page("app_pages/human_eval.py").run()
    assert not app.exception, app.exception
    if app.text_input and any(t.label == "Your answer" for t in app.text_input):
        next(t for t in app.text_input if t.label == "Your answer").set_value("   ")
    submit = next(b for b in app.button if b.label == "Submit")
    submit.click().run()
    assert not app.exception, app.exception
    assert any(w.value == "Enter an answer before submitting." for w in app.warning)
    assert "last_result" not in app.session_state


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


@pytest.mark.parametrize("guided", [True, False])
@pytest.mark.parametrize("page", ["studio", "arena", "forensics", "human_eval", "export"])
def test_pages_render_with_guided_mode_on_and_off(app: AppTest, page: str, guided: bool) -> None:
    app.switch_page(f"app_pages/{page}.py")
    app.session_state["guided_mode"] = guided
    app.run()
    assert not app.exception, app.exception
    assert os.environ["KNOTWORK_NETWORK"] == "deny"
    step = guide.tour_index(page) + 1
    strip = [m.value for m in app.markdown if f"Step {step} of 5" in m.value]
    assert bool(strip) is guided
    assert app.toggle(key="guided_mode").value is guided
    assert any(b.label == "Quick tour" for b in app.button) and _footer_shown(app)


def _protos(app: AppTest) -> str:
    out: list[str] = []

    def walk(node: object) -> None:
        for child in getattr(node, "children", {}).values():
            out.append(str(getattr(child, "proto", "")))
            walk(child)

    walk(app.main)
    return " ".join(out)


def test_vcot_steps_view_draws_the_grounded_trace(app: AppTest) -> None:
    app.switch_page("app_pages/studio.py").run()
    app.segmented_control(key="view").set_value("vCoT Steps").run()
    assert not app.exception, app.exception
    assert os.environ["KNOTWORK_NETWORK"] == "deny"
    panel = next(e for e in app.expander if e.label == "Grounded Visual CoT (SFT trace)")
    trace = panel.code[0].value
    assert trace.startswith("<think>\nStep 1 (perception):") and "<box>[" in trace
    assert trace.endswith("\\boxed{" + app.session_state["item"]["answer"] + "}")
    assert "Reasoning" in panel.dataframe[0].value.columns
    assert "grounded steps" in _protos(app)
