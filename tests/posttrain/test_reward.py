"""The composite RLVR reward: grounded answers win, lucky guesses are flagged, wrong answers lose,
and the drop-in kit file scores exactly like the package."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from knotwork.posttrain import curriculum, reward, vcot
from posttrain_fixtures import FIXTURES, task


def _case(domain: str = "rack_patch") -> tuple[str, str, dict]:  # type: ignore[type-arg]
    trace = vcot.extract(FIXTURES / domain)
    extra = {"vcot_boxes_1000": trace.boxes_1000, "domain": domain}
    return trace.formatted_cot, json.dumps(task(domain)["answer"]), extra


@pytest.mark.parametrize("domain", ["rack_patch", "tangle", "gauge", "panel"])
def test_grounded_correct_answer_scores_about_one(domain: str) -> None:
    cot, truth, extra = _case(domain)
    r = reward.compute_score(f"knotwork/{domain}", cot, truth, extra)
    assert r["score"] >= 0.99 and r["outcome_score"] == 1.0
    assert r["grounding_iou"] == pytest.approx(1.0) and r["format_score"] == 1.0
    assert r["matched_boxes"] == len(extra["vcot_boxes_1000"]) and not r["reward_hack_flag"]
    assert set(r) >= {"score", "outcome_score", "grounding_iou", "format_score",
                      "abstention_score", "matched_boxes", "reward_hack_flag"}  # fmt: skip


def test_ungrounded_lucky_guess_is_flagged_and_scored_lower() -> None:
    cot, truth, extra = _case()
    answer = json.loads(truth)["canonical"]
    guess = f"<think>It is probably {answer}.</think>\n\\boxed{{{answer}}}"
    r = reward.compute_score("knotwork/rack_patch", guess, truth, extra)
    assert r["outcome_score"] == 1.0 and r["grounding_iou"] == 0.0 and r["reward_hack_flag"]
    assert r["score"] < reward.compute_score("knotwork/rack_patch", cot, truth, extra)["score"]
    assert r["effective_outcome_score"] == 0.0 and r["grounding_gated"]
    assert r["score"] == pytest.approx(0.15)  # format weight only: no positive RL advantage


def test_gate_caps_any_ungrounded_correct_answer_at_the_format_weight() -> None:
    _, truth, extra = _case()
    answer = json.loads(truth)["canonical"]
    for text in (f"\\boxed{{{answer}}}", f"ANSWER: {answer}", f"<think>sure</think>\n{answer}"):
        r = reward.compute_score("knotwork/rack_patch", text, truth, extra)
        assert r["outcome_score"] == 1.0 and r["grounding_iou"] == 0.0, text
        assert r["reward_hack_flag"] and r["score"] <= 0.15, (text, r["score"])


def test_gate_leaves_outcome_only_grading_untouched() -> None:
    _, truth, _ = _case()
    answer = json.loads(truth)["canonical"]
    r = reward.compute_score("x", f"\\boxed{{{answer}}}", truth, {"domain": "rack_patch"})
    assert not r["grounding_gated"] and r["effective_outcome_score"] == 1.0
    assert r["score"] == pytest.approx(0.8 + 0.2 * 0.5 * (0.5 + 1.0))


def test_wrong_answers_score_below_a_quarter_even_when_perfectly_grounded() -> None:
    cot, truth, extra = _case()
    answer = json.loads(truth)["canonical"]
    wrong = cot.replace(f"\\boxed{{{answer}}}", "\\boxed{99}")
    r = reward.compute_score("knotwork/rack_patch", wrong, truth, extra)
    assert r["outcome_score"] == 0.0 and r["grounding_iou"] == pytest.approx(1.0)
    assert r["score"] < 0.25 and not r["reward_hack_flag"]
    assert reward.compute_score("x", "no idea", truth, extra)["score"] < 0.25


def test_box_spraying_and_whole_image_boxes_do_not_buy_grounding() -> None:
    cot, truth, extra = _case()
    answer = json.loads(truth)["canonical"]
    whole = "<think>look <box>[0, 0, 1000, 1000]</box></think>\n\\boxed{" + answer + "}"
    assert reward.compute_score("x", whole, truth, extra)["grounding_iou"] < 0.05
    spray = cot.replace("</think>", "<box>[1, 1, 2, 2]</box>" * 40 + "</think>")
    honest = reward.compute_score("x", cot, truth, extra)["grounding_iou"]
    assert reward.compute_score("x", spray, truth, extra)["grounding_iou"] < 0.25 * honest


def test_full_image_box_cannot_bypass_the_grounding_gate() -> None:
    _, truth, extra = _case()
    answer = json.loads(truth)["canonical"]
    text = f"<think>it is here <box>[0,0,1000,1000]</box></think>\n\\boxed{{{answer}}}"
    r = reward.compute_score("knotwork/rack_patch", text, truth, extra)
    assert r["outcome_score"] == 1.0 and r["matched_boxes"] == 0 and r["pred_boxes"] == 1
    assert r["grounding_gated"] and r["reward_hack_flag"] and r["effective_outcome_score"] == 0.0
    assert r["score"] <= 0.15
    near = text.replace("[0,0,1000,1000]", "[20, 30, 990, 985]")  # ~94 % of the canvas
    r = reward.compute_score("knotwork/rack_patch", near, truth, extra)
    assert r["matched_boxes"] == 0 and r["reward_hack_flag"] and r["score"] <= 0.15


def test_one_weak_box_below_half_iou_also_triggers_the_gate() -> None:
    _, truth, extra = _case()
    answer = json.loads(truth)["canonical"]
    x0, y0, x1, y1 = (float(v) for v in extra["vcot_boxes_1000"][0])
    w = x1 - x0
    weak = (x0 + 0.6 * w, y0, x1 + 0.6 * w, y1)  # shifted right: IoU = 0.4 / 1.6 = 0.25
    assert 0.0 < reward.iou(weak, (x0, y0, x1, y1)) < 0.5
    box = "<box>[{:.1f}, {:.1f}, {:.1f}, {:.1f}]</box>".format(*weak)
    text = f"<think>here {box}</think>\n\\boxed{{{answer}}}"
    r = reward.compute_score("knotwork/rack_patch", text, truth, extra)
    assert r["outcome_score"] == 1.0 and r["grounding_iou"] > 0.0 and r["matched_boxes"] == 0
    assert r["grounding_gated"] and r["reward_hack_flag"] and r["score"] <= 0.15


def test_one_box_at_half_iou_or_better_is_not_gated() -> None:
    _, truth, extra = _case()
    answer = json.loads(truth)["canonical"]
    box = "<box>[{}, {}, {}, {}]</box>".format(*extra["vcot_boxes_1000"][0])
    r = reward.compute_score("x", f"<think>{box}</think>\n\\boxed{{{answer}}}", truth, extra)
    assert r["matched_boxes"] >= 1 and not r["grounding_gated"] and not r["reward_hack_flag"]
    assert r["effective_outcome_score"] == 1.0 and r["score"] > 0.75


def test_hedging_two_boxed_answers_gets_no_outcome() -> None:
    _, truth, extra = _case()
    answer = json.loads(truth)["canonical"]
    r = reward.compute_score("x", f"\\boxed{{{answer}}} or \\boxed{{01}}", truth, extra)
    assert r["outcome_score"] == 0.0


def test_without_reference_boxes_weights_renormalise() -> None:
    _, truth, _ = _case()
    answer = json.loads(truth)["canonical"]
    r = reward.compute_score("x", f"<think>ok</think>\n\\boxed{{{answer}}}", truth, None)
    assert r["score"] == pytest.approx(1.0) and r["weights"]["grounding"] == 0.0
    assert r["weights"]["outcome"] == pytest.approx(0.8) and not r["reward_hack_flag"]


def test_false_premise_rewards_calibrated_rejection_not_confident_hallucination() -> None:
    truth = json.dumps({"type": "premise", "value": {"premise_holds": False, "value": 67.0},
                        "canonical": "no; 67", "tolerance": 1.6})  # fmt: skip
    right = reward.compute_score("x", "<think>red, not green</think>\n\\boxed{no; 67}", truth)
    halluc = reward.compute_score("x", "<think>green</think>\n\\boxed{yes; 67}", truth)
    assert right["abstention_score"] == 1.0 and right["score"] == pytest.approx(1.0)
    assert halluc["abstention_score"] == 0.0 and halluc["score"] < 0.25


def test_rack_port_answers_are_graded_on_the_number() -> None:
    truth = json.dumps({"type": "label", "value": "07", "canonical": "07", "tolerance": 0.0})
    extra = {"domain": "rack_patch"}
    assert reward.compute_score("x", "\\boxed{PP-07}", truth, extra)["outcome_score"] == 1.0
    assert reward.compute_score("x", "\\boxed{port 7}", truth, extra)["outcome_score"] == 1.0


def test_trl_reward_unwraps_conversational_completions() -> None:
    cot, truth, extra = _case()
    got = reward.trl_reward([[{"role": "assistant", "content": cot}], "nothing"],
                            ground_truth=[truth, truth], extra_info=[json.dumps(extra), extra])  # fmt: skip
    assert got[0] >= 0.99 and got[1] < 0.25


def test_drop_in_kit_reward_is_self_contained_and_identical(tmp_path: Path) -> None:
    kit = curriculum.export_kit(tmp_path / "kit")
    source = kit["knotwork_verl_reward.py"].read_text(encoding="utf-8")
    assert not re.search(r"^\s*(from|import)\s+knotwork", source, re.M)
    assert "reward.custom_reward_function.path" in kit["run_verl_grpo.sh"].read_text()
    cot, truth, extra = _case("gauge")
    probe = (f"import json, sys; sys.path.insert(0, {str(kit['knotwork_verl_reward.py'].parent)!r});"
             "from knotwork_verl_reward import compute_score as f;"
             f"print(json.dumps(f('knotwork/gauge', {cot!r}, {truth!r}, {extra!r})))")  # fmt: skip
    out = subprocess.run([sys.executable, "-I", "-c", probe], capture_output=True, text=True,
                         check=True, timeout=30)  # fmt: skip
    assert json.loads(out.stdout) == reward.compute_score("knotwork/gauge", cot, truth, extra)
    compile(kit["run_trl_grpo.py"].read_text(encoding="utf-8"), "run_trl_grpo.py", "exec")
