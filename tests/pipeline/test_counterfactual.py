"""Counterfactual miner: failure loading, scene transforms and the committed live failures."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from knotwork.knotcore.contracts import Family
from knotwork.knotcore.seeds import SeedTree
from knotwork.pipeline import counterfactual as cf
from knotwork.pipeline import text_ir
from knotwork.plugins.payloads import validate_payload
from knotwork.plugins.registry import get_plugin
from knotwork.plugins.tangle import braid

REPO = Path(__file__).resolve().parents[2]
SAMPLES = REPO / "samples" / "live_failures"
PROFILE: dict[str, Any] = yaml.safe_load(
    (REPO / "configs" / "default_profile.yaml").read_text(encoding="utf-8")
)


def test_transforms_keep_logic_and_step_down_removes_one_crossing() -> None:
    plugin = get_plugin("tangle")
    at = SeedTree(11).child("tangle", "cf")
    scene = plugin.sample_scene(at, dict(PROFILE, _family="T1_TRACE"))
    question = plugin.questions(scene, at.child("q"), Family.T1_TRACE)
    for edit in (cf.perceptual_clean, cf.palette_swap):
        pay = copy.deepcopy(scene.payload)
        edit("tangle", pay)
        validate_payload("tangle", pay)
        twin = scene.model_copy(update={"payload": pay})
        assert pay["word"] == scene.payload["word"]
        assert plugin.symbolic_answer(twin, question) == plugin.symbolic_answer(scene, question)
    pay = copy.deepcopy(scene.payload)
    letter = cf.step_down_candidates("tangle", pay, question)[0]
    cf.remove_crossing("tangle", pay, letter)
    before = braid.compose(scene.payload["n"], scene.payload["word"])[1]
    after = braid.compose(pay["n"], [tuple(x) if x else None for x in pay["word"]])[1]
    assert len(after) == len(before) - 1
    text = text_ir.from_scene(scene, question.text)
    assert text.endswith(question.text) and "exchange rows" in text


def test_committed_live_failures_load_and_only_the_gauge_gets_a_text_twin(tmp_path: Path) -> None:
    items = cf.load_failures(SAMPLES)
    assert len(items) == 3 and all(len(f.replies) == 3 for f in items)
    bundles = cf.index_bundles(SAMPLES)
    assert {f.bundle_id for f in items} <= set(bundles)
    assert cf.run_counterfactual(SAMPLES, SAMPLES, tmp_path) == 0
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "unavailable" in report and (tmp_path / "report.html").is_file()
    rows = (tmp_path / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 and "needle points at" in rows[0]


def test_trials_file_skips_correct_rows(tmp_path: Path) -> None:
    f = tmp_path / "trials.jsonl"
    f.write_text(
        '{"key": "t1:kw-00000000000000000001", "slot": "a", "reply": "3", "correct": false}\n'
        '{"key": "t1:kw-00000000000000000002", "slot": "a", "reply": "4", "correct": true}\n',
        encoding="utf-8",
    )
    (only,) = cf.load_failures(f)
    assert only.bundle_id == "kw-00000000000000000001" and only.replies == {"a": "3"}
