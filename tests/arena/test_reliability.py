"""pass@1 / pass@k / pass^k reliability (ZeroBench), the repeat-aware trial key, backwards
compatibility with results written before repeats, and the literature references in the report.
Pure checks: no rendering, no runner process."""

from __future__ import annotations

from pathlib import Path

from arena_fixtures import config
from knotwork.arena import report, runner, storage
from knotwork.arena.schemas import RunStatus, SessionConfig, TrialRecord, trial_key

MODEL = "openai/gpt-5.4"
CITED = ("arXiv:2502.09696", "arXiv:2407.06581", "arXiv:2210.02410", "arXiv:2406.08702",
         "2026.emnlp-main.760", "arXiv:2505.23977", "Wilson")  # fmt: skip


def _trials(pattern: dict[int, list[bool]], block: str = "b1") -> dict[str, TrialRecord]:
    out = {}
    for item, results in pattern.items():
        for r, ok in enumerate(results):
            tid = trial_key(block, item, MODEL, r)
            out[tid] = TrialRecord(trial_id=tid, block_id=block, item_index=item, bundle_id="kw-x",
                                   model=MODEL, repeat_idx=r, status="ok", correct=ok)  # fmt: skip
    return out


def _cfg(root: Path, repeats: int) -> SessionConfig:
    cfg = config(root, repeats=repeats)
    return cfg.model_copy(update={"default_models": [MODEL], "blocks": [
        cfg.blocks[0].model_copy(update={"n": 4, "models": None})]})  # fmt: skip


def test_pass_at_k_counts_any_and_all() -> None:
    got = report.pass_at_k({0: [True, True, True], 1: [True, False, False],
                            2: [False, False, False], 3: [True, True, False],
                            4: [True]}, 3)  # fmt: skip
    assert (got["items"], got["items_partial"]) == (4, 1)  # item 4 has 1 of 3 repeats: left out
    assert got["pass_at_1"] == 6 / 12
    assert (got["pass_at_k"], got["pass_hat_k"]) == (3 / 4, 1 / 4)
    assert (got["solved_any"], got["solved_all"]) == (3, 1)
    empty = report.pass_at_k({}, 2)
    assert empty["pass_at_1"] is None and empty["pass_hat_k"] is None


def test_trial_key_keeps_the_pre_repeats_form_for_repeat_zero() -> None:
    assert trial_key("b1", 3, MODEL) == f"b1:3:{MODEL}" == trial_key("b1", 3, MODEL, 0)
    keys = {trial_key("b1", 3, MODEL, r) for r in range(5)}
    assert len(keys) == 5 and f"b1:3:{MODEL}#4" in keys


def test_rows_written_before_repeats_still_load(tmp_path: Path) -> None:
    old = {"trial_id": f"b1:0:{MODEL}", "block_id": "b1", "item_index": 0, "bundle_id": "kw-x",
           "model": MODEL, "status": "ok", "correct": True}  # fmt: skip
    assert TrialRecord.model_validate(old).repeat_idx == 0
    legacy = config(tmp_path).model_dump(mode="json")
    legacy.pop("repeats")
    cfg = SessionConfig.model_validate(legacy)
    assert cfg.repeats == 1 and report.reliability(cfg, {}) is None
    assert config(tmp_path, repeats=3).total_trials == 3 * cfg.total_trials


def test_reliability_reports_a_material_gap(tmp_path: Path) -> None:
    flaky = {0: [True, False, True], 1: [True, True, True], 2: [False, True, False],
             3: [False, False, False]}  # fmt: skip
    rel = report.reliability(_cfg(tmp_path, 3), _trials(flaky))
    assert rel is not None and rel["k"] == 3
    (cell,) = rel["cells"]
    assert (cell["pass_at_k"], cell["pass_hat_k"]) == (3 / 4, 1 / 4)
    assert abs(cell["gap"] - 0.5) < 1e-12 and len(rel["findings"]) == 1
    assert "75.0%" in rel["findings"][0] and "25.0%" in rel["findings"][0]
    steady = {i: [i % 2 == 0] * 3 for i in range(4)}  # same answer every repeat: no gap
    assert report.reliability(_cfg(tmp_path, 3), _trials(steady))["findings"] == []  # type: ignore[index]


def test_reports_carry_reliability_and_references(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, 2)
    trials = _trials({0: [True, False], 1: [True, True]})
    m = report.compute_metrics(cfg, [], trials, RunStatus(),
                               prov={"git_sha": "abc", "kernel_sha256": "def"})  # fmt: skip
    assert m["reliability"]["cells"][0]["pass_hat_k"] == 0.5
    html, md = report.render_html(m, tmp_path), report.render_md(m)
    for text in (html, md):
        assert "Methodology" in text and "Literature References" in text
        assert "pass^k" in text and all(ref in text for ref in CITED)
    assert "repeat_idx" in report.CSV_FIELDS


def test_mock_run_with_repeats_writes_each_repeat_once_and_resumes(
    arena_root: Path, frozen
) -> None:  # type: ignore[no-untyped-def]
    cfg = config(arena_root, "Repeats", repeats=2)
    results = storage.resolve(cfg.results_dir)
    storage.write_session(cfg)
    assert runner.Runner(cfg, results, sleep=lambda s: None).run() == 0
    trials = storage.load_trials(results)
    assert len(trials) == cfg.total_trials
    assert {t.repeat_idx for t in trials.values()} == {0, 1}
    assert all(t.trial_id == trial_key(t.block_id, t.item_index, t.model, t.repeat_idx)
               for t in trials.values())  # fmt: skip
    lines = (results / storage.RESULTS).read_text(encoding="utf-8").splitlines()
    assert runner.Runner(cfg, results, sleep=lambda s: None).run() == 0
    assert (results / storage.RESULTS).read_text(encoding="utf-8").splitlines() == lines
