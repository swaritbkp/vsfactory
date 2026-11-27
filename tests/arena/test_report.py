"""The deterministic report: exact statistics, taxonomy, timing, inline SVG and every file of the
audit bundle, with no secret inside."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from arena_fixtures import SECRET, config
from knotwork.arena import report, runner, storage
from knotwork.arena.schemas import TrialRecord


def _session(arena_root: Path):  # type: ignore[no-untyped-def]
    cfg = config(arena_root, client="Acme Robotics", run_id="run123")
    results = storage.resolve(cfg.results_dir)
    storage.write_session(cfg)
    assert runner.Runner(cfg, results, sleep=lambda s: None).run() == 0
    return cfg, results


def test_metrics_are_exact_and_complete(arena_root: Path, frozen) -> None:  # type: ignore[no-untyped-def]
    cfg, results = _session(arena_root)
    data, items = frozen
    trials = storage.load_trials(results)
    m = report.compute_metrics(cfg, items, trials, storage.read_status(results),
                               prov={"git_sha": "abc", "kernel_sha256": "def"})  # fmt: skip
    for cell in m["leaderboard"]["cells"]:
        group = [t for t in trials.values()
                 if t.block_id == cell["block_id"] and t.model == cell["model"]]  # fmt: skip
        k = sum(bool(t.correct) for t in group)
        assert (cell["n"], cell["correct"]) == (len(group), k)
        assert (cell["ci_lo"], cell["ci_hi"]) == report.wilson(k, len(group))
    contrast = m["contrasts"]
    assert contrast == []  # b2 is answered by a model b1 never sees: no shared model, no contrast
    f = m["factory"]["overall"]
    assert f["items"] == 10 and f["yield"] == 1.0
    assert f["gen_ms"]["p50"] == report.percentile([i.gen_ms for i in items], 50)
    assert set(m["taxonomy"]["categories"]) == set(report.CATEGORIES)
    assert m["header"]["dataset_sha256"] == storage.dataset_fingerprint(items)
    assert m["header"]["client"] == "Acme Robotics" and m["header"]["simulated"]


def test_cross_block_contrasts_use_both_tests(arena_root: Path, frozen) -> None:  # type: ignore[no-untyped-def]
    cfg = config(arena_root, "Shared models").model_copy(
        update={
            "blocks": [b.model_copy(update={"models": None}) for b in config(arena_root).blocks]
        }
    )
    trials: dict[str, TrialRecord] = {}
    for block, n, right in (("b1", 6, 5), ("b2", 4, 1)):
        for k in range(n):
            for model in cfg.default_models:
                tid = f"{block}:{k}:{model}"
                trials[tid] = TrialRecord(trial_id=tid, block_id=block, item_index=k,
                                          bundle_id=f"kw-{k}", model=model, status="ok",
                                          correct=k < right, latency_ms=100.0 * (k + 1))  # fmt: skip
    rows = report.contrasts(cfg, trials)
    assert len(rows) == 2
    row = rows[0]
    assert round(row["diff_pp"], 1) == round(100 * (5 / 6 - 1 / 4), 1)
    assert row["p_fisher"] == report.fisher_exact(5, 6, 1, 4)
    assert row["p_z"] == report.two_proportion_z(5, 6, 1, 4)[1]
    tax = report.taxonomy(cfg, {}, trials)
    assert tax["models"][cfg.default_models[0]]["hardest_accuracy"] == 0.25


def test_export_writes_every_deliverable_without_secrets(arena_root: Path, frozen) -> None:  # type: ignore[no-untyped-def]
    cfg, results = _session(arena_root)
    other = config(arena_root, "Baseline session")
    storage.write_session(other)
    runner.Runner(other, storage.resolve(other.results_dir), sleep=lambda s: None).run()
    compare = (other, storage.load_trials(storage.resolve(other.results_dir)))
    paths = report.export(cfg, compare, brief="Paragraph one.\n\nParagraph two.",
                          prov={"git_sha": "abc", "kernel_sha256": "def"})  # fmt: skip
    out = results / "reports"
    for name in ("report.html", "report.md", "report_metrics.json", "trials.csv"):
        assert (out / name).is_file()
    html = (out / "report.html").read_text(encoding="utf-8")
    assert (
        html.count("<svg") == 2 and "<script" not in html and "http" not in html.split("<style>")[0]
    )
    assert "95% Wilson CI" in html and "Pareto" in html and "© 2026 BilotaAI" in html
    assert report.BRIEF_TITLE in html and "Session regression diff" in html
    assert "Simulated run" in html and "data:image/jpeg;base64," in html  # failure gallery
    metrics = json.loads((out / "report_metrics.json").read_text(encoding="utf-8"))
    assert metrics["regression"]["rows"] and metrics["ai_brief"]["title"] == report.BRIEF_TITLE
    csv_lines = (out / "trials.csv").read_text(encoding="utf-8").splitlines()
    assert (
        csv_lines[0].startswith("trial_id,block_id,domain")
        and len(csv_lines) == cfg.total_trials + 1
    )
    with zipfile.ZipFile(paths["zip"]) as zf:
        assert paths["zip"].name == f"{cfg.slug}_audit_bundle.zip"
        assert set(zf.namelist()) == {"report.html", "report.md", "report_metrics.json",
                                      "trials.csv", "campaign_spec.json"}  # fmt: skip
        assert all(SECRET not in zf.read(n).decode() for n in zf.namelist())
    md = (out / "report.md").read_text(encoding="utf-8")
    assert md.startswith("# Model Arena Report") and "| Block | Model |" in md


def test_export_refuses_a_secret(arena_root: Path, frozen) -> None:  # type: ignore[no-untyped-def]
    cfg, _ = _session(arena_root)
    try:
        report.export(cfg, brief=f"leaked {SECRET}", prov={})
    except ValueError as exc:
        assert "secret" in str(exc)
    else:
        raise AssertionError("a secret in the brief must stop the export")
