"""Tasks 6.1 / 6.3: figures and the self-contained report build from a (mock) benchmark run."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

from knotwork.bench.run import run_bench
from knotwork.report.build import run_report
from knotwork.report.summary import build_summary

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "bench" / "fixtures"


def test_figures_and_report_from_mock_run(tmp_path: Path) -> None:
    art = tmp_path / "artifacts"
    rc = run_bench(FIX / "data", art / "phase5" / "bench_mock", REPO / "configs" / "models.yaml", False,
                   "mock", FIX / "replay", None, "all", 10, 0, False)  # fmt: skip
    assert rc == 0
    s = build_summary(art)
    assert (
        s is not None
        and s["source"] == "mock"
        and set(s["models"]) == {"mock_a", "mock_b", "mock_c"}
    )
    for m in s["models"].values():
        assert {"overall", "tangle", "panel", "gauge_read"} == set(m["accuracy"])
        assert {"original", "causal", "nuisance", "all_three"} <= set(m["twins"])
    out = tmp_path / "report" / "report.html"
    assert run_report(art, out) == 0
    text = out.read_text(encoding="utf-8")
    assert out.stat().st_size > 50_000 and "data:image/png;base64," in text
    assert "SIMULATED mock VLMs" in text or "mock" in text
    figs = sorted((tmp_path / "report" / "figures").glob("fig*.png"))
    assert len(figs) >= 4
    for f in figs:
        assert Image.open(f).size[0] >= 1600


def test_report_from_fresh_clone_layout(tmp_path: Path) -> None:
    """A fresh clone has no artifacts/phase5 or phase2: the report falls back to the committed
    samples/live_run and to the quickstart's `knot mutate` default (artifacts/mutate/report.json)."""
    shutil.copytree(REPO / "samples" / "live_run", tmp_path / "samples" / "live_run")
    art = tmp_path / "artifacts"
    (art / "mutate").mkdir(parents=True)
    fault = {"plugin": "tangle", "expected": "E_ID_MISMATCH", "injected": 4, "caught_exact": 4}
    (art / "mutate" / "report.json").write_text(json.dumps({
        "faults": {"M1": fault}, "clean": {"false_quarantine_rate": 0.0}, "exact_catch_rate": 1.0,
        "catch_rate": 1.0, "n_per_fault": 4}), encoding="utf-8")  # fmt: skip
    s = build_summary(art)
    assert s is not None and s["source"] == "live" and s["dir"] == "samples/live_run"
    assert s["trials"] == 997 and len(s["models"]) == 3
    out = tmp_path / "report" / "report.html"
    assert run_report(art, out) == 0
    text = out.read_text(encoding="utf-8")
    assert "No benchmark run found" not in text and "No mutate report found" not in text
    assert "openai/gpt-5.4" in text and "live OpenRouter run" in text and "E_ID_MISMATCH" in text
    assert out.stat().st_size > 200_000 and text.count("data:image/png;base64,") >= 4
