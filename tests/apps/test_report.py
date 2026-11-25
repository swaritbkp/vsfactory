"""Tasks 6.1 / 6.3: figures and the self-contained report build from a (mock) benchmark run."""

from __future__ import annotations

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
