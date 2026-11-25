"""Task 5.5: `knot bench` in mock, replay and `--live --dry-run` modes on the committed fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from knotwork import cli as K
from knotwork.bench.run import run_bench

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "bench" / "fixtures"
MODELS = REPO / "configs" / "models.yaml"
ROW_KEYS = {"staircase", "per_crossing", "color_ablation", "track_switch", "composition", "twins",
            "gaslight", "parse_fail_rate"}  # fmt: skip


@pytest.fixture(scope="module")
def replay_report(tmp_path_factory: pytest.TempPathFactory) -> dict:
    out = tmp_path_factory.mktemp("replay")
    rc = run_bench(
        FIX / "data", out, MODELS, False, "replay", FIX / "replay", None, "all", 10, 0, False
    )
    assert rc == 0
    return json.loads((out / "bench_report.json").read_text(encoding="utf-8"))


def test_fixtures_present() -> None:
    assert (FIX / "data" / "tangle" / "index.jsonl").is_file()
    assert (FIX / "data" / "panel" / "freshness_receipt.json").is_file()
    assert {p.name for p in (FIX / "replay").iterdir()} == {"vlm_a", "vlm_b", "vlm_c"}


def test_replay_is_offline_and_complete(replay_report: dict) -> None:
    rep = replay_report
    assert rep["mode"] == "replay" and rep["network_calls"] == 0 and rep["cost_usd"] == 0.0
    assert set(rep["models"]) == {"vlm_a", "vlm_b", "vlm_c"}
    for row in rep["models"].values():
        assert set(row) >= ROW_KEYS
        assert row["n_missing"] == 0 and row["n_trials"] > 0
        assert set(row["staircase"]) == {"tangle", "panel"}
        assert row["staircase"]["tangle"]["expected_level"] is None  # unknown truth outside mock
        assert row["parse_fail_rate"] == 0.0


def test_replay_reproduces_recorded_mock(tmp_path: Path, replay_report: dict) -> None:
    """The replayed run equals a fresh mock-recorded run (same answers, same statistics)."""
    out = tmp_path / "rec"
    rc = run_bench(
        FIX / "data",
        out,
        MODELS,
        False,
        "record-mock",
        tmp_path / "replay",
        None,
        "all",
        10,
        0,
        False,
    )
    assert rc == 0
    rec = json.loads((out / "bench_report.json").read_text(encoding="utf-8"))

    def timeless(models: dict) -> dict:
        return {k: {kk: vv for kk, vv in v.items() if kk != "seconds"} for k, v in models.items()}

    assert timeless(rec["models"]) == timeless(replay_report["models"])


def test_mock_mode_report(tmp_path: Path) -> None:
    out = tmp_path / "mock"
    res = CliRunner().invoke(K.cli, ["bench", "--data", str(FIX / "data"), "--out", str(out), "--client", "mock",
                                     "--max-trials", "10", "--seed", "2"])  # fmt: skip
    assert res.exit_code == 0, res.output
    rep = json.loads((out / "bench_report.json").read_text(encoding="utf-8"))
    assert rep["mode"] == "mock" and rep["network_calls"] == 0
    assert set(rep["models"]) == {"mock_a", "mock_b", "mock_c"}
    for row in rep["models"].values():
        assert set(row) >= ROW_KEYS
        for st in row["staircase"].values():
            assert st["expected_level"] is not None and st["trials"] <= 10
        assert {"p_hat", "ci", "n"} == set(row["per_crossing"])
    assert (out / "trials.jsonl").read_text(encoding="utf-8").count("\n") > 0


def test_live_dry_run_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO)
    out = tmp_path / "pre"
    res = CliRunner().invoke(K.cli, ["bench", "--data", str(FIX / "data"), "--out", str(out), "--live",
                                     "--budget-usd", "8", "--dry-run"])  # fmt: skip
    assert res.exit_code == 0, res.output
    pre = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
    assert pre["network_calls"] == 0 and 0 < pre["estimated_usd"] <= 8.0
    assert pre["calls"] == 3 * pre["calls_per_slot"]["vlm_a"] and pre["budget_usd"] == 8.0
    assert set(pre["slugs"]) == {"vlm_a", "vlm_b", "vlm_c"}


def test_live_refused_without_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # no .checkpoints/CP2_APPROVED here
    res = CliRunner().invoke(K.cli, ["bench", "--data", str(FIX / "data"), "--out", str(tmp_path / "x"), "--live",
                                     "--budget-usd", "5", "--models", str(MODELS)])  # fmt: skip
    assert res.exit_code == 4 and "CHECKPOINT" in res.output
