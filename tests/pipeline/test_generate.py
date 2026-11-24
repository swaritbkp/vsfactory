"""Task 2.9: `knot generate` + offline freshness receipt (SPEC 3, 8.4, 11.1)."""

from __future__ import annotations

import re
from pathlib import Path

from knotwork.knotcore.contracts import FreshnessReceipt, TaskBundle
from knotwork.pipeline.io import read_json, read_jsonl
from knotwork.plugins.panel.programs import round_half_up

ROW_KEYS = {
    "bundle_id",
    "plugin",
    "family",
    "split",
    "role",
    "base_id",
    "answer_canonical",
    "dir",
    "scene_id",
}


def test_dataset_files(datasets: Path) -> None:
    for plugin, n in (("tangle", 3), ("gauge", 2), ("panel", 6)):
        ds = datasets / f"data_{plugin}"
        index = read_jsonl(ds / "index.jsonl")
        stats = read_json(ds / "stats.json")
        receipt = FreshnessReceipt.model_validate(read_json(ds / "freshness_receipt.json"))
        assert (ds / "quarantine.jsonl").is_file()
        assert (
            sum(r["role"] == "base" for r in index)
            == n
            == stats["n_base_certified"]
            == stats["n_requested"]
        )
        assert len(index) == 3 * n == stats["n_bundles"]
        assert [r["role"] for r in index[:3]] == ["base", "causal", "nuisance"]
        assert set(stats["families_expected"]) == {
            r["family"] for r in index if r["role"] == "base"
        }
        assert all(set(r) == ROW_KEYS for r in index)
        assert receipt.source == "offline" and receipt.round == 5 and receipt.fetched_at_utc is None
        for r in index:
            task = TaskBundle.model_validate(read_json(ds / r["dir"] / "task.json"))
            assert task.canary == receipt.canary and re.fullmatch(
                r"KNOTWORK-CANARY-[0-9a-f]{32}", task.canary
            )
            assert task.freshness_receipt_id == receipt.receipt_id
            assert task.bundle_id == r["bundle_id"] and (ds / r["dir"] / "beauty.png").is_file()


def test_panel_premise_schedule(datasets: Path) -> None:
    bases = [r for r in read_jsonl(datasets / "data_panel" / "index.jsonl") if r["role"] == "base"]
    want = round_half_up(0.15 * len(bases))
    assert sum(r["family"] == "T_GASLIGHT" for r in bases) == want
    assert sum(r["family"] == "P5_PREMISE_CONTROL" for r in bases) == want


def test_cli_generate_small(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from knotwork.cli import cli

    res = CliRunner().invoke(cli, ["generate", "--plugin", "gauge", "--n", "1", "--seed", "2", "--out",
                                   str(tmp_path / "g"), "--no-twins"])  # fmt: skip
    assert res.exit_code == 0, res.output
    assert len(read_jsonl(tmp_path / "g" / "index.jsonl")) == 1
