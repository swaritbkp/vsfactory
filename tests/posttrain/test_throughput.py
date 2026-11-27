"""Throughput math, and `knot bench-throughput` end to end with the renderer stubbed by a fixture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from knotwork.arena import generation
from knotwork.arena.schemas import ItemRecord
from knotwork.cli import cli
from knotwork.posttrain import throughput
from posttrain_fixtures import bundle_copy, task


def test_unit_economics_formula() -> None:
    e = throughput.unit_economics([1000.0, 1000.0], [500.0, 500.0], 1, [100.0], vcpu_usd_hr=0.04)
    assert e["busy_s"] == pytest.approx(3.1) and e["s_per_certified"] == pytest.approx(3.1)
    assert e["certified_items_per_hour"] == pytest.approx(3600 / 3.1, rel=1e-4)
    assert e["usd_per_1m_certified"] == pytest.approx(1e6 * 3.1 / 3600 * 0.04, abs=0.01)
    assert e["yield"] == 0.5 and e["includes_vcot"] and "vcpu_usd_per_hour" in e["formula"]
    none = throughput.unit_economics([10.0], [10.0], 0)
    assert none["usd_per_1m_certified"] is None and none["certified_items_per_hour"] is None


def _fake_item(block_id: str, domain: str, preset: str, index: int, seed: int,
               data_dir: Path, certify_attempts: int = 3) -> ItemRecord:  # fmt: skip
    rel = f"{block_id}/{index:05d}/a0"
    bundle_copy(domain, data_dir / rel)
    t = task(domain)
    return ItemRecord(block_id=block_id, item_index=index, seed=seed, domain=domain, preset=preset,
                      family=t["family"], certified=index % 2 == 0, dir=rel, gen_ms=200.0,
                      verify_ms=100.0)  # fmt: skip


def test_bench_throughput_cli_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generation, "make_item", _fake_item)
    out = tmp_path / "t.json"
    res = CliRunner().invoke(cli, ["bench-throughput", "--domain", "gauge", "--n", "4",
                                   "--vcpu-usd-hr", "0.08", "--json", str(out)])  # fmt: skip
    assert res.exit_code == 0, res.output
    assert "USD per 1M certified samples @ $0.08/vCPU-hr" in res.output and "formula:" in res.output
    data = json.loads(out.read_text())
    e = data["economics"]
    assert data["domain"] == "gauge" and data["n"] == 4 and e["certified"] == 2
    assert data["seed_range"] == [throughput.SEED_BASE, throughput.SEED_BASE + 3]
    assert all(r["vcot_ms"] > 0 for r in data["items"] if r["certified"])
    vcot = sum(r["vcot_ms"] for r in data["items"] if r["certified"])
    per = (4 * 300.0 + vcot) / 1000 / 2
    assert e["s_per_certified"] == pytest.approx(per, abs=1e-4)
    assert e["usd_per_1m_certified"] == pytest.approx(1e6 * per / 3600 * 0.08, abs=0.01)
