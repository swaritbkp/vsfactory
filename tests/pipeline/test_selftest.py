"""Task 3.7: golden selftest -- the committed golden reproduces exactly and reports are stable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from knotwork.pipeline.io import read_json, read_jsonl
from knotwork.pipeline.selftest import PER_FAMILY, PLUGINS, _diff, run_selftest
from knotwork.plugins.registry import get_plugin

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "tests" / "golden" / "golden.json"


def test_golden_file_shape() -> None:
    g = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert g["seed"] == 1234 and g["per_family"] == PER_FAMILY
    for name in PLUGINS:
        rows = g["plugins"][name]
        fams = [f.value for f in get_plugin(name).families]
        bases = [r for r in rows if r["role"] == "base"]
        assert sorted(r["family"] for r in bases) == sorted(
            f for f in fams for _ in range(PER_FAMILY)
        )
        assert len(rows) == 3 * len(bases)  # base + causal + nuisance per slot
        assert all(r["verdict"] == "CERTIFIED" and r["bundle_id"].startswith("kw-") for r in rows)


def test_diff_reports_first_mismatch() -> None:
    a = [{"family": "G1_READ", "role": "base", "bundle_id": "kw-1", "answer": "3"}]
    assert _diff(a, a) == ""
    b = [dict(a[0], answer="4")]
    assert "answer" in _diff(a, b)
    assert "golden has 1" in _diff(a, [])


@pytest.mark.timeout(3600)
def test_selftest_reproduces_golden(tmp_path: Path) -> None:
    out = tmp_path / "selftest"
    assert run_selftest(GOLDEN, False, out_dir=out) == 0
    rep = read_json(out / "report.json")
    assert rep["ok"] is True
    names = {c["name"] for c in rep["checks"]}
    assert {"seeds_kat", "kernel_sha_parity", "verify", *(f"golden:{p}" for p in PLUGINS)} <= names
    first = (out / "report.json").read_bytes()
    index = read_jsonl(out / "data" / "index.jsonl")
    assert len(index) == sum(len(json.loads(GOLDEN.read_text())["plugins"][p]) for p in PLUGINS)
    assert all((out / "data" / r["dir"] / "task.json").is_file() for r in index)
    # the report carries no timings: re-writing it from the same run is byte-identical
    assert b"seconds" not in first


def test_cross_platform_rows_compare_semantics_only() -> None:
    from knotwork.pipeline.selftest import RASTER_KEYS, _strip

    a = [{"family": "T1_TRACE", "role": "base", "answer": "3", "verdict": "CERTIFIED",
          "bundle_id": "kw-1", "id_buffer": "x", "masks": "y"}]  # fmt: skip
    b = [dict(a[0], bundle_id="kw-2", id_buffer="z", masks="w")]
    assert _diff(_strip(a), _strip(b)) == "" and _diff(a, b) != ""
    assert _diff(_strip(a), _strip([dict(b[0], answer="4")])) != ""
    assert set(RASTER_KEYS) == {"bundle_id", "id_buffer", "masks"}
