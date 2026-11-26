"""Rack patch-bay yield gate: 10 items through the unmodified kernel, >= 80 % certified on the first
attempt with no E_* code on a certified bundle; plus the closed loop to Forensics."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from knotwork.arena import forensics as arena_forensics
from knotwork.arena.generation import make_item_job
from knotwork.arena.schemas import ItemRecord
from knotwork.pipeline.io import read_json
from knotwork.plugins.registry import available, get_plugin

PRESETS = ("High stress", "Hard logic", "Clean", "CVD", "High stress")


@pytest.fixture(scope="module")
def rack_items(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[ItemRecord]]:
    data = tmp_path_factory.mktemp("rack")
    jobs = [{"block_id": "rack", "domain": "rack_patch", "preset": PRESETS[k % len(PRESETS)],
             "index": k, "seed": 500 + k, "data_dir": str(data)} for k in range(10)]  # fmt: skip
    workers = max(1, min(5, (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(workers) as pool:
        items = [ItemRecord.model_validate(r) for r in pool.map(make_item_job, jobs)]
    return data, items


def test_rack_is_a_discovered_plugin_with_trace_and_cross() -> None:
    assert "rack_patch" in available()
    assert [f.value for f in get_plugin("rack_patch").families] == ["T1_TRACE", "T2_CROSS"]


def test_rack_first_attempt_yield_is_at_least_80_percent(rack_items) -> None:  # type: ignore[no-untyped-def]
    data, items = rack_items
    first_try = [i for i in items if i.certified and i.attempts == 1]
    assert len(first_try) >= 8, [(i.preset, i.codes) for i in items]
    for item in (i for i in items if i.certified):
        task = read_json(data / item.dir / "task.json")
        assert task["certificate"]["verdict"] == "CERTIFIED" and task["certificate"]["codes"] == []
        assert task["plugin"] == "rack_patch" and item.codes == []
        assert (data / item.dir / "certificate.json").is_file()
    dense = [read_json(data / i.dir / "scene.json")["payload"]["n"] for i in items if i.certified]
    assert min(dense) >= 8 and max(dense) <= 14
    t1 = next(i for i in items if i.certified and i.family == "T1_TRACE")
    assert "top switch port SW-" in t1.question and len(t1.truth) == 2


def test_rack_failure_builds_a_twin_suite(rack_items, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    data, items = rack_items
    item = next(i for i in items if i.certified and i.family == "T1_TRACE")
    out = arena_forensics.build_suite(item, {"mock/model": "ANSWER: 99"}, data, tmp_path / "fx")
    suite = read_json(out / "suites.json")["suites"][0]
    assert suite["plugin"] == "rack_patch" and suite["replies"] == {"mock/model": "ANSWER: 99"}
    assert set(suite["twins"]) == {"perceptual_clean", "palette_swap", "reasoning_step_down",
                                   "text_ir_twin"}  # fmt: skip
    assert suite["twins"]["text_ir_twin"]["status"] == "text"
    assert suite["twins"]["perceptual_clean"]["status"] in ("certified", "quarantined")
