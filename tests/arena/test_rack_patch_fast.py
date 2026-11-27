"""Fast rack patch-bay yield check for tier1: 3 renders through the unmodified kernel, each certified
on the first attempt with no E_* code. The full 10-render >= 80 % yield gate and the closed loop to
Forensics live in `test_rack_patch.py` (tier2, nightly)."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from knotwork.arena.generation import make_item_job
from knotwork.arena.schemas import ItemRecord
from knotwork.pipeline.io import read_json
from knotwork.plugins.registry import available, get_plugin

#: (preset, seed) pairs covering both families and the CVD palette
CASES = (("Clean", 500), ("Hard logic", 501), ("CVD", 502))


def test_rack_is_a_discovered_plugin_with_trace_and_cross() -> None:
    assert "rack_patch" in available()
    assert [f.value for f in get_plugin("rack_patch").families] == ["T1_TRACE", "T2_CROSS"]


def test_rack_fast_first_attempt_yield(tmp_path: Path) -> None:
    jobs = [{"block_id": "rack", "domain": "rack_patch", "preset": preset, "index": k,
             "seed": seed, "data_dir": str(tmp_path)} for k, (preset, seed) in enumerate(CASES)]  # fmt: skip
    workers = max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(workers) as pool:
        items = [ItemRecord.model_validate(r) for r in pool.map(make_item_job, jobs)]
    assert all(i.certified and i.attempts == 1 for i in items), [
        (i.preset, i.attempts, i.codes) for i in items
    ]
    for item in items:
        task = read_json(tmp_path / item.dir / "task.json")
        assert task["certificate"]["verdict"] == "CERTIFIED" and task["certificate"]["codes"] == []
        assert task["plugin"] == "rack_patch" and item.codes == []
        n = read_json(tmp_path / item.dir / "scene.json")["payload"]["n"]
        assert 8 <= n <= 14
    assert {i.family for i in items} == {"T1_TRACE", "T2_CROSS"}
