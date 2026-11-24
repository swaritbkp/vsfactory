#!/usr/bin/env python3
"""Rebuild the offline bench fixtures (Task 5.5). Offline only; no API key, no network.

* ``tests/bench/fixtures/data/{tangle,panel}`` -- a small split=test dataset from `knot generate`
  (T1 items; P1 + T_GASLIGHT items). Masks and id buffers are dropped: the bench reads beauty.png,
  task.json, claims.json and manifest.json only (derived items are re-rendered from seed paths).
* ``tests/bench/fixtures/replay/<slot>/<key>.json`` -- the mock VLMs' answers for the exact requests
  of ``knot bench --data tests/bench/fixtures/data --client replay --seed 0 --max-trials 10``, written
  in replay format under the live slot names.

    uv run python scripts/build_bench_fixtures.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

FIX = REPO / "tests" / "bench" / "fixtures"
PROFILE = REPO / "configs" / "default_profile.yaml"
SETS = (("tangle", 6, ["T1_TRACE"]), ("panel", 6, ["P1_READ_TRACE", "T_GASLIGHT"]))


def main() -> int:
    from knotwork.bench.run import run_bench
    from knotwork.pipeline.generate import run_generate

    data, replay = FIX / "data", FIX / "replay"
    shutil.rmtree(data, ignore_errors=True)
    shutil.rmtree(replay, ignore_errors=True)
    for plugin, n, fams in SETS:
        out = data / plugin
        rc = run_generate(plugin, n, out, 3, "offline", PROFILE, "test", False, fams, 0)
        if rc != 0:
            return rc
        for b in (out / "bundles").iterdir():
            shutil.rmtree(b / "masks")
            (b / "id_buffer.png").unlink()
        (out / "quarantine.jsonl").write_text("", encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        rc = run_bench(data, Path(tmp) / "bench", REPO / "configs" / "models.yaml", False, "record-mock",
                       replay, None, "all", 10, 0, False)  # fmt: skip
    print(f"fixtures: {sum(1 for _ in replay.rglob('*.json'))} replay entries")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
