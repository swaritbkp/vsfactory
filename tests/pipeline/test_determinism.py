"""Task 2.10: determinism under parallelism (SPEC 1.5, 5)."""

from __future__ import annotations

from pathlib import Path

from knotwork.pipeline.generate import run_generate
from knotwork.pipeline.io import read_jsonl

PROFILE = Path(__file__).resolve().parents[2] / "configs" / "default_profile.yaml"


def _ids(out: Path, seed: int, workers: int) -> list[str]:
    assert (
        run_generate("tangle", 3, out, seed, "offline", PROFILE, "dev", False, None, workers) == 0
    )
    return [r["bundle_id"] for r in read_jsonl(out / "index.jsonl")]


def test_same_seed_same_ids_any_workers(tmp_path: Path) -> None:
    a = _ids(tmp_path / "a", 3, 1)
    b = _ids(tmp_path / "b", 3, 2)
    assert a == b and len(a) == 3


def test_different_seed_disjoint(tmp_path: Path) -> None:
    a = _ids(tmp_path / "a", 3, 2)
    c = _ids(tmp_path / "c", 4, 2)
    assert not set(a) & set(c)
