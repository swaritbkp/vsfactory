"""Model Arena fixtures: a workspace root and a frozen dataset under the test's temporary folder."""

from __future__ import annotations

from pathlib import Path

import pytest

from arena_fixtures import make_dataset
from knotwork.arena import storage
from knotwork.arena.schemas import ItemRecord


@pytest.fixture
def arena_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "arena"
    monkeypatch.setenv(storage.ROOT_ENV, str(root))
    return root


@pytest.fixture
def frozen(arena_root: Path) -> tuple[Path, list[ItemRecord]]:
    data = arena_root / "data"
    items = make_dataset(
        data, {"b1": ("rack_patch", "High stress", 6), "b2": ("panel", "Hard logic", 4)}
    )
    return data, items
