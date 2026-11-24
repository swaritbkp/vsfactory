"""Small split=test datasets (no twins) for the probe / diversity CLI tests."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def small_split(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from knotwork.pipeline.generate import run_generate

    root = tmp_path_factory.mktemp("bench_split")
    for plugin, n in (("tangle", 12), ("gauge", 10), ("panel", 14)):
        rc = run_generate(
            plugin, n, root / plugin, 7, "offline", REPO / "configs/default_profile.yaml",
            "test", False, None, 0,
        )  # fmt: skip
        assert rc == 0, plugin
    return root
