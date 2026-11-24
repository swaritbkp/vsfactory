"""Session fixtures for pipeline tests: small generated datasets and one kept mutate run."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPO / "configs" / "default_profile.yaml"
sys.path.insert(0, str(REPO / "tests" / "plugins"))


@pytest.fixture(scope="session")
def profile_path() -> Path:
    return PROFILE_PATH


@pytest.fixture(scope="session")
def datasets(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """data_tangle (n=3), data_gauge (n=2), data_panel (n=6, every family), all with twins."""
    from knotwork.pipeline.generate import run_generate

    root = tmp_path_factory.mktemp("phase2data")
    for plugin, n in (("tangle", 3), ("gauge", 2), ("panel", 6)):
        rc = run_generate(
            plugin, n, root / f"data_{plugin}", 5, "offline", PROFILE_PATH, "dev", True, None, 0
        )
        assert rc == 0, plugin
    return root


@pytest.fixture(scope="session")
def mutate_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """`knot mutate --fault all --n 1` with the faulty bundles kept on disk."""
    from knotwork.pipeline.io import read_json
    from knotwork.pipeline.mutate import run_mutate

    root = tmp_path_factory.mktemp("mutate")
    out = root / "report.json"
    rc = run_mutate("all", 1, 0, out, PROFILE_PATH, keep_dir=root / "kept")
    return {"rc": rc, "report": read_json(out), "path": out}
