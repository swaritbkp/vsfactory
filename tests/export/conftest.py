"""The shared dev dataset (every family, with twins) and its full export for the export tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def export_data(datasets: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("export_data")
    for plugin in ("tangle", "gauge", "panel"):
        shutil.copytree(datasets / f"data_{plugin}", root / plugin)
    return root


@pytest.fixture(scope="session")
def exported(export_data: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    from knotwork.export.run import run_export

    out = tmp_path_factory.mktemp("export_out")
    assert run_export(export_data, "all", out, None) == 0
    return out
