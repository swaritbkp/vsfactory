"""A small dev dataset (every family, with twins) and its full export, shared by the export tests."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def export_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from knotwork.pipeline.generate import run_generate

    root = tmp_path_factory.mktemp("export_data")
    for plugin, n in (("tangle", 3), ("gauge", 2), ("panel", 6)):
        rc = run_generate(
            plugin, n, root / plugin, 21, "offline", REPO / "configs/default_profile.yaml",
            "dev", True, None, 0,
        )  # fmt: skip
        assert rc == 0, plugin
    return root


@pytest.fixture(scope="session")
def exported(export_data: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    from knotwork.export.run import run_export

    out = tmp_path_factory.mktemp("export_out")
    assert run_export(export_data, "all", out, None) == 0
    return out
