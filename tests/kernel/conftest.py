"""Shared fixtures for kernel tests: synthetic bundles cached per test session."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import CFG, Dial, gauge_bundle, wires_bundle, x_scene  # noqa: E402

from knotwork.knotcore.io import Bundle, load_bundle  # noqa: E402


@pytest.fixture(scope="session")
def cfg() -> dict:  # type: ignore[type-arg]
    return CFG


@pytest.fixture(scope="session")
def x_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return wires_bundle(tmp_path_factory.mktemp("x"), x_scene())


@pytest.fixture()
def x_bundle(x_dir: Path) -> Bundle:
    return load_bundle(x_dir)


@pytest.fixture(scope="session")
def gauge_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return gauge_bundle(tmp_path_factory.mktemp("g"), Dial())
