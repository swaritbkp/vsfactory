"""Pipeline test fixtures (shared rendered datasets live in tests/conftest.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPO / "configs" / "default_profile.yaml"
sys.path.insert(0, str(REPO / "tests" / "plugins"))


@pytest.fixture(scope="session")
def profile_path() -> Path:
    return PROFILE_PATH
