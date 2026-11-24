"""Plugin-test fixtures (helpers live in plugin_helpers.py so tests can import them)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plugin_helpers import PROFILE  # noqa: E402


@pytest.fixture(scope="session")
def profile() -> dict[str, Any]:
    return PROFILE
