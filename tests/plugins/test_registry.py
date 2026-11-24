"""Task 2.6: plugin registry (SPEC 8.0)."""

from __future__ import annotations

import pytest

from knotwork.knotcore.contracts import TaskPlugin
from knotwork.plugins.registry import NAMES, get_plugin


@pytest.mark.parametrize("name", NAMES)
def test_get_plugin(name: str) -> None:
    p = get_plugin(name)
    assert isinstance(p, TaskPlugin) and p.name == name


def test_unknown_plugin_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_plugin("nope")
