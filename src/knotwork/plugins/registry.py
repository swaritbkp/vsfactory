"""Plugin registry: name -> TaskPlugin instance (SPEC 8.0)."""

from __future__ import annotations

from types import ModuleType

from knotwork.knotcore.contracts import FaultCode, SceneIR, TaskPlugin

NAMES = ("tangle", "gauge", "panel")


def _module(name: str) -> ModuleType:
    if name == "tangle":
        from knotwork.plugins import tangle

        return tangle
    if name == "gauge":
        from knotwork.plugins import gauge

        return gauge
    if name == "panel":
        from knotwork.plugins import panel

        return panel
    raise KeyError(f"unknown plugin {name!r}; known: {', '.join(NAMES)}")


def get_plugin(name: str) -> TaskPlugin:
    plugin: TaskPlugin = _module(name).PLUGIN
    return plugin


def fault_effective(name: str, scene: SceneIR, fault: FaultCode) -> bool:
    return bool(_module(name).fault_effective(scene, fault))


def twin_unavailable(name: str) -> type[Exception]:
    exc: type[Exception] = _module(name).TwinUnavailable
    return exc
