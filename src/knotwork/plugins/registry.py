"""Plugin registry: name -> TaskPlugin instance (SPEC 8.0)."""

from __future__ import annotations

import importlib
from pathlib import Path
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
    if name in available():  # a scaffolded plugin package beside the built-in ones
        return importlib.import_module(f"knotwork.plugins.{name}")
    raise KeyError(f"unknown plugin {name!r}; known: {', '.join(available())}")


def available() -> tuple[str, ...]:
    """The built-in plugins followed by any other package here that exports `PLUGIN`."""
    root = Path(__file__).resolve().parent
    extra = sorted(
        p.name
        for p in root.iterdir()
        if p.name not in NAMES
        and (p / "plugin.py").is_file()
        and (p / "__init__.py").is_file()
        and "PLUGIN" in (p / "__init__.py").read_text(encoding="utf-8")
    )
    return (*NAMES, *extra)


def get_plugin(name: str) -> TaskPlugin:
    plugin: TaskPlugin = _module(name).PLUGIN
    return plugin


def fault_effective(name: str, scene: SceneIR, fault: FaultCode) -> bool:
    check = getattr(_module(name), "fault_effective", None)
    return bool(check(scene, fault)) if check else False


def twin_unavailable(name: str) -> type[Exception]:
    exc: type[Exception] = getattr(_module(name), "TwinUnavailable", RuntimeError)
    return exc
