"""Plugin registry: name -> TaskPlugin instance (SPEC 8.0).

Three sources, in this order:

1. the built-in plugins (`tangle`, `gauge`, `panel`);
2. scaffolded packages beside them in `knotwork/plugins/` that export `PLUGIN` (e.g. `rack_patch`);
3. external wheels that declare an entry point in the `knotwork.plugins` group:

       [project.entry-points."knotwork.plugins"]
       my_domain = "my_pkg.plugin:PLUGIN"     # or "my_pkg" (a module that exports PLUGIN)

External registrations are validated before use and never crash the registry: a name that
collides with a local plugin, a load error, a missing protocol method or attribute, or an entry
point whose name differs from `PLUGIN.name` is *rejected* and reported by `knot plugins` with the
reason. Plugins stay untrusted either way: the kernel re-derives every answer from pixels.
In the public demo (`KNOTWORK_PUBLIC_DEMO=1`) entry points are not scanned at all, so only the
local plugins load.
"""

from __future__ import annotations

import functools
import importlib
import os
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from knotwork.knotcore.contracts import FaultCode, SceneIR, TaskPlugin

NAMES = ("tangle", "gauge", "panel")
ENTRY_POINT_GROUP = "knotwork.plugins"
PUBLIC_DEMO_ENV = "KNOTWORK_PUBLIC_DEMO"  # same switch as knotwork.studio.demo_guard
#: the TaskPlugin protocol: methods an external plugin must provide, and attributes it must set
REQUIRED_METHODS = ("sample_scene", "render", "claims", "questions", "symbolic_answer",
                    "geometric_answer", "causal_twin", "nuisance_twin")  # fmt: skip
REQUIRED_ATTRS = ("name", "version", "families")


@dataclass(frozen=True)
class PluginInfo:
    """One row of `knot plugins`."""

    name: str
    source: str  # builtin | local | external
    version: str
    status: str  # ok | rejected
    origin: str  # module path, or "dist==version (entry point value)"
    detail: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _local() -> tuple[str, ...]:
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


def _entry_points() -> list[Any]:
    """Installed `knotwork.plugins` entry points (a seam for tests)."""
    return list(metadata.entry_points(group=ENTRY_POINT_GROUP))


def check_plugin(plugin: Any) -> list[str]:
    """Protocol problems of a plugin object (empty list = it satisfies TaskPlugin)."""
    problems = [f"missing attribute {a!r}" for a in REQUIRED_ATTRS if not hasattr(plugin, a)]
    problems += [f"missing method {m}()" for m in REQUIRED_METHODS
                 if not callable(getattr(plugin, m, None))]  # fmt: skip
    return problems


def _dist(ep: Any) -> str:
    dist = getattr(ep, "dist", None)
    return f"{dist.name}=={dist.version}" if dist is not None else "unknown-dist"


@functools.lru_cache(maxsize=1)
def _external() -> tuple[dict[str, Any], tuple[PluginInfo, ...]]:
    """(accepted name -> module-like holder, every external registration with its status)."""
    if os.environ.get(PUBLIC_DEMO_ENV) == "1":  # hosted sandbox: never load third-party code
        return {}, ()
    local = set(_local())
    accepted: dict[str, Any] = {}
    infos: list[PluginInfo] = []
    try:
        eps = _entry_points()
    except Exception as exc:  # a broken environment must not take the built-ins down with it
        return {}, (PluginInfo("?", "external", "", "rejected", ENTRY_POINT_GROUP,
                               f"entry point scan failed: {type(exc).__name__}"),)  # fmt: skip
    for ep in sorted(eps, key=lambda e: str(e.name)):
        name, origin = str(ep.name), f"{_dist(ep)} ({ep.value})"

        def reject(why: str, version: str = "", name: str = name, origin: str = origin) -> None:
            infos.append(PluginInfo(name, "external", version, "rejected", origin, why))

        if name in local or name in accepted:
            reject(f"name collides with an already registered plugin {name!r}")
            continue
        try:
            loaded = ep.load()
        except Exception as exc:  # untrusted code: report, never crash
            reject(f"load failed: {type(exc).__name__}: {exc}"[:200])
            continue
        plugin = getattr(loaded, "PLUGIN", loaded)
        version = str(getattr(plugin, "version", ""))
        problems = check_plugin(plugin)
        if problems:
            reject("does not implement TaskPlugin: " + "; ".join(problems), version)
            continue
        if str(plugin.name) != name:
            reject(f"entry point name {name!r} != PLUGIN.name {plugin.name!r}", version)
            continue
        holder = loaded if hasattr(loaded, "PLUGIN") else SimpleNamespace(PLUGIN=plugin)
        accepted[name] = holder
        infos.append(PluginInfo(name, "external", version, "ok", origin))
    return accepted, tuple(infos)


def refresh() -> None:
    """Forget the cached entry point scan (after installing a wheel, or in tests)."""
    _external.cache_clear()


def _module(name: str) -> Any:
    if name == "tangle":
        from knotwork.plugins import tangle

        return tangle
    if name == "gauge":
        from knotwork.plugins import gauge

        return gauge
    if name == "panel":
        from knotwork.plugins import panel

        return panel
    if name in _local():  # a scaffolded plugin package beside the built-in ones
        return importlib.import_module(f"knotwork.plugins.{name}")
    external = _external()[0]
    if name in external:
        return external[name]
    raise KeyError(f"unknown plugin {name!r}; known: {', '.join(available())}")


def available() -> tuple[str, ...]:
    """Built-in, then scaffolded, then accepted external plugin names."""
    return (*_local(), *sorted(_external()[0]))


def list_plugins() -> list[PluginInfo]:
    """Every registration: local plugins (loaded to read their version) and external ones."""
    rows = []
    for name in _local():
        source = "builtin" if name in NAMES or name == "rack_patch" else "local"
        try:
            plugin = _module(name).PLUGIN
            rows.append(PluginInfo(name, source, str(plugin.version), "ok",
                                   f"knotwork.plugins.{name}"))  # fmt: skip
        except Exception as exc:
            rows.append(PluginInfo(name, source, "", "rejected", f"knotwork.plugins.{name}",
                                   f"{type(exc).__name__}: {exc}"[:200]))  # fmt: skip
    return [*rows, *_external()[1]]


def get_plugin(name: str) -> TaskPlugin:
    plugin: TaskPlugin = _module(name).PLUGIN
    return plugin


def fault_effective(name: str, scene: SceneIR, fault: FaultCode) -> bool:
    check = getattr(_module(name), "fault_effective", None)
    return bool(check(scene, fault)) if check else False


def twin_unavailable(name: str) -> type[Exception]:
    exc: type[Exception] = getattr(_module(name), "TwinUnavailable", RuntimeError)
    return exc
