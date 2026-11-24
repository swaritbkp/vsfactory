"""Panel plugin: sockets -> tangle -> gauges composition (untrusted; SPEC 8.3)."""

from knotwork.plugins.panel.plugin import PLUGIN, PanelPlugin, TwinUnavailable, fault_effective

__all__ = ["PLUGIN", "PanelPlugin", "TwinUnavailable", "fault_effective"]
