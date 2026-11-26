"""Gauge scale math (SPEC 8.2): value <-> angle, zones on a 5-unit grid, admissible needle values.

`angle(v) = start_deg - (v - vmin) / (vmax - vmin) * sweep_deg` (CW sweep, math convention).
Untrusted: the kernel never uses this; it interpolates OCR-verified tick values instead.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

ZONES = ("green", "amber", "red")


def angle(v: float, vmin: float, vmax: float, start: float = 225.0, sweep: float = 270.0) -> float:
    return start - (v - vmin) / (vmax - vmin) * sweep


def sample_zones(rng: np.random.Generator, vmin: float, vmax: float) -> tuple[float, float]:
    """(z1, z2) on a 5-unit grid with z1 in [0.45, 0.65] span and z2 in [0.70, 0.85] span."""
    span = vmax - vmin
    z1 = vmin + 5 * round(float(rng.uniform(0.45, 0.65)) * span / 5)
    z2 = vmin + 5 * round(float(rng.uniform(0.70, 0.85)) * span / 5)
    if z2 - z1 < 10:
        z2 = z1 + 10
    return float(z1), float(z2)


def arc_index(v: float, zones: tuple[float, float]) -> int:
    """Which of the three zone arcs (counted from the scale start) holds ``v``."""
    return 0 if v < zones[0] else (1 if v < zones[1] else 2)


def zone_of(v: float, zones: tuple[float, float], order: tuple[str, ...] = ZONES) -> str:
    return order[arc_index(v, zones)]


def dial_zone(d: dict[str, Any], v: float | None = None) -> str:
    """Zone name of ``v`` (default: the dial's value) under the dial's own arc colour order."""
    value = float(d["value"] if v is None else v)
    low, high = d["zones"]
    return zone_of(value, (low, high), tuple(d.get("zone_order") or ZONES))


def order_with(rng: np.random.Generator, name: str, index: int) -> list[str]:
    """A colour order that puts ``name`` on arc ``index``; the other two arcs are random."""
    rest = [z for z in ZONES if z != name]
    rest = [rest[int(i)] for i in rng.permutation(2)]
    return [*rest[:index], name, *rest[index:]]


def tick_values(vmin: float, vmax: float, n: int = 5) -> list[float]:
    return [vmin + k * (vmax - vmin) / (n - 1) for k in range(n)]


def admissible(
    vmin: float, vmax: float, zones: tuple[float, float], boundary_gap: float, tick_gap: float
) -> list[int]:
    """Integer needle values >= boundary_gap from zone boundaries/scale ends and >= tick_gap from
    every tick value (the needle would hide that tick's label)."""
    bounds = [vmin, zones[0], zones[1], vmax]
    ticks = tick_values(vmin, vmax)
    out = []
    for v in range(int(math.ceil(vmin)), int(math.floor(vmax)) + 1):
        if (
            min(abs(v - b) for b in bounds) >= boundary_gap
            and min(abs(v - t) for t in ticks) >= tick_gap
        ):
            out.append(v)
    return out


def tick_gap_for(
    radius: float,
    label_frac: float = 0.62,
    label_half_px: float = 18.0,
    needle_half_px: float = 6.0,
) -> float:
    """Value distance at which the needle clears a tick label drawn at label_frac R (plus 1 unit)."""
    deg = math.degrees(math.atan((label_half_px + needle_half_px) / (label_frac * radius)))
    return math.ceil(deg / 270.0 * 80.0) + 1.0
