"""D2 for gauges: the reading recovered from the needle polygon geometry with shapely (SPEC 8.2)."""

from __future__ import annotations

import math
from typing import Any

from shapely.geometry import Point, Polygon

from knotwork.plugins.gauge.scale import zone_of


def needle_polygon(d: dict[str, Any]) -> Polygon:
    cx, cy = d["center"]
    r, vmin, vmax = float(d["radius"]), float(d["vmin"]), float(d["vmax"])
    deg = 225.0 - (float(d["value"]) - vmin) / (vmax - vmin) * 270.0
    ux, uy = math.cos(math.radians(deg)), -math.sin(math.radians(deg))
    nx, ny = -uy, ux
    tail, tip, wt, wp = 0.15 * r, 0.85 * r, max(6.0, 0.03 * r), 5.5
    return Polygon([(cx - ux * tail + nx * wt, cy - uy * tail + ny * wt), (cx + ux * tip + nx * wp, cy + uy * tip + ny * wp),
                    (cx + ux * tip - nx * wp, cy + uy * tip - ny * wp), (cx - ux * tail - nx * wt, cy - uy * tail - ny * wt)])  # fmt: skip


def geometric_reading(d: dict[str, Any]) -> float:
    """Value from the angle of the needle polygon's farthest vertex pair about the dial centre."""
    poly, c = needle_polygon(d), Point(d["center"])
    coords = list(poly.exterior.coords)[:-1]
    far = sorted(coords, key=lambda p: -Point(p).distance(c))[:2]
    tx, ty = (far[0][0] + far[1][0]) / 2, (far[0][1] + far[1][1]) / 2
    deg = math.degrees(math.atan2(-(ty - c.y), tx - c.x))
    vmin, vmax = float(d["vmin"]), float(d["vmax"])
    phi = (225.0 - deg) % 360.0
    return round(vmin + phi / 270.0 * (vmax - vmin), 6)


def geometric_zone(d: dict[str, Any]) -> str:
    return zone_of(geometric_reading(d), tuple(d["zones"]))
