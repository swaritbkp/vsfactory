"""Shapely validation of tangle layouts and the D2 geometric answer (SPEC 8.1). Untrusted.

`validate` rejects any layout that could fold or be illegible before it is ever rendered: every strand
simple, pairwise intersection counts equal braid crossing counts, strands >= 2 strokes apart away
from crossings, nominal crossing angles >= 35 deg, consecutive crossings along a strand >= 4.5
strokes apart, everything inside the canvas margin.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import unary_union

from knotwork.plugins.tangle.layout import Layout

MIN_ANGLE_DEG = 35.0


def _points(geom: Any) -> list[Point]:
    if geom.is_empty:
        return []
    if isinstance(geom, Point):
        return [geom]
    if isinstance(geom, MultiPoint):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Point)] or [geom.centroid]


def _direction(line: LineString, d: float) -> np.ndarray:
    a = np.asarray(line.interpolate(max(d - 3.0, 0.0)).coords[0])
    b = np.asarray(line.interpolate(min(d + 3.0, line.length)).coords[0])
    v = b - a
    return np.asarray(v / (np.linalg.norm(v) or 1.0), dtype=np.float64)


def crossing_angle(la: LineString, lb: LineString, p: Point) -> float:
    u, v = _direction(la, la.project(p)), _direction(lb, lb.project(p))
    return math.degrees(math.acos(min(1.0, abs(float(u @ v)))))


def validate(lay: Layout, n: int, stroke: float, margin: float = 30.0) -> str | None:
    """None if the layout is acceptable, else a short reason."""
    lines = [LineString(p) for p in lay.paths]
    for k, line in enumerate(lines):
        if not line.is_simple:
            return f"strand {k} folds"
        x, y = lay.paths[k][:, 0], lay.paths[k][:, 1]
        if (
            x.min() < margin
            or y.min() < margin
            or x.max() > 1024 - margin
            or y.max() > 1024 - margin
        ):
            return f"strand {k} leaves the canvas margin"
    want: dict[tuple[int, int], int] = {}
    for c in lay.crossings:
        key = (min(c["a"], c["b"]), max(c["a"], c["b"]))
        want[key] = want.get(key, 0) + 1
    hits: dict[int, list[float]] = {k: [] for k in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            pts = _points(lines[i].intersection(lines[j]))
            if len(pts) != want.get((i, j), 0):
                return f"strands {i},{j}: {len(pts)} intersections != {want.get((i, j), 0)}"
            for p in pts:
                if crossing_angle(lines[i], lines[j], p) < MIN_ANGLE_DEG:
                    return f"strands {i},{j}: crossing angle below {MIN_ANGLE_DEG}"
                hits[i].append(lines[i].project(p))
                hits[j].append(lines[j].project(p))
            holes = unary_union([p.buffer(3.5 * stroke) for p in pts]) if pts else None
            a = lines[i].difference(holes) if holes is not None else lines[i]
            b = lines[j].difference(holes) if holes is not None else lines[j]
            if not a.is_empty and not b.is_empty and a.distance(b) < 2.0 * stroke + 1.0:
                return f"strands {i},{j} run closer than 2 strokes"
    for k, ds in hits.items():
        gaps = np.diff(sorted(ds))
        if len(gaps) and gaps.min() < 4.5 * stroke:
            return f"strand {k}: crossings {gaps.min():.0f} px apart"
    return None


def geometric_answer(lay: Layout, family: str, args: list[Any], ends: list[np.ndarray]) -> Any:
    """D2 from shapely geometry: T1 end index, T2 intersection count, T3 top strand index.

    ``args`` are strand indices: T1 [s], T2 [s], T3 [a, b, idx]; ``ends`` are end-terminal centres
    top to bottom.
    """
    lines = [LineString(p) for p in lay.paths]
    if family == "T1_TRACE":
        tip = np.asarray(lines[args[0]].coords[-1])
        return int(np.argmin([np.hypot(*(tip - e)) for e in ends]))
    if family == "T2_CROSS":
        s = args[0]
        return sum(
            len(_points(lines[s].intersection(lines[o]))) for o in range(len(lines)) if o != s
        )
    a, b, idx = args
    pts = sorted(_points(lines[a].intersection(lines[b])), key=lambda p: lines[a].project(p))
    p = pts[idx]
    near = min(
        (c for c in lay.crossings if {c["a"], c["b"]} == {a, b}),
        key=lambda c: math.hypot(c["center"][0] - p.x, c["center"][1] - p.y),
    )
    return int(near["over"])
