"""Consumption-resolution legibility guard (SPEC 7.11): a task is only certified if a VLM seeing the
768-px image could resolve it -- strokes wide enough, crossings not tangent nor crowded, terminals
clear of foreign ink, and every label readable by OCR at 768 px (the single OCR pass).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage

from knotwork.knotcore.contracts import ErrorCode, WitnessCheck
from knotwork.knotcore.io import Bundle, KernelFault, sl
from knotwork.knotcore.ocr import label_text
from knotwork.knotcore.topo import (
    Crossing,
    Pixels,
    crossings,
    overlaps,
    pixels,
    strand_of,
    stroke,
)
from knotwork.knotcore.witness import _chk

E = ErrorCode
SCALE = 0.75


def _direction(px: Pixels, c: list[float], r: float) -> np.ndarray[Any, np.dtype[np.float64]]:
    ys, xs = px
    near = np.hypot(xs - c[0], ys - c[1]) <= r
    p = np.stack([xs[near], ys[near]], axis=1).astype(np.float64)
    if len(p) < 3:
        return np.zeros(2)
    w, v = np.linalg.eigh(np.cov((p - p.mean(axis=0)).T))
    return np.asarray(v[:, int(np.argmax(w))], dtype=np.float64)


def _strand_crossings(b: Bundle, start: str, sa: int) -> tuple[list[Crossing], float]:
    """(crossings of the strand, smallest geodesic gap between neighbours). The geodesic sweep is
    needed only for two or more crossings; pairwise overlaps are shared between all strands."""
    cs = [{**x, "other": o} for o in b.kind("strand") if o != sa for x in overlaps(b, sa, o)]
    if len(cs) < 2:
        return cs, math.inf
    cs = crossings(b, start, "*")
    return cs, float(np.diff([c["geo"] for c in cs]).min())


def legibility(b: Bundle, cfg: dict[str, Any]) -> list[WitnessCheck]:
    g, out = cfg["legibility"], []
    strokes = {o: stroke(b, o) for o in b.kind("strand") + b.kind("needle")}
    floor = float(g["min_stroke_px_at_consumption"])
    for o, s in strokes.items():
        out.append(_chk("stroke_768", o, s * SCALE >= floor, s * SCALE, floor, E.E_ILLEGIBLE_RES))
    typical = float(np.median(list(strokes.values()))) if strokes else 0.0
    min_ang, gap_k = float(g["min_crossing_angle_deg"]), float(g["min_crossing_spacing_strokes"])
    for t in (t for t in b.manifest.terminals if t.role == "start"):
        try:
            sa = strand_of(b, t.terminal_id)
            cs, gap = _strand_crossings(b, t.terminal_id, sa)
        except KernelFault:
            continue
        s = strokes[sa]
        for c in cs:
            u, v = (_direction(pixels(b, o), c["center"], 3 * s) for o in (sa, c["other"]))
            ang = math.degrees(math.acos(min(1.0, abs(float(u @ v)))))
            out.append(
                _chk("crossing_angle", sa, ang >= min_ang, ang, min_ang, E.E_TANGENT_CROSSING)
            )
        ok = gap >= gap_k * s
        out.append(_chk("crossing_spacing", sa, ok, min(gap, 1e9), gap_k * s, E.E_CROSSING_SPACING))
    k_clear = float(g["min_terminal_clearance_strokes"])
    for t in b.manifest.terminals:
        try:
            own: int | None = strand_of(b, t.terminal_id)
        except KernelFault:
            own = None
        s = strokes.get(own, typical) if own is not None else typical
        pad = int(math.ceil(2 * s)) + 2
        win = sl(t.box, pad)
        other = np.zeros_like(b.masks[next(iter(b.masks))][win])
        for oid, m in b.masks.items():
            if oid not in (own, t.label_obj):
                other |= m[win]
        inside = np.zeros_like(other)
        rows, cols = sl(t.box)
        inside[
            rows.start - win[0].start : max(rows.stop - win[0].start, 0),
            cols.start - win[1].start : max(cols.stop - win[1].start, 0),
        ] = True
        dist = np.asarray(ndimage.distance_transform_edt(~other), dtype=np.float64)
        d = float(dist[inside].min()) if other.any() and inside.any() else float(pad)
        out.append(
            _chk(
                "terminal_clearance", own, d >= k_clear * s, d, k_clear * s, E.E_TERMINAL_CLEARANCE
            )
        )
    for oid in b.kind("label"):
        try:
            label_text(b, oid, cfg)
            out.append(_chk("ocr_768", oid, True, 1.0, 1.0, E.E_LABEL_MISMATCH))
        except KernelFault as f:
            out.append(_chk("ocr_768", oid, False, 0.0, 1.0, f.code))
    return out
