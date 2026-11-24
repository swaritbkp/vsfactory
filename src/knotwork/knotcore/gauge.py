"""Gauge primitives from masks + OCR: READ, ZONE, GAUGE_OF, READ_ALL (SPEC 7.5-7.7).

Invariants: the dial centre is the dial-mask centroid (never `claimed_center`); the value comes from
the needle PCA angle interpolated between OCR-verified tick values (never plugin scale math); the
tolerance is derived from the raster resolution.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage

from knotwork.knotcore.contracts import ErrorCode, GaugeEntry
from knotwork.knotcore.io import Bundle, KernelFault, sl
from knotwork.knotcore.ocr import label_text
from knotwork.knotcore.topo import label_of, trace

DEG = 180.0 / math.pi


def gauge(b: Bundle, gid: str) -> GaugeEntry:
    for g in b.manifest.gauges:
        if g.gauge_id == gid:
            return g
    raise KernelFault(ErrorCode.E_TRACE_NO_TERMINAL, f"unknown gauge {gid}")


def _angle(cx: float, cy: float, xs: Any, ys: Any) -> Any:
    return np.degrees(np.arctan2(-(np.asarray(ys) - cy), np.asarray(xs) - cx)) % 360.0


def _phi(ref: float, theta: Any) -> Any:
    """CW progress from `ref`, wrapped to [-45, 315) so pixels just before the scale start stay
    near 0 instead of 360 (SPEC 3 convention; D-11)."""
    return (ref - theta + 45.0) % 360.0 - 45.0


def needle(b: Bundle, g: GaugeEntry, cfg: dict[str, Any]) -> dict[str, Any]:
    """Dial centre/radius and needle PCA axis, angle, length, tip; E_NEEDLE_OFFCENTER check."""
    dy, dx = np.nonzero(b.masks[g.dial_obj])
    cx, cy, radius = float(dx.mean()), float(dy.mean()), math.sqrt(len(dx) / math.pi)
    ny, nx = np.nonzero(b.masks[g.needle_obj])
    if len(nx) < 3:
        raise KernelFault(ErrorCode.E_SCALE_MISMATCH, "needle mask empty")
    p = np.stack([nx, ny], axis=1).astype(np.float64)
    m = p.mean(axis=0)
    w, v = np.linalg.eigh(np.cov((p - m).T))
    u = v[:, int(np.argmax(w))]
    proj = (p - m) @ u
    if np.hypot(*(m + u * proj.max() - (cx, cy))) < np.hypot(*(m + u * proj.min() - (cx, cy))):
        u, proj = -u, -proj
    offset = abs(float((cx - m[0]) * u[1] - (cy - m[1]) * u[0]))
    limit = max(float(cfg["gauge"]["max_center_offset_px"]), 0.03 * radius)
    tip = m + u * (proj.max() - 3.0)
    ev = {"center": [cx, cy], "R": radius, "axis_offset": offset, "limit": limit}
    ev |= {"theta_needle": float(math.degrees(math.atan2(-u[1], u[0])) % 360.0)}
    ev |= {"length": float(proj.max() - proj.min()), "tip": [float(tip[0]), float(tip[1])]}
    if offset > limit:
        raise KernelFault(ErrorCode.E_NEEDLE_OFFCENTER, f"axis offset {offset:.1f} px")
    return ev


def read(b: Bundle, gid: str, cfg: dict[str, Any]) -> tuple[list[float], dict[str, Any]]:
    """[value, tolerance] of the gauge reading."""
    g = gauge(b, gid)
    ev = needle(b, g, cfg)
    cx, cy = ev["center"]
    ticks: list[dict[str, Any]] = []
    for t in g.ticks:
        ys, xs = np.nonzero(b.masks[t.tick_obj])
        tx, ty = float(xs.mean()), float(ys.mean())
        text = label_text(b, t.label_obj, cfg)[0]
        ticks.append({"v": int(text), "theta": float(_angle(cx, cy, tx, ty)), "ocr": text})
        ticks[-1]["r"] = math.hypot(tx - cx, ty - cy)
    ref = min(ticks, key=lambda k: k["v"])["theta"]
    for k in ticks:
        k["phi"] = float(_phi(ref, k["theta"]))
    ticks.sort(key=lambda k: k["phi"])
    phi_n = float(_phi(ref, ev["theta_needle"]))
    vals = [k["v"] for k in ticks]
    ev |= {"ticks": ticks, "phi_needle": phi_n}
    if any(x >= y for x, y in zip(vals, vals[1:], strict=False)):
        raise KernelFault(ErrorCode.E_SCALE_MISMATCH, f"tick values out of order {vals}")
    if not ticks[0]["phi"] <= phi_n <= ticks[-1]["phi"]:
        raise KernelFault(ErrorCode.E_SCALE_MISMATCH, f"needle outside scale ({phi_n:.1f} deg)")
    i = max(j for j in range(len(ticks) - 1) if ticks[j]["phi"] <= phi_n)
    lo, hi = ticks[i], ticks[i + 1]
    slope = (hi["v"] - lo["v"]) / (hi["phi"] - lo["phi"])
    value = lo["v"] + (phi_n - lo["phi"]) * slope
    sigma = DEG / ev["length"] + DEG / lo["r"] + DEG / hi["r"]
    tol = float(cfg["gauge"]["tolerance_guard_k"]) * abs(slope) * sigma
    ev |= {"lo": lo["v"], "hi": hi["v"], "tol": tol}
    return [float(value), tol], ev


def zone(b: Bundle, gid: str, cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Name of the zone arc whose CW-progress interval holds the needle with >= 1 deg margin."""
    _, ev = read(b, gid, cfg)
    cx, cy = ev["center"]
    ref = min(ev["ticks"], key=lambda k: k["v"])["theta"]
    spans = {}
    for z in gauge(b, gid).zones:
        ys, xs = np.nonzero(b.masks[z.zone_obj])
        phi = _phi(ref, _angle(cx, cy, xs, ys))
        spans[z.name.value] = [float(phi.min()), float(phi.max())]
    phi_n = ev["phi_needle"]
    hits = [n for n, (a, c) in spans.items() if a + 1.0 <= phi_n <= c - 1.0]
    if len(hits) != 1:
        raise KernelFault(ErrorCode.E_ZONE_UNRESOLVED, f"needle {phi_n:.1f} vs {spans}")
    return hits[0], {"zones": spans, "phi_needle": phi_n}


def gauge_of(b: Bundle, port: str, cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Unique gauge whose 8-px dilated dial mask touches the port box (port id or its label)."""
    ports = [
        t
        for t in b.manifest.terminals
        if t.role == "port" and port in (t.terminal_id, label_of(b, t.terminal_id, cfg))
    ]
    if len(ports) != 1:
        raise KernelFault(ErrorCode.E_TRACE_NO_TERMINAL, f"{len(ports)} ports named {port}")
    box, hits = ports[0].box, []
    for g in b.manifest.gauges:
        win = b.masks[g.dial_obj][sl(box, 9)]
        grown = ndimage.binary_dilation(win, iterations=8)
        inner = grown[min(box.y0, 9) :, min(box.x0, 9) :][: box.y1 - box.y0, : box.x1 - box.x0]
        if inner.any():
            hits.append(g.gauge_id)
    if len(hits) != 1:
        code = ErrorCode.E_TRACE_AMBIGUOUS if hits else ErrorCode.E_TRACE_NO_TERMINAL
        raise KernelFault(code, f"port {port} touches {hits}")
    return hits[0], {"port": ports[0].terminal_id}


def read_all(
    b: Bundle, sockets: list[str], cfg: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    """[[socket label, value, tol], ...] following each socket's cable to its gauge."""
    out = []
    for s in sockets:
        g = gauge_of(b, trace(b, s, cfg)[0], cfg)[0]
        v, tol = read(b, g, cfg)[0]
        out.append([label_of(b, s, cfg), v, tol])
    return out, {}
