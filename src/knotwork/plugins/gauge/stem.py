"""Mounting plate and process-connection stem of a pressure gauge (D-27): a hex socket under the bezel and a pipe that
runs down and optionally elbows to one side, ending in a union nut. One registered `decor` object
(mask + id explain its ink); drawn before the dial so the bezel covers the socket's top."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from knotwork.render.canvas import LayeredCanvas

#: steel, brass, black-painted, galvanised pipe finishes (body, highlight, shadow)
FINISHES = [
    ((150, 154, 160), (196, 200, 206), (104, 108, 114)),
    ((176, 148, 90), (214, 190, 132), (128, 104, 60)),
    ((70, 72, 76), (118, 120, 126), (44, 46, 50)),
    ((168, 172, 170), (206, 210, 208), (120, 124, 122)),
]
EDGE = 40.0  # stay clear of the canvas margin band and the corner screws
SCREW_KEEPOUT = 80.0


def sample_stem(
    rng: np.random.Generator, d: dict[str, Any], rb: float, size: float
) -> dict[str, Any]:
    """Stem geometry for a dial with centre ``d["center"]`` and outer (bezel) radius ``rb``."""
    cx, cy = d["center"]
    top = cy + rb - 4.0
    room = size - EDGE - top
    kind = (
        "none"
        if room < 50.0 or rng.random() < 0.2
        else ("elbow" if rng.random() < 0.55 else "down")
    )
    width = float(np.clip(0.16 * float(d["radius"]), 26.0, 52.0))
    length = float(rng.uniform(min(40.0, room), room)) if kind != "none" else 0.0
    side = 1 if rng.random() < 0.5 else -1
    x_end = cx + side * float(rng.uniform(0.3, 1.0)) * (size / 2.0)
    x_end = float(np.clip(x_end, SCREW_KEEPOUT + width, size - SCREW_KEEPOUT - width))
    finish = int(rng.integers(len(FINISHES)))
    header_t = float(rng.uniform(56.0, 120.0))
    flanges = sorted(float(v) for v in rng.uniform(EDGE + 40.0, size - EDGE - 40.0, size=2))
    if kind != "none" and room - header_t / 2 - 20.0 >= 50.0 and rng.random() < 0.5:
        # the gauge sits on a process header pipe spanning the housing (D-27)
        kind = "header"
        length = float(rng.uniform(50.0, room - header_t / 2 - 20.0))
    return {"kind": kind, "x": float(cx), "top": float(top), "length": length, "width": width,
            "x_end": x_end, "finish": finish, "header_t": header_t, "flanges": flanges}  # fmt: skip


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def draw_stem(c: LayeredCanvas, prefix: str, s: dict[str, Any]) -> int | None:
    if s["kind"] == "none":
        return None
    body, hi, lo = FINISHES[int(s["finish"])]
    oid = c.new_object(f"{prefix}/decor:stem", "decor", body)
    x, top, w, length = s["x"], s["top"], s["width"], s["length"]
    bottom = top + length
    hw = w / 2.0
    runs = [_rect(x - hw * 0.7, top, x + hw * 0.7, bottom)]
    if s["kind"] == "elbow":
        y_mid = bottom - hw * 0.7
        x0, x1 = sorted((x, s["x_end"]))
        runs.append(_rect(x0 - hw * 0.7, y_mid - hw * 0.7, x1 + hw * 0.7, y_mid + hw * 0.7))
    if s["kind"] == "header":
        t, size = float(s["header_t"]), float(c.size)
        y0, y1 = bottom - t / 2, bottom + t / 2
        runs.append(_rect(EDGE, y0, size - EDGE, y1))
    for r in runs:
        c.polygon(oid, r, beauty_rgb=body)
    if s["kind"] == "header":
        c.polygon(oid, _rect(EDGE, y0 + 0.18 * t, size - EDGE, y0 + 0.34 * t), beauty_rgb=hi,
                  mask=False, idbuf=False)  # fmt: skip
        c.polygon(
            oid, _rect(EDGE, y1 - 0.2 * t, size - EDGE, y1), beauty_rgb=lo, mask=False, idbuf=False
        )
        for fx in s["flanges"]:
            c.polygon(oid, _rect(fx - 9.0, y0 - 8.0, fx + 9.0, y1 + 8.0), beauty_rgb=lo)
    # cylindrical shading: a highlight and a shadow stripe along every run
    vx0, vx1 = x - hw * 0.7, x + hw * 0.7
    c.polygon(
        oid,
        _rect(vx0 + 0.15 * w, top, vx0 + 0.32 * w, bottom),
        beauty_rgb=hi,
        mask=False,
        idbuf=False,
    )
    c.polygon(oid, _rect(vx1 - 0.18 * w, top, vx1, bottom), beauty_rgb=lo, mask=False, idbuf=False)
    # hex socket right under the bezel
    hex_r, hy = 0.75 * w, top + 0.55 * w
    hexagon = [(x + hex_r * math.cos(math.radians(60 * k)), hy + hex_r * 0.62 * math.sin(math.radians(60 * k)))
               for k in range(6)]  # fmt: skip
    c.polygon(oid, hexagon, beauty_rgb=lo)
    c.polygon(oid, _rect(x - hex_r * 0.5, hy - hex_r * 0.62, x + hex_r * 0.5, hy + hex_r * 0.62),
              beauty_rgb=body, mask=False, idbuf=False)  # fmt: skip
    # union nut at the far end of the pipe
    if s["kind"] == "down":
        c.polygon(oid, _rect(x - hw, bottom - 0.5 * w, x + hw, bottom), beauty_rgb=lo)
    elif s["kind"] == "elbow":
        xe, ym = s["x_end"], bottom - hw * 0.7
        c.polygon(oid, _rect(xe - 0.5 * w, ym - hw, xe, ym + hw) if xe < x else _rect(xe, ym - hw, xe + 0.5 * w, ym + hw),
                  beauty_rgb=lo)  # fmt: skip
    return oid


#: painted mounting plates: dark grey, machinery green, blue-grey, mid grey, black, oxide red, ochre
PLATES = [
    (64, 68, 72),
    (54, 90, 70),
    (58, 76, 102),
    (128, 132, 136),
    (34, 36, 40),
    (150, 60, 44),
    (178, 150, 60),
]
PLATE_EDGE = 30.0


def sample_plate(
    rng: np.random.Generator, d: dict[str, Any], rb: float, size: float
) -> dict[str, Any] | None:
    """Painted plate the dial is mounted on: a rectangle around the bezel with independent random
    padding per side (clipped to the canvas), or None (dial mounted directly on the housing)."""
    if rng.random() < 0.15:
        return None
    cx, cy = d["center"]
    pad = rng.uniform(24.0, 320.0, size=4)
    lo, hi = PLATE_EDGE, size - PLATE_EDGE
    box = [max(lo, cx - rb - pad[0]), max(lo, cy - rb - pad[1]),
           min(hi, cx + rb + pad[2]), min(hi, cy + rb + pad[3])]  # fmt: skip
    return {"box": [float(v) for v in box], "rgb": list(PLATES[int(rng.integers(len(PLATES)))])}


def draw_plate(c: LayeredCanvas, prefix: str, p: dict[str, Any] | None) -> int | None:
    if p is None:
        return None
    x0, y0, x1, y1 = p["box"]
    rgb = tuple(int(v) for v in p["rgb"])
    oid = c.new_object(f"{prefix}/decor:plate", "decor", (rgb[0], rgb[1], rgb[2]))
    c.polygon(oid, _rect(x0, y0, x1, y1), beauty_rgb=(rgb[0], rgb[1], rgb[2]))
    light = (min(255, rgb[0] + 40), min(255, rgb[1] + 40), min(255, rgb[2] + 40))
    dark = (max(0, rgb[0] - 24), max(0, rgb[1] - 24), max(0, rgb[2] - 24))
    for r, col in (
        (_rect(x0, y0, x1, y0 + 3), light),
        (_rect(x0, y0, x0 + 3, y1), light),
        (_rect(x0, y1 - 3, x1, y1), dark),
        (_rect(x1 - 3, y0, x1, y1), dark),
    ):
        c.polygon(oid, r, beauty_rgb=col, mask=False, idbuf=False)
    return oid


__all__ = ["draw_plate", "draw_stem", "sample_plate", "sample_stem"]
