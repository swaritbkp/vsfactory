"""Analog dial rendering with fault hooks M8/M9 (SPEC 8.2, 10). Reused by the panel plugin.

Draw order: dial face (+ bezel ring inside the face footprint), zone arcs, major ticks, minor ticks,
needle, hub, tick labels. Each tick label sits on a face-coloured numeral plate drawn over the
needle, so a needle passing a numeral never enters that numeral's OCR crop (D-15).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from knotwork.knotcore.contracts import FaultCode, GaugeEntry, TickEntry, ZoneEntry, ZoneName
from knotwork.plugins.gauge.scale import ZONES, angle, tick_values
from knotwork.render.canvas import LayeredCanvas

ZONE_RGB = {"green": (40, 160, 70), "amber": (232, 160, 20), "red": (210, 40, 40)}
NEEDLE_RGB = (200, 30, 30)
FACE_RGB = (255, 255, 255)
M9_SHIFT = (14.0, -9.0)
M8_SWEEP = 1.12
#: Numeral plates extend this far past the glyphs (> 4-px box pad + 5-px OCR crop margin at 768).
PLATE_MARGIN_PX = 10.0


def _pt(c: tuple[float, float], r: float, deg: float) -> tuple[float, float]:
    t = math.radians(deg)
    return c[0] + r * math.cos(t), c[1] - r * math.sin(t)


def draw_dial(
    c: LayeredCanvas, prefix: str, d: dict[str, Any], fault: FaultCode | None = None
) -> GaugeEntry:
    """Draw one dial described by ``d`` (name, center, radius, labels, vmin, vmax, zones, value,
    bezel_rgb, band_inner) and return its manifest entry."""
    name, (cx, cy), r = d["name"], d["center"], float(d["radius"])
    vmin, vmax = float(d["vmin"]), float(d["vmax"])
    gid = f"{prefix}/gauge:{name}"
    dial = c.new_object(f"{prefix}/dial:{name}", "dial", FACE_RGB, parent=gid)
    c.disk(dial, (cx, cy), r, outline_rgb=tuple(d["bezel_rgb"]), outline_px=4)
    bounds = [vmin, d["zones"][0], d["zones"][1], vmax]
    zones = []
    for k, zn in enumerate(ZONES):
        z = c.new_object(f"{prefix}/zone_arc:{name}_{zn}", "zone_arc", ZONE_RGB[zn], parent=gid)
        a0, a1 = angle(bounds[k + 1], vmin, vmax), angle(bounds[k], vmin, vmax)
        c.arc_band(z, (cx, cy), float(d["band_inner"]) * r, r - 6.0, a0, a1)
        zones.append(ZoneEntry(zone_obj=z, name=ZoneName(zn)))
    majors = tick_values(vmin, vmax, len(d["labels"]))
    tick_objs = []
    for k, v in enumerate(majors):
        deg = angle(v, vmin, vmax)
        t = c.new_object(f"{prefix}/tick:{name}_{k}", "tick", (40, 40, 40), parent=gid)
        c.polyline(t, np.array([_pt((cx, cy), 0.78 * r, deg), _pt((cx, cy), 0.90 * r, deg)]), 6)
        tick_objs.append(t)
    for k in range(len(majors) - 1):
        for m in range(1, 5):
            deg = angle(majors[k] + m * (majors[k + 1] - majors[k]) / 5, vmin, vmax)
            o = c.new_object(
                f"{prefix}/tick_minor:{name}_{k}_{m}", "tick_minor", (95, 95, 95), parent=gid
            )
            c.polyline(o, np.array([_pt((cx, cy), 0.84 * r, deg), _pt((cx, cy), 0.90 * r, deg)]), 3)
    sweep = 270.0 * (M8_SWEEP if fault is FaultCode.M8 else 1.0)
    deg = angle(float(d["value"]), vmin, vmax, sweep=sweep)
    shift = M9_SHIFT if fault is FaultCode.M9 else (0.0, 0.0)
    px, py = cx + shift[0], cy + shift[1]
    ux, uy = math.cos(math.radians(deg)), -math.sin(math.radians(deg))
    nx, ny = -uy, ux
    tail, tip, wt, wp = 0.15 * r, 0.85 * r, max(6.0, 0.03 * r), 5.5
    needle = c.new_object(f"{prefix}/needle:{name}", "needle", NEEDLE_RGB, parent=gid)
    c.polygon(needle, [
        (px - ux * tail + nx * wt, py - uy * tail + ny * wt), (px + ux * tip + nx * wp, py + uy * tip + ny * wp),
        (px + ux * tip - nx * wp, py + uy * tip - ny * wp), (px - ux * tail - nx * wt, py - uy * tail - ny * wt),
    ])  # fmt: skip
    hub = c.new_object(f"{prefix}/decor:{name}_hub", "decor", (60, 60, 60), parent=gid)
    c.disk(hub, (px, py), max(8.0, 0.035 * r))
    ticks = []
    for k, (v, text) in enumerate(zip(majors, d["labels"], strict=True)):
        lab = c.new_object(f"{prefix}/label:{name}_t{k}", "label", (30, 30, 30),
                           parent=f"{prefix}/labels:{name}_ticks", label_text=text)  # fmt: skip
        x, y = _pt((cx, cy), float(d.get("label_frac", 0.62)) * r, angle(v, vmin, vmax))
        gl, gt, gr, gb = c.font(26).getbbox(text, anchor="mm")
        pm = PLATE_MARGIN_PX  # face-coloured numeral plate: covers the label's whole OCR crop
        c.polygon(lab, [(x + gl - pm, y + gt - pm), (x + gr + pm, y + gt - pm), (x + gr + pm, y + gb + pm),
                        (x + gl - pm, y + gb + pm)], beauty_rgb=FACE_RGB)  # fmt: skip
        c.text(lab, (x, y), text, 26)
        ticks.append(TickEntry(tick_obj=tick_objs[k], label_obj=lab))
    return GaugeEntry(gauge_id=gid, dial_obj=dial, needle_obj=needle, ticks=ticks, zones=zones,
                      claimed_center=(float(cx), float(cy)), claimed_radius=r)  # fmt: skip
