"""Analog dial rendering with fault hooks M8/M9 (SPEC 8.2, 10). Reused by the panel plugin.

Draw order: metal bezel + face (one dial object), zone arcs, major ticks, minor/medium ticks (one
object), printed unit marks, needle contact shadow (beauty only, on the face), tapered needle with
counterweight, metallic hub, tick labels; then a seeded photographic finish (`render.realism`). Each tick label sits on a face-coloured numeral plate drawn over the
needle, so a needle passing a numeral never enters that numeral's OCR crop (D-15).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from knotwork.knotcore.contracts import FaultCode, GaugeEntry, TickEntry, ZoneEntry, ZoneName
from knotwork.plugins.gauge.scale import ZONES, angle, tick_values
from knotwork.render import realism
from knotwork.render.canvas import LayeredCanvas

ZONE_RGB = {"green": (40, 160, 70), "amber": (232, 160, 20), "red": (210, 40, 40)}
NEEDLE_RGB = (200, 30, 30)
FACE_RGB = (250, 249, 244)
M9_SHIFT = (14.0, -9.0)
M8_SWEEP = 1.12
#: Numeral plates extend this far past the glyphs (> 4-px box pad + 5-px OCR crop margin at 768).
PLATE_MARGIN_PX = 10.0


def _pt(c: tuple[float, float], r: float, deg: float) -> tuple[float, float]:
    t = math.radians(deg)
    return c[0] + r * math.cos(t), c[1] - r * math.sin(t)


def _rgb(v: Any) -> tuple[int, int, int]:
    return int(v[0]), int(v[1]), int(v[2])


def bezel_width(r: float) -> float:
    return float(max(10, round(0.07 * r)))


def draw_dial(
    c: LayeredCanvas, prefix: str, d: dict[str, Any], fault: FaultCode | None = None
) -> GaugeEntry:
    """Draw one dial described by ``d`` and return its manifest entry.

    ``d``: name, center, radius (face), labels, vmin, vmax, zones, zone_order (optional colour order
    of the three arcs), value, bezel_rgb, face_rgb,
    band_inner, label_frac, minor_per_major, unit, serial, surface. The metal bezel is part of the dial
    object (a concentric ring), so the kernel's dial centroid stays the true centre.
    """
    name, (cx, cy), r = d["name"], d["center"], float(d["radius"])
    vmin, vmax = float(d["vmin"]), float(d["vmax"])
    face_rgb = _rgb(d.get("face_rgb", FACE_RGB))
    rb = r + bezel_width(r)
    gid = f"{prefix}/gauge:{name}"
    dial = c.new_object(f"{prefix}/dial:{name}", "dial", face_rgb, parent=gid)
    c.disk(dial, (cx, cy), rb, beauty_rgb=_rgb(d["bezel_rgb"]))
    c.disk(dial, (cx, cy), r, beauty_rgb=face_rgb)
    bounds = [vmin, d["zones"][0], d["zones"][1], vmax]
    zones = []
    for k, zn in enumerate(d.get("zone_order") or ZONES):
        z = c.new_object(f"{prefix}/zone_arc:{name}_{zn}", "zone_arc", ZONE_RGB[zn], parent=gid)
        a0, a1 = angle(bounds[k + 1], vmin, vmax), angle(bounds[k], vmin, vmax)
        c.arc_band(z, (cx, cy), float(d["band_inner"]) * r, r - 6.0, a0, a1)
        zones.append(ZoneEntry(zone_obj=z, name=ZoneName(zn)))
    majors = tick_values(vmin, vmax, len(d["labels"]))
    tick_objs = []
    for k, v in enumerate(majors):
        deg = angle(v, vmin, vmax)
        t = c.new_object(f"{prefix}/tick:{name}_{k}", "tick", (35, 35, 35), parent=gid)
        c.polyline(t, np.array([_pt((cx, cy), 0.78 * r, deg), _pt((cx, cy), 0.90 * r, deg)]), 6)
        tick_objs.append(t)
    per = int(d.get("minor_per_major", 9))
    minor = c.new_object(f"{prefix}/tick_minor:{name}", "tick_minor", (70, 70, 70), parent=gid)
    for k in range(len(majors) - 1):
        for m in range(1, per + 1):
            deg = angle(majors[k] + m * (majors[k + 1] - majors[k]) / (per + 1), vmin, vmax)
            medium = per % 2 == 1 and m == (per + 1) // 2
            r0 = (0.81 if medium else 0.85) * r
            c.polyline(
                minor,
                np.array([_pt((cx, cy), r0, deg), _pt((cx, cy), 0.90 * r, deg)]),
                4 if medium else 2,
            )
    if r >= 200 and d.get("unit"):
        mark = c.new_object(f"{prefix}/decor:{name}_print", "decor", (70, 70, 70), parent=gid)
        c.text(mark, (cx, cy + 0.30 * r), str(d["unit"]), max(14, round(0.075 * r)), box_pad=2)
        if d.get("serial"):
            c.text(
                mark, (cx, cy + 0.44 * r), str(d["serial"]), max(10, round(0.037 * r)), box_pad=2
            )
    sweep = 270.0 * (M8_SWEEP if fault is FaultCode.M8 else 1.0)
    deg = angle(float(d["value"]), vmin, vmax, sweep=sweep)
    shift = M9_SHIFT if fault is FaultCode.M9 else (0.0, 0.0)
    px, py = cx + shift[0], cy + shift[1]
    ux, uy = math.cos(math.radians(deg)), -math.sin(math.radians(deg))
    nx, ny = -uy, ux
    tail, tip, wt, wp = 0.15 * r, 0.85 * r, max(6.0, 0.03 * r), 5.5
    poly = [
        (px - ux * tail + nx * wt, py - uy * tail + ny * wt), (px + ux * tip + nx * wp, py + uy * tip + ny * wp),
        (px + ux * tip - nx * wp, py + uy * tip - ny * wp), (px - ux * tail - nx * wt, py - uy * tail - ny * wt),
    ]  # fmt: skip
    weight = (px - ux * tail, py - uy * tail)
    rw = max(8.0, 0.045 * r)
    off = max(3.0, 0.012 * r)  # soft contact shadow cast on the face (beauty only, inside the dial)
    shadow = _rgb([v * 0.80 for v in face_rgb])
    needle = c.new_object(f"{prefix}/needle:{name}", "needle", NEEDLE_RGB, parent=gid)
    c.polygon(
        needle, [(x + off, y + off) for x, y in poly], mask=False, idbuf=False, beauty_rgb=shadow
    )
    c.disk(
        needle, (weight[0] + off, weight[1] + off), rw, mask=False, idbuf=False, beauty_rgb=shadow
    )
    c.polygon(needle, poly)
    c.disk(needle, weight, rw)  # counterweight tail: symmetric about the needle axis
    hub_r = max(9.0, 0.045 * r)
    hub = c.new_object(f"{prefix}/decor:{name}_hub", "decor", (150, 152, 158), parent=gid)
    c.disk(hub, (px, py), hub_r)
    ticks = []
    for k, (v, text) in enumerate(zip(majors, d["labels"], strict=True)):
        lab = c.new_object(f"{prefix}/label:{name}_t{k}", "label", (30, 30, 30),
                           parent=f"{prefix}/labels:{name}_ticks", label_text=text)  # fmt: skip
        x, y = _pt((cx, cy), float(d.get("label_frac", 0.62)) * r, angle(v, vmin, vmax))
        gl, gt, gr, gb = c.font(26).getbbox(text, anchor="mm")
        pm = PLATE_MARGIN_PX  # face-coloured numeral plate: covers the label's whole OCR crop
        c.polygon(lab, [(x + gl - pm, y + gt - pm), (x + gr + pm, y + gt - pm), (x + gr + pm, y + gb + pm),
                        (x + gl - pm, y + gb + pm)], beauty_rgb=face_rgb)  # fmt: skip
        c.text(lab, (x, y), text, 26)
        ticks.append(TickEntry(tick_obj=tick_objs[k], label_obj=lab))
    keep_clean = [needle] + [t.label_obj for t in ticks]
    c.post.append(lambda img, cv: _finish(img, cv, d, r, rb, hub_r, keep_clean))
    return GaugeEntry(gauge_id=gid, dial_obj=dial, needle_obj=needle, ticks=ticks, zones=zones,
                      claimed_center=(float(cx), float(cy)), claimed_radius=r)  # fmt: skip


def _finish(img: Any, cv: LayeredCanvas, d: dict[str, Any], r: float, rb: float, hub_r: float,
            keep_clean: list[int]) -> Any:  # fmt: skip
    """Run realism.dial_finish on the dial's window; needle and numeral plates stay crisp."""
    cx, cy = d["center"]
    pad = int(rb) + 8
    y0, y1 = max(int(cy) - pad, 0), min(int(cy) + pad, img.shape[0])
    x0, x1 = max(int(cx) - pad, 0), min(int(cx) + pad, img.shape[1])
    win = (slice(y0, y1), slice(x0, x1))
    needle = cv.mask_array(keep_clean[0])[win]
    plates = np.zeros(needle.shape, dtype=bool)
    for oid in keep_clean[1:]:
        plates |= cv.mask_array(oid)[win]
    # needle fully protected (colour witnesses); numeral plates take half the finish so they age with
    # the face instead of reading as pasted-on labels, while glare and grime stay off the glyphs
    protect = np.maximum(realism.soft(needle, 5, 2.0), 0.6 * realism.soft(plates, 2, 1.5))
    local = dict(d, center=[float(cx) - x0, float(cy) - y0])
    out = img.copy()
    out[y0:y1, x0:x1] = realism.dial_finish(img[y0:y1, x0:x1], local, r, rb, hub_r, protect)
    return out
