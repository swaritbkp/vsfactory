"""Tangle rendering onto a LayeredCanvas, with render-time fault hooks (SPEC 6.1, 8.1, 10).

Public helpers (`draw_cables`, `add_terminal_labels`) are reused by the panel plugin. Fault hooks
only change what is drawn; claims, questions and D1 stay those of the clean scene, so the kernel must
catch the discrepancy from pixels alone.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np

from knotwork.knotcore.contracts import FaultCode, PixelBox, TerminalEntry
from knotwork.render.canvas import LayeredCanvas, Shade

Role = Literal["start", "end", "port"]

RGB = tuple[int, int, int]
LABEL_RGB: RGB = (30, 30, 30)
HALO_PX, HALO_RADIUS = 5.0, 22.0


def hex_rgb(h: str) -> RGB:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def blend(rgb: RGB, bg: RGB, keep: float) -> RGB:
    r, g, b = (int(round(keep * c + (1 - keep) * k)) for c, k in zip(rgb, bg, strict=True))
    return r, g, b


def unit(v: np.ndarray) -> np.ndarray:
    return np.asarray(v / (np.linalg.norm(v) or 1.0), dtype=np.float64)


def draw_cables(
    c: LayeredCanvas,
    prefix: str,
    paths: list[np.ndarray],
    crossings: list[dict[str, Any]],
    colors: list[RGB],
    width: float,
    shade: str,
    halos: bool = True,
    beauty_rgb: dict[int, RGB] | None = None,
) -> list[int]:
    """Draw every cable fully in index order, then per-crossing halos in increasing x (SPEC 6.1)."""
    ids = [c.new_object(f"{prefix}/strand:s{k}", "strand", colors[k]) for k in range(len(paths))]
    for k, p in enumerate(paths):
        c.polyline(ids[k], p, width, beauty_rgb=(beauty_rgb or {}).get(k), shade=cast(Shade, shade))
    if halos:
        for x in sorted(crossings, key=lambda x: (x["center"][0], x["center"][1])):
            top = int(x["over"])
            c.halo_patch(ids[top], paths[top], width, HALO_PX, HALO_RADIUS, tuple(x["center"]))
    return ids


def box_at(p: np.ndarray, half: int = 10) -> PixelBox:
    x, y = int(round(float(p[0]))), int(round(float(p[1])))
    return PixelBox(x0=x - half, y0=y - half, x1=x + half, y1=y + half)


def add_terminal_labels(
    c: LayeredCanvas,
    prefix: str,
    role: str,
    points: list[np.ndarray],
    outward: list[np.ndarray],
    names: list[str],
    texts: list[str | None],
    drawn: list[str | None] | None = None,
    font_px: int = 26,
) -> list[TerminalEntry]:
    """Terminal boxes at ``points`` with labels placed ``outward`` (unit vectors) from them.

    ``drawn`` overrides the glyphs actually rendered (fault M3) while the manifest keeps ``texts``.
    """
    out = []
    for k, (p, u, name, text) in enumerate(zip(points, outward, names, texts, strict=True)):
        label_obj = None
        if text is not None:
            label_obj = c.new_object(f"{prefix}/label:{role}_{name}", "label", LABEL_RGB,
                                     parent=f"{prefix}/labels:{role}", label_text=text)  # fmt: skip
            gap = 26.0 + 7.0 * max(len(text) - 1, 0)
            glyph = (drawn[k] if drawn is not None else None) or text
            c.text(label_obj, tuple(p + u * gap), glyph, font_px)
        out.append(TerminalEntry(terminal_id=f"{prefix}/terminal:{role}_{name}", role=cast(Role, role),
                                 box=box_at(p), label_obj=label_obj))  # fmt: skip
    return out


def render_tangle(
    c: LayeredCanvas,
    pay: dict[str, Any],
    paths: list[np.ndarray],
    crossings: list[dict[str, Any]],
    fault: FaultCode | None,
) -> list[TerminalEntry]:
    """Draw a tangle scene (cables, halos, terminals, labels) with an optional render fault."""
    n, width = pay["n"], float(pay["stroke"])
    colors = [tuple(x) for x in pay["colors"]]
    bg = c.background
    paths = [p.copy() for p in paths]
    if fault is FaultCode.M1:
        victim = pay["perm"][-1]
        paths[victim][-1, 0] = 1024.0 + 60.0
    beauty = {}
    if fault is FaultCode.M10:
        beauty[0] = blend(colors[0], bg, 0.08)
    if fault is FaultCode.M5:
        width = 1.0  # ~2 px footprint at 1024, 1.5 px at 768
    ids = draw_cables(c, "tangle", paths, crossings, colors, width, pay["style"],
                      halos=fault is not FaultCode.M2, beauty_rgb=beauty)  # fmt: skip
    if fault is FaultCode.M6 and crossings:
        x = crossings[len(crossings) // 2]
        occ = c.new_object("tangle/occluder:o", "occluder", (128, 128, 128))
        c.disk(occ, tuple(x["center"]), 2.0 * float(pay["stroke"]))
    if fault is FaultCode.M7:
        p = paths[0]
        k0, k1 = int(len(p) * 0.35), int(len(p) * 0.65)
        c.raw_beauty_polyline(
            p[k0:k1] + np.array([0.0, float(pay["stroke"]) + 9.0]), 4.0, (55, 55, 55)
        )
    del ids
    starts = [paths[s][0] for s in range(n)]
    s_out = [-unit(paths[s][1] - paths[s][0]) for s in range(n)]
    perm = pay["perm"]
    ends = [paths[perm[j]][-1] for j in range(n)]
    e_out = [unit(paths[perm[j]][-1] - paths[perm[j]][-2]) for j in range(n)]
    end_texts: list[str | None] = list(pay["end_labels"])
    drawn: list[str | None] | None = None
    if fault is FaultCode.M3:
        drawn = list(end_texts)
        drawn[0], drawn[1] = end_texts[1], end_texts[0]
    terms = add_terminal_labels(
        c, "tangle", "start", starts, s_out, list(pay["start_labels"]), list(pay["start_labels"])
    )
    terms += add_terminal_labels(
        c, "tangle", "end", ends, e_out, list(pay["end_labels"]), end_texts, drawn
    )
    return terms
