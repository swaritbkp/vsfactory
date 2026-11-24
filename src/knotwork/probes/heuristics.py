"""Geometric / colour shortcut heuristics (SPEC 11.2). They look only at what a VLM could see
(terminal positions and beauty pixels; terminal boxes and printed label strings come from the
declarative manifest) -- never at masks, the id buffer, claims or the certificate trace.

Each function returns the predicted terminal *label* (an end label for tangle T1, a gauge label
``Gk`` for panel trace families) or None when it does not apply.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from knotwork.probes.items import Item

PATCH = 9  # 9x9 colour patch
RING_IN, RING_OUT = 3, 16  # search band around a terminal box for the entering stroke


def _labels(manifest: dict[str, Any]) -> dict[int, str]:
    return {o["obj_id"]: o.get("label_text") or "" for o in manifest["objects"]}


def terminals(manifest: dict[str, Any], role: str) -> list[tuple[str, str, dict[str, int]]]:
    """(terminal_id, printed label, box) of every terminal with ``role``, in manifest (slot) order."""
    labels = _labels(manifest)
    return [
        (t["terminal_id"], labels.get(t.get("label_obj") or -1, ""), t["box"])
        for t in manifest["terminals"]
        if t["role"] == role
    ]


def centre(box: dict[str, int]) -> np.ndarray:
    return np.array([(box["x0"] + box["x1"]) / 2.0, (box["y0"] + box["y1"]) / 2.0])


def start_terminal(it: Item) -> str | None:
    """Terminal the question starts from (first TRACE argument of the program)."""
    for st in it.program:
        if st["op"] == "TRACE":
            return str(st["args"][0])
    return None


def _targets(it: Item) -> list[tuple[str, str, dict[str, int]]]:
    return terminals(it.manifest, "port" if it.plugin == "panel" else "end")


def _start_box(it: Item) -> dict[str, int] | None:
    sid = start_terminal(it)
    for tid, _, box in terminals(it.manifest, "start"):
        if tid == sid:
            return box
    return None


def straight_across(it: Item) -> str | None:
    """End label at the start terminal's position index (tangle T1 only)."""
    if it.plugin != "tangle" or it.family != "T1_TRACE":
        return None
    starts = [tid for tid, _, _ in terminals(it.manifest, "start")]
    ends = terminals(it.manifest, "end")
    sid = start_terminal(it)
    if sid not in starts:
        return None
    k = starts.index(sid)
    return ends[k][1] if k < len(ends) else None


def nearest(it: Item) -> str | None:
    """Target terminal whose box centre is nearest (Euclidean) to the start terminal box centre."""
    box = _start_box(it)
    targets = _targets(it)
    if box is None or not targets:
        return None
    c = centre(box)
    d = [float(np.linalg.norm(centre(b) - c)) for _, _, b in targets]
    return targets[int(np.argmin(d))][1]


def entry_colour(img: np.ndarray, box: dict[str, int], bg: np.ndarray) -> np.ndarray:
    """Mean colour of the 9x9 patch centred on the most background-different pixel in a band
    around ``box`` (where the stroke enters the terminal)."""
    h, w = img.shape[:2]
    x0, y0 = max(0, box["x0"] - RING_OUT), max(0, box["y0"] - RING_OUT)
    x1, y1 = min(w, box["x1"] + RING_OUT), min(h, box["y1"] + RING_OUT)
    region = img[y0:y1, x0:x1].astype(np.float64)
    diff = np.abs(region - bg).max(axis=2)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inner = (
        (xx >= box["x0"] - RING_IN)
        & (xx < box["x1"] + RING_IN)
        & (yy >= box["y0"] - RING_IN)
        & (yy < box["y1"] + RING_IN)
    )
    # centre the patch where a 9x9 window holds the most ink (on the stroke, not its edge)
    score = ndimage.uniform_filter(diff, size=PATCH, mode="constant")
    score[inner] = -1.0
    py, px = np.unravel_index(int(np.argmax(score)), score.shape)
    cy, cx, r = py + y0, px + x0, PATCH // 2
    patch = img[max(0, cy - r) : cy + r + 1, max(0, cx - r) : cx + r + 1].astype(np.float64)
    return np.asarray(patch.reshape(-1, 3).mean(axis=0), dtype=np.float64)


def color_match(it: Item, img: np.ndarray | None = None) -> str | None:
    """Target terminal whose entering-stroke colour is closest (RGB L2) to the start's."""
    box = _start_box(it)
    targets = _targets(it)
    if box is None or not targets:
        return None
    if img is None:
        with Image.open(it.dir / "beauty.png") as im:
            img = np.asarray(im.convert("RGB"))
    bg = np.asarray(it.manifest["background_rgb"], dtype=np.float64)
    c0 = entry_colour(img, box, bg)
    d = [float(np.linalg.norm(entry_colour(img, b, bg) - c0)) for _, _, b in targets]
    return targets[int(np.argmin(d))][1]


def true_target_label(it: Item) -> str | None:
    """Label the TRACE step actually reached (kernel trace), used only to score trace hits."""
    for s in it.steps:
        if s["op"] == "TRACE":
            return str(s["output"])
    return None


HEURISTICS = {"straight_across": straight_across, "nearest": nearest, "color_match": color_match}

#: families each heuristic is evaluated on (SPEC 11.2)
APPLIES = {
    "straight_across": {"T1_TRACE"},
    "nearest": {"T1_TRACE", "P1_READ_TRACE", "P2_ZONE_TRACE"},
    "color_match": {"T1_TRACE", "P1_READ_TRACE", "P2_ZONE_TRACE"},
}

__all__ = [
    "APPLIES",
    "HEURISTICS",
    "color_match",
    "entry_colour",
    "nearest",
    "start_terminal",
    "straight_across",
    "terminals",
    "true_target_label",
]
