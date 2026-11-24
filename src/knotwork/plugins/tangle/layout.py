"""Braid diagram layouts: strip and horseshoe (SPEC 8.1 "Layout"). Untrusted plugin code.

Strip: columns of (possibly weighted) width across ``[x0, x1]``; strand at position j sits at
``y_top + j * dy``; a strand in a crossing follows a cosine S-curve over its column, so the two
strands of a letter cross exactly at the column centre. A smooth jitter identical for every strand is
added to y (topology preserved). Horseshoe maps strip coordinates onto an annulus.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from knotwork.plugins.tangle.braid import Letter, columns, compose

SAMPLES_PER_COLUMN = 16
LEAD_PX = 24.0
HORSESHOE_CENTER = (512.0, 560.0)


@dataclass
class Layout:
    paths: list[np.ndarray]  # strand id -> (N, 2) polyline in canvas px
    crossings: list[dict[str, Any]]  # letter, a, b, over, eps, center [x, y]
    dy: float
    col_width: float
    n_cols: int
    meta: dict[str, Any] = field(default_factory=dict)


def jitter_fn(amps: list[float], freqs: list[float], phases: list[float], width: float) -> Any:
    def j(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=np.float64)
        for a, f, ph in zip(amps, freqs, phases, strict=True):
            out += a * np.sin(2 * math.pi * f * x / width + ph)
        return out

    return j


def strip(
    n: int,
    word: list[Letter],
    x0: float,
    x1: float,
    y_top: float,
    dy: float,
    jitter: Any,
    col_weights: dict[int, float] | None = None,
) -> Layout:
    """Strand polylines for ``word`` in strip coordinates (canvas px)."""
    cols = columns(n, word)
    weights = np.array(
        [(col_weights or {}).get(k, 1.0) for k in range(len(cols))], dtype=np.float64
    )
    edges = x0 + np.concatenate([[0.0], np.cumsum(weights)]) / weights.sum() * (x1 - x0)
    pos = list(range(n))  # position -> strand
    ys: dict[int, list[float]] = {s: [] for s in range(n)}
    xs: list[float] = []
    t = np.linspace(0.0, 1.0, SAMPLES_PER_COLUMN, endpoint=False)
    ease = (1 - np.cos(math.pi * t)) / 2
    _, braid_crossings = compose(n, word)
    by_letter = {c.letter: c for c in braid_crossings}
    crossings: list[dict[str, Any]] = []
    for k, col in enumerate(cols):
        xa, xb = float(edges[k]), float(edges[k + 1])
        xs.extend((xa + (xb - xa) * t).tolist())
        moving: dict[int, tuple[int, int]] = {}
        for idx in col:
            letter = word[idx]
            if letter is None:
                continue
            i = letter[0]
            moving[pos[i]] = (i, i + 1)
            moving[pos[i + 1]] = (i + 1, i)
            c = by_letter[idx]
            xc = (xa + xb) / 2
            crossings.append({"letter": idx, "a": c.a, "b": c.b, "over": c.over, "eps": c.eps,
                              "col": k, "center": [xc, y_top + (i + 0.5) * dy]})  # fmt: skip
        for j, s in enumerate(pos):
            src, dst = moving.get(s, (j, j))
            ys[s].extend((y_top + (src + (dst - src) * ease) * dy).tolist())
        for idx in col:
            letter = word[idx]
            if letter is not None:
                i = letter[0]
                pos[i], pos[i + 1] = pos[i + 1], pos[i]
    xs.append(x1)
    for j, s in enumerate(pos):
        ys[s].append(y_top + j * dy)
    xarr = np.asarray(xs)
    jit = jitter(xarr - x0)
    paths = [np.stack([xarr, np.asarray(ys[s]) + jit], axis=1) for s in range(n)]
    for item in crossings:
        item["center"][1] += float(jitter(np.array([item["center"][0] - x0]))[0])
    width = (x1 - x0) / max(len(cols), 1)
    return Layout(paths, crossings, dy, width, len(cols))


def to_horseshoe(lay: Layout, x0: float, x1: float, y_top: float, r_out: float) -> Layout:
    """Map strip coordinates onto the annulus theta = 200 - 220 (x - x0)/W, r = r_out - (y - y_top)."""
    cx, cy = HORSESHOE_CENTER
    w = x1 - x0

    def f(p: np.ndarray) -> np.ndarray:
        th = np.radians(200.0 - 220.0 * (p[:, 0] - x0) / w)
        r = r_out - (p[:, 1] - y_top)
        return np.stack([cx + r * np.cos(th), cy - r * np.sin(th)], axis=1)

    paths = [f(p) for p in lay.paths]
    crossings = [dict(c, center=f(np.array([c["center"]]))[0].tolist()) for c in lay.crossings]
    return Layout(paths, crossings, lay.dy, lay.col_width, lay.n_cols, dict(lay.meta))


def build(
    n: int, word: list[Letter], params: dict[str, Any], col_weights: dict[int, float] | None = None
) -> Layout:
    """Layout from scene params (kind, x0, x1, y_top, dy, r_out, jitter)."""
    jit = jitter_fn(params["jitter_amps"], params["jitter_freqs"], params["jitter_phases"],
                    params["x1"] - params["x0"])  # fmt: skip
    lay = strip(
        n, word, params["x0"], params["x1"], params["y_top"], params["dy"], jit, col_weights
    )
    lead = float(params.get("lead", LEAD_PX))
    for k, p in enumerate(lay.paths):  # straight lead-in/out so terminals sit on straight wire
        a, b = (
            np.array([[params["x0"] - lead, p[0, 1]]]),
            np.array([[params["x1"] + lead, p[-1, 1]]]),
        )
        lay.paths[k] = np.vstack([a, p, b])
    if params["kind"] == "horseshoe":
        lay = to_horseshoe(lay, params["x0"], params["x1"], params["y_top"], params["r_out"])
    return lay


def plan(n: int, n_cols: int, kind: str, stroke: float) -> dict[str, Any] | None:
    """Canvas geometry (x0, x1, y_top, dy, r_out) for a diagram, or None if it cannot be legible.

    Vertical spacing dy is kept within [0.6, 1.3] column widths so nominal crossing angles stay
    well above 35 degrees, and >= 3.6 strokes so labels and parallel strands stay apart.
    """
    x0, x1 = 136.0, 888.0
    w = (x1 - x0) / n_cols
    if kind == "strip":
        dy = min(760.0 / max(n - 1, 1), 1.3 * w)
        if dy < 0.6 * w:
            shrink = (0.6 * w - dy) / 0.6 * n_cols
            x0, x1, w = x0 + shrink / 2, x1 - shrink / 2, dy / 0.6
        y_top = 512.0 - dy * (n - 1) / 2
        out = {"kind": kind, "x0": x0, "x1": x1, "y_top": y_top, "dy": dy, "r_out": 0.0}
    else:
        r_out = 430.0
        w = 310.0 * math.radians(220.0) / n_cols  # column arc width at mid radius
        dy = min((r_out - 190.0) / max(n - 1, 1), 1.3 * w)
        out = {"kind": kind, "x0": x0, "x1": x1, "y_top": 0.0, "dy": dy, "r_out": r_out}
    if dy < max(3.6 * stroke, 34.0) or w < 3.4 * stroke:
        return None
    return out
