"""Panel geometry (SPEC 8.3): dial slots, a braid region on the left, and channel routing from braid
exits to gauge ports / blank pads. Untrusted.

Routing is Manhattan with one vertical channel per cable: exits and destinations are both sorted top
to bottom and matched in order, and channels are assigned so that among cables moving down the upper
one turns later, among cables moving up the lower one turns later. Under these rules no two routed
segments intersect, so every crossing in the panel is a braid crossing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from knotwork.plugins.gauge.render import bezel_width
from knotwork.plugins.tangle import braid, layout

DIAL_R = 120.0
BRAID_X0, BRAID_X1 = 136.0, 306.0  # plus 24-px straight leads: cables span 112..330
CHANNEL_X0, CHANNEL_GAP = 346.0, 16.0
PORT_GAP = 7  # dial mask -> port box: 8 px (GAUGE_OF dilation reach) and >= 1 stroke clearance
DEST_MIN_GAP = 46.0


def dial_slots(k: int) -> list[tuple[float, float]]:
    if k == 2:
        return [(860.0, 300.0), (860.0, 724.0)]
    if k == 3:
        return [(860.0, 200.0), (860.0, 512.0), (860.0, 824.0)]
    return [(630.0, 200.0), (870.0, 400.0), (630.0, 640.0), (870.0, 830.0)]


#: per-k (dx range, dy range) of the dial column shift (D-27): dials, G labels and pads stay on canvas
#: and clear of the routing channels
SLOT_SHIFT = {2: ((-90.0, 0.0), (-95.0, 75.0)), 3: ((-90.0, 0.0), (-40.0, 8.0)),
              4: ((-15.0, 0.0), (-40.0, 4.0))}  # fmt: skip
CANVAS = 1024.0


def shifted_slots(k: int, rng: np.random.Generator) -> tuple[list[tuple[float, float]], float]:
    """Dial centres moved by one common random (dx, dy), and that dx (pads move with the column)."""
    (ax, bx), (ay, by) = SLOT_SHIFT[min(k, 4)]
    dx, dy = float(rng.uniform(ax, bx)), float(rng.uniform(ay, by))
    return [(x + dx, y + dy) for x, y in dial_slots(k)], dx


def pad_x(k: int) -> float:
    return 480.0 if k == 4 else 600.0


def mirror_x(x: float) -> float:
    return CANVAS - x


def port_xy(center: tuple[float, float]) -> tuple[float, float]:
    """Port box centre left of the dial (box right edge PORT_GAP + 1 px left of the rim)."""
    return center[0] - DIAL_R - bezel_width(DIAL_R) - PORT_GAP - 10.0, center[1]


def braid_plan(n: int, word: list[braid.Letter], stroke: float) -> dict[str, Any] | None:
    ncols = max(len(braid.columns(n, word)), 1)  # an identity word still gets one straight column
    w = (BRAID_X1 - BRAID_X0) / ncols
    dy = min(1.3 * w, 64.0)
    if dy < max(0.6 * w, 3.6 * stroke, 30.0):
        return None
    return {"kind": "strip", "x0": BRAID_X0, "x1": BRAID_X1, "y_top": 512.0 - dy * (n - 1) / 2, "dy": dy,
            "r_out": 0.0, "jitter_amps": [0.0, 0.0, 0.0], "jitter_freqs": [1.0, 2.0, 3.0],
            "jitter_phases": [0.0, 0.0, 0.0]}  # fmt: skip


def pad_slots(
    rng: np.random.Generator, ports_y: list[float], pattern: list[bool]
) -> list[float] | None:
    """y of the blank pads so that the sorted destination list matches ``pattern`` (True = port).

    Pads between consecutive ports are spread inside the gap; None if a gap cannot hold its pads.
    """
    groups: list[int] = [0] * (len(ports_y) + 1)
    seen = 0
    for is_port in pattern:
        if is_port:
            seen += 1
        else:
            groups[seen] += 1
    bounds = [60.0, *ports_y, 964.0]
    out: list[float] = []
    for g, count in enumerate(groups):
        if not count:
            continue
        lo = bounds[g] + (DEST_MIN_GAP if g > 0 else 0.0)
        hi = bounds[g + 1] - (DEST_MIN_GAP if g < len(ports_y) else 0.0)
        if hi - lo < (count - 1) * DEST_MIN_GAP:
            return None
        slack = hi - lo - (count - 1) * DEST_MIN_GAP
        start = lo + float(rng.uniform(0.0, slack))
        out.extend(start + i * DEST_MIN_GAP for i in range(count))
    return out


def route(
    braid_paths: list[np.ndarray], exits: list[int], dests: list[tuple[float, float]]
) -> list[np.ndarray]:
    """Extend braid paths: exit position j -> destination j via a dedicated vertical channel.

    ``exits[j]`` is the strand at final braid position j; ``dests`` are sorted by y.
    """
    exit_y = {s: float(braid_paths[s][-1, 1]) for s in exits}
    down = [j for j, s in enumerate(exits) if dests[j][1] > exit_y[s] + 0.5]
    up = [j for j, s in enumerate(exits) if dests[j][1] < exit_y[s] - 0.5]
    chan: dict[int, float] = {}
    for r, j in enumerate(down):
        chan[j] = CHANNEL_X0 + (len(down) - 1 - r) * CHANNEL_GAP
    for r, j in enumerate(up):
        chan[j] = CHANNEL_X0 + r * CHANNEL_GAP
    out = [p.copy() for p in braid_paths]
    for j, s in enumerate(exits):
        x_end, y_end = dests[j]
        y0 = exit_y[s]
        xc = chan.get(j)
        pts = [[xc, y0], [xc, y_end]] if xc is not None else []
        tail = np.array([*pts, [x_end, y_end]], dtype=np.float64)
        seg = [out[s]] + [np.linspace(a, b, max(2, int(np.hypot(*(b - a)) // 6) + 1))[1:]
                          for a, b in zip(np.vstack([out[s][-1:], tail[:-1]]), tail, strict=True)]  # fmt: skip
        out[s] = np.vstack(seg)
    return out


def build_paths(
    pay: dict[str, Any],
) -> tuple[list[np.ndarray], list[dict[str, Any]], list[tuple[float, float]]]:
    """(cable paths, braid crossings, destinations sorted by y) for a panel payload."""
    n = pay["n"]
    word = [None if x is None else tuple(int(v) for v in x) for x in pay["word"]]
    lay = layout.build(n, word, pay["braid"])
    exits, _ = braid.compose(n, word)
    dests = [tuple(d) for d in pay["dests"]]
    paths, crossings = route(lay.paths, exits, dests), lay.crossings
    if pay.get("mirror"):  # whole composition mirrored left <-> right (D-27)
        paths = [np.column_stack([CANVAS - p[:, 0], p[:, 1]]) for p in paths]
        crossings = [dict(c, center=[mirror_x(c["center"][0]), c["center"][1]]) for c in crossings]
        dests = [(mirror_x(x), y) for x, y in dests]
    return paths, crossings, dests
