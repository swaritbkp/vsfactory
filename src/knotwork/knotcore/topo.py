"""Topology primitives from amodal masks + modal id buffer: TRACE, CROSS, ONTOP (SPEC 7.1-7.4).

Invariants: strands are resolved from masks intersecting terminal boxes (never from plugin data);
crossings are 8-connected components of pairwise mask overlaps ordered by geodesic distance
(`MCP_Geometric`) from the start box; the top strand is the majority owner of the crossing core in
the id buffer (halo pixels do not vote), confirmed on the surrounding ring when the vote is close.
Per-object windows, pixel lists and pairwise overlaps are computed once per bundle.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage
from skimage.graph import MCP_Geometric
from skimage.morphology import skeletonize

from knotwork.knotcore.contracts import ErrorCode
from knotwork.knotcore.io import BoolArr, Bundle, KernelFault, sl
from knotwork.knotcore.ocr import label_text

STRUCT = np.ones((3, 3), dtype=np.bool_)
Crossing = dict[str, Any]
Pixels = tuple[np.ndarray[Any, np.dtype[np.int64]], np.ndarray[Any, np.dtype[np.int64]]]
#: crossing-core disk radius as a fraction of the thinner stroke (never below 2 px)
CORE_STROKE_FRAC = 0.35
#: a core vote at or above this share is decisive; below it the ring continuity must agree
DECISIVE_SHARE = 0.75


def window(b: Bundle, oid: int) -> tuple[slice, slice]:
    """Bounding box of the object's mask grown by one background pixel, clipped (memoised)."""
    memo: dict[int, tuple[slice, slice]] = b.memo.setdefault("window", {})
    if oid not in memo:
        m = b.masks[oid]
        ys, xs = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
        if not ys.size:
            memo[oid] = (slice(0, 1), slice(0, 1))
        else:
            y0, x0 = max(int(ys[0]) - 1, 0), max(int(xs[0]) - 1, 0)
            memo[oid] = (slice(y0, int(ys[-1]) + 2), slice(x0, int(xs[-1]) + 2))
    return memo[oid]


def pixels(b: Bundle, oid: int) -> Pixels:
    """(rows, cols) of the object's mask in canvas coordinates (memoised)."""
    memo: dict[int, Pixels] = b.memo.setdefault("pixels", {})
    if oid not in memo:
        sy, sx = window(b, oid)
        ys, xs = np.nonzero(b.masks[oid][sy, sx])
        memo[oid] = (ys + sy.start, xs + sx.start)
    return memo[oid]


def stroke(b: Bundle, oid: int) -> float:
    """2 x median EDT on the skeleton of the object's mask (memoised)."""
    memo: dict[int, float] = b.memo.setdefault("stroke", {})
    if oid not in memo:
        m = b.masks[oid][window(b, oid)]
        sk = skeletonize(m)  # type: ignore[no-untyped-call]
        edt = np.asarray(ndimage.distance_transform_edt(m), dtype=np.float64)
        memo[oid] = 2.0 * float(np.median(edt[sk])) if sk.any() else 0.0
    return memo[oid]


def strand_of(b: Bundle, tid: str) -> int:
    box = b.terminal(tid).box
    hits = [o for o in b.kind("strand") if b.masks[o][sl(box)].any()]
    if not hits:
        raise KernelFault(ErrorCode.E_TRACE_NO_TERMINAL, f"no strand at {tid}")
    if len(hits) > 1:
        raise KernelFault(ErrorCode.E_TRACE_AMBIGUOUS, f"{len(hits)} strands at {tid}")
    return hits[0]


def label_of(b: Bundle, tid: str, cfg: dict[str, Any]) -> str:
    """OCR'd label of a terminal (unlabelled terminals answer with their terminal id)."""
    t = b.terminal(tid)
    return tid if t.label_obj is None else label_text(b, t.label_obj, cfg)[0]


def trace_component(b: Bundle, start: str) -> tuple[int, BoolArr]:
    """(strand id, 8-connected component R of its mask that contains the start box)."""
    s = strand_of(b, start)
    lab, _ = ndimage.label(b.masks[s], structure=STRUCT)
    comps = set(np.unique(lab[sl(b.terminal(start).box)]).tolist()) - {0}
    if len(comps) != 1:
        raise KernelFault(ErrorCode.E_TRACE_AMBIGUOUS, f"start box spans {len(comps)} pieces")
    return s, np.asarray(lab == comps.pop(), dtype=np.bool_)


def trace(b: Bundle, start: str, cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    _, r = trace_component(b, start)
    ends = [
        t
        for t in b.manifest.terminals
        if t.terminal_id != start and t.role in ("end", "port") and r[sl(t.box)].any()
    ]
    if len(ends) != 1:
        code = ErrorCode.E_TRACE_AMBIGUOUS if ends else ErrorCode.E_TRACE_NO_TERMINAL
        raise KernelFault(code, f"{len(ends)} end terminals reached from {start}")
    text = label_of(b, ends[0].terminal_id, cfg)
    ev = {"path_px": int(r.sum()), "end_terminal": ends[0].terminal_id, "ocr_text": text}
    return text, ev


def geodesic(b: Bundle, start: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Geodesic distance inside the start strand's mask from its start-box pixels (memoised). The
    sweep runs on the strand's window only: no path can leave the mask, so distances are exact."""
    memo: dict[str, Any] = b.memo.setdefault("geo", {})
    if start not in memo:
        s = strand_of(b, start)
        win = window(b, s)
        a = b.masks[s][win]
        inside = np.zeros_like(b.masks[s])
        inside[sl(b.terminal(start).box)] = True
        ys, xs = np.nonzero(a & inside[win])
        starts = list(zip(ys.tolist(), xs.tolist(), strict=True))
        full = np.full(b.masks[s].shape, np.inf)
        full[win] = MCP_Geometric(np.where(a, 1.0, np.inf)).find_costs(starts)[0]
        memo[start] = full
    out: np.ndarray[Any, np.dtype[np.float64]] = memo[start]
    return out


def overlaps(b: Bundle, a: int, o: int) -> list[Crossing]:
    """8-connected components of mask[a] & mask[o], computed once per unordered pair."""
    memo: dict[tuple[int, int], list[Crossing]] = b.memo.setdefault("overlap", {})
    key = (min(a, o), max(a, o))
    if key not in memo:
        wa, wo = window(b, key[0]), window(b, key[1])
        y0, x0 = max(wa[0].start, wo[0].start), max(wa[1].start, wo[1].start)
        y1, x1 = min(wa[0].stop, wo[0].stop), min(wa[1].stop, wo[1].stop)
        found: list[Crossing] = []
        if y1 > y0 and x1 > x0:
            both = b.masks[a][y0:y1, x0:x1] & b.masks[o][y0:y1, x0:x1]
            lab, n = ndimage.label(both, structure=STRUCT)
            for k in range(1, n + 1):
                ys, xs = np.nonzero(lab == k)
                ys, xs = ys + y0, xs + x0
                c = [float(xs.mean()), float(ys.mean())]
                found.append({"center": c, "ys": ys, "xs": xs})
        memo[key] = found
    return memo[key]


def crossings(b: Bundle, a_start: str, b_start: str) -> list[Crossing]:
    """Crossings of strand(a) with strand(b) (or every other strand for "*"), geodesic order."""
    key = f"{a_start}|{b_start}"
    memo: dict[str, list[Crossing]] = b.memo.setdefault("cross", {})
    if key not in memo:
        sa = strand_of(b, a_start)
        others = (
            [o for o in b.kind("strand") if o != sa] if b_start == "*" else [strand_of(b, b_start)]
        )
        d, out = geodesic(b, a_start), []
        for o in others:
            for x in overlaps(b, sa, o):
                geo = float(d[x["ys"], x["xs"]].min())
                out.append({**x, "other": o, "geo": geo})
        memo[key] = sorted(out, key=lambda c: (c["geo"], c["center"][0], c["center"][1]))
    return memo[key]


def evidence(cs: list[Crossing]) -> list[dict[str, Any]]:
    return [{"other": c["other"], "center": c["center"], "geo": c["geo"]} for c in cs]


def cross(b: Bundle, a_start: str, b_start: str) -> tuple[int, dict[str, Any]]:
    cs = crossings(b, a_start, b_start)
    return len(cs), {"crossings": evidence(cs)}


def core_vote(b: Bundle, c: Crossing, ida: int, idb: int) -> dict[str, Any]:
    """Who owns the crossing: majority of the non-background id pixels in the core disk (the whole
    overlap component when neighbouring halos cover the core), plus how continuously each of the two
    strands stays visible on the ring around it."""
    widths = (stroke(b, ida), stroke(b, idb))
    r = max(2.0, CORE_STROKE_FRAC * min(widths))
    outer = r + max(widths)
    cx, cy = c["center"]
    y0, x0 = max(int(cy - outer) - 1, 0), max(int(cx - outer) - 1, 0)
    win = (slice(y0, int(cy + outer) + 2), slice(x0, int(cx + outer) + 2))
    ids = b.ids[win]
    yy, xx = np.ogrid[: ids.shape[0], : ids.shape[1]]
    dist = np.hypot(xx + x0 - cx, yy + y0 - cy)
    comp = b.ids[c["ys"], c["xs"]]
    shown = comp[comp > 0]
    votes = ids[(dist <= r) & (ids > 0)]
    if votes.size < 0.5 * int((dist <= r).sum()):
        votes = shown
    found, counts = np.unique(votes, return_counts=True)
    top = int(found[counts.argmax()]) if found.size else 0
    ring = (dist > r) & (dist <= outer)
    cont = {}
    for own, other in ((ida, idb), (idb, ida)):
        px = ring & b.masks[own][win] & ~b.masks[other][win]
        cont[own] = float((ids[px] == own).mean()) if px.any() else 1.0
    pair = int(np.isin(shown, [ida, idb]).sum())
    return {
        "top_obj": top,
        "share": float(counts.max() / votes.size) if votes.size else 0.0,
        "visible_frac": pair / shown.size if shown.size else 0.0,
        "coverage": pair / comp.size,
        "ring_continuity": [cont[ida], cont[idb]],
    }


def ontop(
    b: Bundle, a_start: str, b_start: str, idx: int, cfg: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    cs = crossings(b, a_start, b_start)
    if not 0 <= idx < len(cs):
        raise KernelFault(ErrorCode.E_CLAIM_MISMATCH, f"crossing {idx} of {len(cs)}")
    c, ida, idb = cs[idx], strand_of(b, a_start), strand_of(b, b_start)
    v = core_vote(b, c, ida, idb)
    ev = {"center": c["center"], **v}
    floor, top = float(cfg["legibility"]["min_visible_crossing_frac"]), v["top_obj"]
    hidden = v["visible_frac"] < floor or v["coverage"] < floor / 2
    if hidden or top not in (ida, idb) or v["share"] <= 0.5:
        raise KernelFault(ErrorCode.E_OCCLUDED_CROSSING, f"crossing {idx}: {ev}")
    won, lost = v["ring_continuity"] if top == ida else v["ring_continuity"][::-1]
    if v["share"] < DECISIVE_SHARE and won < lost:
        raise KernelFault(ErrorCode.E_OCCLUDED_CROSSING, f"crossing {idx} ring disagrees: {ev}")
    return label_of(b, a_start if top == ida else b_start, cfg), ev
