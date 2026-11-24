"""Topology primitives from amodal masks + modal id buffer: TRACE, CROSS, ONTOP (SPEC 7.1-7.4).

Invariants: strands are resolved from masks intersecting terminal boxes (never from plugin data);
crossings are 8-connected components of pairwise mask overlaps ordered by geodesic distance
(`MCP_Geometric`) from the start box; the top strand is read from the id buffer at the centre.
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


def stroke(b: Bundle, oid: int) -> float:
    """2 x median EDT on the skeleton of the object's mask (memoised)."""
    memo: dict[int, float] = b.memo.setdefault("stroke", {})
    if oid not in memo:
        m = b.masks[oid]
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
    """Geodesic distance inside the start strand's mask from its start-box pixels (memoised)."""
    memo: dict[str, Any] = b.memo.setdefault("geo", {})
    if start not in memo:
        a = b.masks[strand_of(b, start)]
        sy, sx = sl(b.terminal(start).box)
        ys, xs = np.nonzero(a[sy, sx])
        starts = list(zip((ys + sy.start).tolist(), (xs + sx.start).tolist(), strict=True))
        cost = np.where(a, 1.0, np.inf)
        memo[start] = np.asarray(MCP_Geometric(cost).find_costs(starts)[0], dtype=np.float64)
    out: np.ndarray[Any, np.dtype[np.float64]] = memo[start]
    return out


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
            lab, n = ndimage.label(b.masks[sa] & b.masks[o], structure=STRUCT)
            for k in range(1, n + 1):
                ys, xs = np.nonzero(lab == k)
                geo = float(d[ys, xs].min())
                c = [float(xs.mean()), float(ys.mean())]
                out.append({"other": o, "center": c, "geo": geo, "ys": ys, "xs": xs})
        memo[key] = sorted(out, key=lambda c: (c["geo"], c["center"][0], c["center"][1]))
    return memo[key]


def evidence(cs: list[Crossing]) -> list[dict[str, Any]]:
    return [{"other": c["other"], "center": c["center"], "geo": c["geo"]} for c in cs]


def cross(b: Bundle, a_start: str, b_start: str) -> tuple[int, dict[str, Any]]:
    cs = crossings(b, a_start, b_start)
    return len(cs), {"crossings": evidence(cs)}


def ontop(
    b: Bundle, a_start: str, b_start: str, idx: int, cfg: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    cs = crossings(b, a_start, b_start)
    if not 0 <= idx < len(cs):
        raise KernelFault(ErrorCode.E_CLAIM_MISMATCH, f"crossing {idx} of {len(cs)}")
    c, ida, idb = cs[idx], strand_of(b, a_start), strand_of(b, b_start)
    vis = float(np.isin(b.ids[c["ys"], c["xs"]], [ida, idb]).mean())
    top = int(b.ids[round(c["center"][1]), round(c["center"][0])])
    ev = {"center": c["center"], "visible_frac": vis, "top_obj": top}
    if vis < float(cfg["legibility"]["min_visible_crossing_frac"]) or top not in (ida, idb):
        raise KernelFault(ErrorCode.E_OCCLUDED_CROSSING, f"crossing {idx}: {ev}")
    return label_of(b, a_start if top == ida else b_start, cfg), ev
