"""Tolerance-aware logic over primitive outputs: COUNT, CMP, ARGMAX (SPEC 7.8).

Invariants: COUNT only counts objects with enough *visible* pixels in the id buffer; CMP and ARGMAX
treat values within the summed tolerances as indistinguishable (ARGMAX then refuses to answer).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from knotwork.knotcore.contracts import ErrorCode
from knotwork.knotcore.gauge import zone
from knotwork.knotcore.io import Bundle, KernelFault


def count(b: Bundle, kind: str, predicate: str, cfg: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Objects of `kind` with >= min_visible_px visible pixels satisfying `predicate`."""
    floor = int(cfg["count"]["min_visible_px"])
    hits: list[int] = []
    for oid in b.kind(kind):
        if int(np.count_nonzero(b.ids == oid)) < floor:
            continue
        if predicate.startswith("zone:"):
            parent = b.manifest.obj(oid).parent
            if parent is None or zone(b, parent, cfg)[0] != predicate[5:]:
                continue
        elif predicate != "any":
            raise KernelFault(ErrorCode.E_CLAIM_MISMATCH, f"unknown predicate {predicate}")
        hits.append(oid)
    return len(hits), {"objects": hits}


def cmp(x: list[float], y: list[float]) -> tuple[str, dict[str, Any]]:
    """'equal' within tol_x + tol_y, else 'greater' / 'less'."""
    if abs(x[0] - y[0]) <= x[1] + y[1]:
        return "equal", {}
    return ("greater" if x[0] > y[0] else "less"), {}


def argmax(pairs: list[list[Any]]) -> tuple[str, dict[str, Any]]:
    """Key of the largest [key, value, tol]; E_AMBIGUOUS_ARGMAX if the top two are within tol."""
    ranked = sorted(pairs, key=lambda p: float(p[1]), reverse=True)
    if not ranked:
        raise KernelFault(ErrorCode.E_AMBIGUOUS_ARGMAX, "no candidates")
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] <= ranked[0][2] + ranked[1][2]:
        raise KernelFault(ErrorCode.E_AMBIGUOUS_ARGMAX, f"top two within tolerance: {ranked[:2]}")
    return str(ranked[0][0]), {"ranked": ranked}
