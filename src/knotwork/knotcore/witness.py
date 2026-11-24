"""Beauty-to-mask witness: the pixels people see must agree with the masks the kernel reasons on
(SPEC 7.10). Checks: per-object contrast, crossing-centre and needle-tip colour, unexplained ink, OOB.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from knotwork.knotcore.contracts import CANVAS_PX, ErrorCode, WitnessCheck
from knotwork.knotcore.gauge import needle
from knotwork.knotcore.io import BoolArr, Bundle, KernelFault
from knotwork.knotcore.topo import STRUCT, crossings

E = ErrorCode


def _chk(name: str, oid: int | None, ok: bool, metric: float, thr: float, code: E) -> WitnessCheck:
    return WitnessCheck(
        name=name, obj_id=oid, passed=bool(ok), metric=float(metric), threshold=thr, code=code
    )


def _dil(m: BoolArr, n: int) -> BoolArr:
    return np.asarray(ndimage.binary_dilation(m, STRUCT, iterations=n), dtype=np.bool_)


def _patch(b: Bundle, xy: list[float]) -> float:
    x, y = round(xy[0]), round(xy[1])
    return float(b.lum[max(y - 2, 0) : y + 3, max(x - 2, 0) : x + 3].mean())


def witness(b: Bundle, cfg: dict[str, Any]) -> list[WitnessCheck]:
    w, out, inner = cfg["witness"], [], {}
    thr = float(w["min_contrast"])
    for oid in b.kind("strand") + b.kind("needle"):
        ys, xs = np.nonzero(b.ids == oid)
        if not len(xs):
            out.append(_chk("contrast", oid, False, 0.0, thr, E.E_BEAUTY_MISMATCH))
            continue
        p = int(w["ring_px"]) + 2
        win = (
            slice(max(ys.min() - p, 0), ys.max() + p + 1),
            slice(max(xs.min() - p, 0), xs.max() + p + 1),
        )
        v, lum = b.ids[win] == oid, b.lum[win]
        core = np.asarray(ndimage.binary_erosion(v, STRUCT), dtype=np.bool_)
        ring = _dil(v, int(w["ring_px"])) & ~_dil(v, 1)
        inner[oid] = float(lum[core if core.any() else v].mean())
        c = abs(inner[oid] - float(lum[ring].mean())) if ring.any() else 0.0
        out.append(_chk("contrast", oid, c >= thr, c, thr, E.E_BEAUTY_MISMATCH))
    points: list[tuple[str, list[float]]] = []
    for cl in b.claims.claims:
        if cl.op.value == "ONTOP" and len(cl.args) == 3:
            try:
                cs = crossings(b, str(cl.args[0]), str(cl.args[1]))
            except KernelFault:
                continue
            if isinstance(cl.args[2], int) and 0 <= cl.args[2] < len(cs):
                points.append(("crossing_color", cs[cl.args[2]]["center"]))
    for g in b.manifest.gauges:
        try:
            points.append(("needle_tip_color", needle(b, g, cfg)["tip"]))
        except KernelFault:
            continue
    for name, xy in points:
        top = int(b.ids[round(xy[1]), round(xy[0])])
        d = abs(_patch(b, xy) - inner[top]) if top in inner else 1.0
        out.append(_chk(name, top or None, d <= 0.10, d, 0.10, E.E_BEAUTY_MISMATCH))
    bg = np.asarray(b.manifest.background_rgb, dtype=np.int16)
    ink = np.abs(b.beauty.astype(np.int16) - bg).max(axis=2) > int(w["ink_delta"])
    stray = ink & ~_dil(b.ids > 0, int(w["ink_dilate_px"]))
    lab, n = ndimage.label(stray, structure=STRUCT)
    area = int(np.bincount(lab.ravel())[1:].max()) if n else 0
    limit = float(w["max_unexplained_area_px"])
    out.append(_chk("unexplained_ink", None, area <= limit, area, limit, E.E_UNEXPLAINED_INK))
    edge = np.ones((CANVAS_PX, CANVAS_PX), dtype=np.bool_)
    edge[2:-2, 2:-2] = False
    hit = sum(int((m & edge).any()) for m in b.masks.values())
    out.append(_chk("oob_mask", None, hit == 0, hit, 0.0, E.E_OOB))
    boxes = [o.box for o in b.manifest.objects if o.box] + [t.box for t in b.manifest.terminals]
    bad = sum(
        not (x.x0 >= 0 and x.y0 >= 0 and x.x1 <= CANVAS_PX and x.y1 <= CANVAS_PX) for x in boxes
    )
    out.append(_chk("oob_box", None, bad == 0, bad, 0.0, E.E_OOB))
    return out
