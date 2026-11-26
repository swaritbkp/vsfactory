"""Colour-vision-deficiency gate: wherever colour distinguishes things, the rendered colours must stay
apart for protanopes and deuteranopes.

Colours are measured from the beauty pixels (median over the visible core of each object); the
manifest only says which objects are meant to differ (distinct declared colours of strands, the
named zone arcs of a gauge). Simulation uses the Machado et al. (2009) severity-1.0 matrices in
linear RGB; separation is the CIE76 distance in CIELAB, and the worse of the two deficiencies counts.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import ndimage
from skimage.color import rgb2lab

from knotwork.knotcore.contracts import ErrorCode, WitnessCheck
from knotwork.knotcore.io import Bundle

Floats = npt.NDArray[np.float64]
MATRICES: dict[str, Floats] = {
    "protanopia": np.array(
        [
            [0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998],
        ]
    ),
    "deuteranopia": np.array(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ]
    ),
}
#: smallest CIE76 distance two semantically different colours may have after simulation
MIN_DELTA_E = 8.0
#: objects with fewer visible pixels carry no reliable colour
MIN_PIXELS = 12


def simulate(rgb: Any, kind: str) -> Floats:
    """sRGB values (0..255, shape (..., 3)) as seen with the deficiency `kind`, same scale."""
    x = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    seen = np.clip(linear @ MATRICES[kind].T, 0.0, 1.0)
    out = np.where(seen <= 0.0031308, 12.92 * seen, 1.055 * seen ** (1 / 2.4) - 0.055)
    return np.asarray(out * 255.0, dtype=np.float64)


def lab(rgb: Any) -> Floats:
    x = np.asarray(rgb, dtype=np.float64).reshape(1, -1, 3) / 255.0
    out = rgb2lab(x)
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)


def separation(a: Any, b: Any) -> float:
    """Worst-case simulated CIELAB distance between two sRGB colours."""
    pair = np.stack([np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)])
    worst = [lab(simulate(pair, kind)) for kind in MATRICES]
    return min(float(np.linalg.norm(p[0] - p[1])) for p in worst)


def min_separation(colors: Sequence[Any]) -> float:
    """Smallest pairwise `separation` (infinite for fewer than two colours)."""
    return min((separation(a, b) for a, b in combinations(colors, 2)), default=float("inf"))


def measured_color(b: Bundle, oid: int) -> Floats | None:
    """Median beauty colour of the visible core of an object, or None if it barely shows."""
    ys, xs = np.nonzero(b.ids == oid)
    if len(xs) < MIN_PIXELS:
        return None
    win = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    seen = b.ids[win] == oid
    core = np.asarray(ndimage.binary_erosion(seen), dtype=np.bool_)
    px = b.beauty[win][core if core.sum() >= MIN_PIXELS else seen]
    return np.asarray(np.median(px, axis=0), dtype=np.float64)


def _check(name: str, colors: list[Floats], floor: float) -> WitnessCheck:
    worst = min_separation(colors)
    return WitnessCheck(
        name=name,
        passed=worst >= floor,
        metric=min(worst, 1e9),
        threshold=floor,
        code=ErrorCode.E_CVD_AMBIGUOUS,
    )


def cvd_checks(b: Bundle, cfg: dict[str, Any]) -> list[WitnessCheck]:
    """One check per set of colours that carry meaning: the distinct strand colours (skipped when all
    cables share one colour) and the zone arcs of each gauge."""
    floor = float(cfg.get("cvd", {}).get("min_delta_e", MIN_DELTA_E))
    declared: dict[tuple[int, int, int], list[Floats]] = defaultdict(list)
    for oid in b.kind("strand"):
        seen = measured_color(b, oid)
        if seen is not None:
            declared[b.manifest.obj(oid).color_rgb].append(seen)
    out = []
    if len(declared) > 1:
        groups = [np.median(np.stack(v), axis=0) for _, v in sorted(declared.items())]
        out.append(_check("cvd_strands", groups, floor))
    for g in b.manifest.gauges:
        zones = [c for z in g.zones if (c := measured_color(b, z.zone_obj)) is not None]
        if len(zones) > 1:
            out.append(_check("cvd_zones", zones, floor))
    return out
