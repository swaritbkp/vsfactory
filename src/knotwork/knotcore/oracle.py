"""Beauty-only sanity oracle: an independent look at what a viewer sees.

The kernel's answers come from masks and the id buffer, which the (untrusted) renderer also writes.
A renderer that colluded with its own masks could describe objects that are not visible. This module
opens `beauty.png` itself and checks, using only coordinates the kernel hands it, that

* strands and needles show contrast against their surroundings along their claimed paths,
* every crossing and needle tip has local edge energy, and
* no declared label or tick is a ghost (a mask object with nothing visible under it).

It never reads masks or the id buffer and trusts no declared colour.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy import ndimage

from knotwork.knotcore.contracts import ErrorCode, WitnessCheck

Floats = npt.NDArray[np.float64]
DIRECTIONS = 16
SMOOTH_SIGMA = 0.8
DEFAULTS = {
    "min_delta": 0.05,  # luminance step between a path point and the median of its ring
    "min_range": 0.08,  # luminance range inside a spot that must contain an edge
    "min_strand_fraction": 0.7,
    "min_needle_fraction": 0.35,  # hubs and numeral plates legitimately cover part of a needle
}


@dataclass(frozen=True)
class PathProbe:
    """Points (x, y) along one object and the radius of the surrounding ring."""

    obj_id: int
    kind: str  # "strand" | "needle"
    points: Floats
    ring: float


@dataclass(frozen=True)
class SpotProbe:
    """A place that must show an edge: a crossing, a needle tip, a label or a tick."""

    obj_id: int | None
    name: str
    xy: tuple[float, float]
    radius: float


def load_luminance(root: Path) -> Floats:
    """Lightly smoothed Rec.709 luminance of `beauty.png`, read from disk by the oracle itself."""
    rgb = np.asarray(Image.open(root / "beauty.png").convert("RGB"), dtype=np.float64) / 255.0
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    return np.asarray(ndimage.gaussian_filter(lum, SMOOTH_SIGMA), dtype=np.float64)


def visible_fraction(lum: Floats, probe: PathProbe, min_delta: float) -> float:
    """Share of path points whose luminance differs from the median of the ring around them."""
    if not len(probe.points):
        return 0.0
    angle = np.linspace(0.0, 2 * np.pi, DIRECTIONS, endpoint=False)
    offsets = probe.ring * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    ring = probe.points[:, None, :] + offsets[None, :, :]
    h, w = lum.shape
    rx = np.clip(np.rint(ring[..., 0]).astype(np.int64), 0, w - 1)
    ry = np.clip(np.rint(ring[..., 1]).astype(np.int64), 0, h - 1)
    px = np.clip(np.rint(probe.points[:, 0]).astype(np.int64), 0, w - 1)
    py = np.clip(np.rint(probe.points[:, 1]).astype(np.int64), 0, h - 1)
    step = np.abs(lum[py, px] - np.median(lum[ry, rx], axis=1))
    return float((step >= min_delta).mean())


def local_range(lum: Floats, xy: tuple[float, float], radius: float) -> float:
    """max - min luminance in the square window around `xy` (0 when it is off the canvas)."""
    r = max(int(np.ceil(radius)), 1)
    x, y = int(round(xy[0])), int(round(xy[1]))
    win = lum[max(y - r, 0) : max(y + r + 1, 0), max(x - r, 0) : max(x + r + 1, 0)]
    return float(win.max() - win.min()) if win.size else 0.0


def inspect(
    root: Path, paths: list[PathProbe], spots: list[SpotProbe], cfg: dict[str, Any]
) -> list[WitnessCheck]:
    """Checks of every probe against the beauty image alone."""
    c = {**DEFAULTS, **cfg.get("oracle", {})}
    lum, out = load_luminance(root), []
    for p in paths:
        floor = float(c[f"min_{p.kind}_fraction"])
        got = visible_fraction(lum, p, float(c["min_delta"]))
        out.append(
            WitnessCheck(
                name="oracle_path",
                obj_id=p.obj_id,
                passed=got >= floor,
                metric=got,
                threshold=floor,
                code=ErrorCode.E_ORACLE_GHOST,
            )  # fmt: skip
        )
    floor = float(c["min_range"])
    for s in spots:
        got = local_range(lum, s.xy, s.radius)
        out.append(
            WitnessCheck(
                name=s.name,
                obj_id=s.obj_id,
                passed=got >= floor,
                metric=got,
                threshold=floor,
                code=ErrorCode.E_ORACLE_GHOST,
            )  # fmt: skip
        )
    return out
