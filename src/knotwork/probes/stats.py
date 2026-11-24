"""Statistics helpers for the shortcut audit (SPEC 11.2-11.3): Wilson bound, chi-square balance,
64-bit pHash near-duplicate counting."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy import stats as sps

Z95 = 1.959963984540054


def wilson_upper(k: int, n: int, z: float = Z95) -> float:
    """Upper end of the Wilson score interval for k successes out of n (1.0 when n == 0)."""
    if n <= 0:
        return 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
    return (centre + half) / denom


def chi2_uniform(counts: Sequence[int]) -> dict[str, float]:
    """Chi-square goodness of fit of ``counts`` against the uniform distribution."""
    obs = np.asarray(counts, dtype=np.float64)
    dof = len(obs) - 1
    if dof < 1 or obs.sum() <= 0:
        return {"chi2": 0.0, "dof": float(max(dof, 0)), "p": 1.0}
    res = sps.chisquare(obs)
    return {"chi2": float(res.statistic), "dof": float(dof), "p": float(res.pvalue)}


def phash_bits(path: Path) -> np.ndarray:
    """64-bit perceptual hash (imagehash.phash, hash_size 8) as a bool vector."""
    import imagehash
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(imagehash.phash(im.convert("RGB")).hash, dtype=bool).reshape(-1)


def near_duplicate_pairs(bits: np.ndarray, max_hamming: int = 4) -> int:
    """Number of unordered pairs (i < j) whose Hamming distance is <= ``max_hamming``."""
    b = np.asarray(bits, dtype=np.uint8)
    if len(b) < 2:
        return 0
    dist = (b[:, None, :] != b[None, :, :]).sum(axis=2)
    iu = np.triu_indices(len(b), k=1)
    return int((dist[iu] <= max_hamming).sum())


def majority_freq(values: Sequence[str]) -> float:
    """Frequency of the most common value (0.0 for an empty sequence)."""
    if not values:
        return 0.0
    _, counts = np.unique(np.asarray(values, dtype=object).astype(str), return_counts=True)
    return float(counts.max() / len(values))


__all__ = [
    "Z95",
    "chi2_uniform",
    "majority_freq",
    "near_duplicate_pairs",
    "phash_bits",
    "wilson_upper",
]
