"""`gbdt_image_stats` probe (SPEC 11.2): gradient-boosted trees on global image statistics only --
16-bin histograms per RGB channel, per-channel mean/std, Sobel edge density and colourfulness.
No spatial features, so it can only exploit global appearance leaks (e.g. busier image => larger
answer)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

FEATURE_PX = 512
EDGE_THRESHOLD = 0.15  # Sobel magnitude on [0, 1] luminance counted as an edge


def image_stats(rgb: np.ndarray) -> np.ndarray:
    """56 global statistics of an RGB uint8 image (48 histogram bins, 6 moments, edges, colour)."""
    x = rgb.astype(np.float64) / 255.0
    hists = [
        np.histogram(x[..., c], bins=16, range=(0.0, 1.0))[0] / x[..., c].size for c in range(3)
    ]
    moments = [x[..., c].mean() for c in range(3)] + [x[..., c].std() for c in range(3)]
    lum = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    mag = np.hypot(ndimage.sobel(lum, axis=0), ndimage.sobel(lum, axis=1)) / 8.0
    edge = float((mag > EDGE_THRESHOLD).mean())
    rg, yb = x[..., 0] - x[..., 1], 0.5 * (x[..., 0] + x[..., 1]) - x[..., 2]
    colourful = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    return np.concatenate([*hists, moments, [edge, colourful]]).astype(np.float64)


def load_features(paths: Sequence[Path]) -> np.ndarray:
    rows = []
    for p in paths:
        with Image.open(p) as im:
            small = im.convert("RGB").resize((FEATURE_PX, FEATURE_PX), Image.Resampling.BOX)
        rows.append(image_stats(np.asarray(small)))
    return np.vstack(rows) if rows else np.zeros((0, 56))


def fit_predict(
    x_train: np.ndarray, y_train: Sequence[str], x_test: np.ndarray, seed: int
) -> list[str]:
    """Train on one fold, predict the held-out fold (constant prediction if one class only)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    classes = sorted(set(y_train))
    if len(classes) == 1:
        return [classes[0]] * len(x_test)
    clf = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.1, max_leaf_nodes=15, random_state=seed % (2**31)
    )
    clf.fit(x_train, np.asarray(y_train))
    return [str(v) for v in clf.predict(x_test)]


__all__ = ["fit_predict", "image_stats", "load_features"]
