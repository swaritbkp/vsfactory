"""Trivial baselines a sound dataset must not reward (v2, SPEC D-39).

Each baseline is cross-validated like the learned probes and sees almost nothing:

* ``majority_class``   -- the most frequent training target of the item's family group;
* ``mean_luminance``   -- the most frequent target among training images of similar mean brightness;
* ``option_position``  -- always the answer position that was correct most often in training, pooled
  over every item with options (a position bias shared across families);
* ``text_length``      -- the most frequent target among training questions of similar length.

A baseline fails when its accuracy exceeds chance + `MARGIN` on at least `MIN_ITEMS` items.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Sequence
from typing import Any

import numpy as np
from PIL import Image

from knotwork.probes.items import Item

MARGIN = 0.02
MIN_ITEMS = 200
BUCKETS = 8
NAMES = ("majority_class", "mean_luminance", "option_position", "text_length")


def mean_luminance(item: Item) -> float:
    with Image.open(item.dir / "beauty.png") as im:
        small = im.convert("L").resize((64, 64), Image.Resampling.BOX)
    return float(np.asarray(small, dtype=np.float64).mean())


def _mode(values: Sequence[str]) -> str:
    counts = Counter(values)
    return min(counts, key=lambda t: (-counts[t], t)) if counts else "none"


def _bucketed(values: Sequence[float], train: list[int]) -> list[int]:
    """Quantile bucket of every item, with edges fitted on the training items only."""
    fitted = np.asarray([values[i] for i in train], dtype=np.float64)
    edges = np.unique(np.quantile(fitted, np.linspace(0, 1, BUCKETS + 1)[1:-1]))
    return [int(np.searchsorted(edges, v, side="right")) for v in values]


def _by_key(
    items: Sequence[Item], train: list[int], test: list[int], key: Callable[[int], Hashable]
) -> dict[int, str]:
    table: dict[tuple[str, Hashable], list[str]] = defaultdict(list)
    group: dict[str, list[str]] = defaultdict(list)
    for i in train:
        table[(items[i].group, key(i))].append(items[i].target)
        group[items[i].group].append(items[i].target)
    return {i: _mode(table.get((items[i].group, key(i))) or group[items[i].group]) for i in test}


def _option_position(items: Sequence[Item], train: list[int], test: list[int]) -> dict[int, str]:
    positions = Counter(items[i].option_index for i in train if items[i].options)
    ranked = sorted(positions, key=lambda p: (-positions[p], str(p)))
    out: dict[int, str] = {}
    for i in test:
        options = items[i].options or []
        pick = next((p for p in ranked if p is not None and p < len(options)), None)
        if pick is None:
            out[i] = "none"
        elif items[i].answer["type"] == "zone":
            out[i] = options[pick]
        elif items[i].answer["type"] == "label":
            out[i] = f"idx{pick}"
        else:
            out[i] = str(pick)
    return out


def predictions(items: Sequence[Item], fold: np.ndarray) -> dict[str, dict[int, str]]:
    """Out-of-fold predictions of every baseline."""
    light = [mean_luminance(it) for it in items]
    length = [float(len(it.text_with_options)) for it in items]
    out: dict[str, dict[int, str]] = {name: {} for name in NAMES}
    for f in sorted({int(v) for v in fold}):
        test = [i for i in range(len(items)) if fold[i] == f]
        train = [i for i in range(len(items)) if fold[i] != f]
        if not train:
            continue
        lum_bin, len_bin = _bucketed(light, train), _bucketed(length, train)
        out["majority_class"].update(_by_key(items, train, test, lambda _: 0))
        out["mean_luminance"].update(_by_key(items, train, test, lum_bin.__getitem__))
        out["text_length"].update(_by_key(items, train, test, len_bin.__getitem__))
        out["option_position"].update(_option_position(items, train, test))
    return out


def evaluate(items: Sequence[Item], fold: np.ndarray) -> dict[str, Any]:
    """Accuracy and chance of every baseline, plus the names that break the margin."""
    rows: dict[str, Any] = {}
    violations: list[str] = []
    for name, preds in predictions(items, fold).items():
        idx = sorted(preds)
        if name == "option_position":
            idx = [i for i in idx if items[i].options]
        if not idx:
            continue
        acc = sum(preds[i] == items[i].target for i in idx) / len(idx)
        chance = float(np.mean([items[i].chance for i in idx]))
        rows[name] = {"acc": round(acc, 6), "chance": round(chance, 6), "n": len(idx)}
        if len(idx) >= MIN_ITEMS and acc > chance + MARGIN:
            violations.append(name)
    return {"margin": MARGIN, "min_items": MIN_ITEMS, "probes": rows, "violations": violations}


__all__ = ["MARGIN", "MIN_ITEMS", "NAMES", "evaluate", "mean_luminance", "predictions"]
