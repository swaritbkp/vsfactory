"""Trivial baselines and the binary premise task (v2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from knotwork.probes import baselines
from knotwork.probes.items import Item, assign_targets


def _items(root: Path, n: int, leak: bool) -> list[Item]:
    """Premise items whose truth value leaks through image brightness only when `leak`."""
    out = []
    for i in range(n):
        holds = i % 2 == 0
        level = (230 if holds else 120) if leak else 128
        d = root / f"item{i}"
        d.mkdir(parents=True)
        Image.new("RGB", (16, 16), (level, level, level)).save(d / "beauty.png")
        value = {"premise_holds": holds, "value": 40.0}
        out.append(
            Item(
                bundle_id=f"b{i}",
                plugin="panel",
                family="P5_PREMISE_CONTROL" if holds else "T_GASLIGHT",
                scene_id=f"s{i}",
                dir=d,
                text="Since the gauge on socket A is in the red zone, what value does it read?",
                options=None,
                answer={"type": "premise", "value": value},
                manifest={"gauges": []},
                steps=[],
                program=[],
                difficulty={},
            )  # fmt: skip
        )
    assign_targets(out)
    return out


def test_premise_targets_are_the_binary_truth_token(tmp_path: Path) -> None:
    items = _items(tmp_path, 4, leak=False)
    assert [it.target for it in items] == ["yes", "no", "yes", "no"]
    assert all(it.chance == 0.5 and it.group == "PREMISE" for it in items)


def test_brightness_leak_is_reported_and_clean_data_is_not(tmp_path: Path) -> None:
    n = baselines.MIN_ITEMS
    fold = np.array([(i // 2) % 5 for i in range(n)])
    leaky = baselines.evaluate(_items(tmp_path / "leak", n, leak=True), fold)
    assert "mean_luminance" in leaky["violations"]
    clean = baselines.evaluate(_items(tmp_path / "clean", n, leak=False), fold)
    assert clean["violations"] == []
    assert clean["probes"]["majority_class"]["acc"] <= 0.5 + baselines.MARGIN
