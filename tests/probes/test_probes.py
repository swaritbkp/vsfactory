"""Each shortcut probe has power: it beats chance on a deliberately leaky synthetic dataset.
Statistics helpers match reference values."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw

from knotwork.probes import gbdt, heuristics, text_only
from knotwork.probes.items import Item, assign_targets, value_bin
from knotwork.probes.run import fold_ids, heuristic_predictions, learned_predictions
from knotwork.probes.stats import (
    chi2_uniform,
    majority_freq,
    near_duplicate_pairs,
    phash_bits,
    wilson_upper,
)

BG = (246, 244, 238)
PALETTE = [(200, 30, 30), (30, 160, 40), (30, 60, 200), (220, 160, 0), (140, 0, 160), (0, 150, 150)]


def _box(x: int, y: int) -> dict[str, int]:
    return {"x0": x - 10, "x1": x + 10, "y0": y - 10, "y1": y + 10}


def _tangle_item(tmp: Path, i: int, rng: np.random.Generator, leak: str, n: int = 5) -> Item:
    """A T1 item with n start terminals (left) and n end terminals (right); ``leak`` decides the
    answer: 'straight' (same index), 'nearest' (vertically shifted ends) or 'color' (stroke colours)."""
    ys = [150 + 160 * k for k in range(n)]
    perm = rng.permutation(n)
    k0 = int(rng.integers(n))
    end_labels = [str(j + 1) for j in range(n)]
    if leak == "straight":
        ans = k0
        end_ys = ys
    elif leak == "nearest":
        ans = int(perm[0])
        k0 = ans
        end_ys = ys
    else:
        ans = int(perm[k0])
        end_ys = ys
    starts_x = 100 if leak != "nearest" else 450
    img = Image.new("RGB", (1024, 1024), BG)
    dr = ImageDraw.Draw(img)
    for k in range(n):
        colour = PALETTE[k % len(PALETTE)]
        target = int(perm[k]) if leak == "color" else k
        dr.line([(starts_x + 12, ys[k]), (900 - 12, end_ys[target])], fill=colour, width=9)
    d = tmp / f"b{i}"
    d.mkdir(parents=True)
    img.save(d / "beauty.png")
    objects = [{"obj_id": 100 + k, "label_text": end_labels[k]} for k in range(n)]
    terminals = [
        {"terminal_id": f"tangle/terminal:start_{k}", "role": "start", "box": _box(starts_x, ys[k])}
        for k in range(n)
    ] + [
        {
            "terminal_id": f"tangle/terminal:end_{k}",
            "role": "end",
            "box": _box(900, end_ys[k]),
            "label_obj": 100 + k,
        }
        for k in range(n)
    ]
    return Item(
        bundle_id=f"kw-{i:020d}",
        plugin="tangle",
        family="T1_TRACE",
        scene_id=f"{i:064x}",
        dir=d,
        text=f"Follow the cable from terminal {k0}.",
        options=end_labels,
        answer={"type": "label", "value": end_labels[ans], "canonical": end_labels[ans]},
        manifest={"objects": objects, "terminals": terminals, "background_rgb": list(BG)},
        steps=[
            {"op": "TRACE", "inputs": [f"tangle/terminal:start_{k0}"], "output": end_labels[ans]}
        ],
        program=[{"op": "TRACE", "args": [f"tangle/terminal:start_{k0}"]}],
        difficulty={"n": float(n)},
    )


def _acc(items: list[Item], preds: dict[int, str]) -> float:
    return sum(preds[i] == items[i].target for i in preds) / len(preds)


@pytest.mark.parametrize(
    ("leak", "name"),
    [("straight", "straight_across"), ("nearest", "nearest"), ("color", "color_match")],
)
def test_heuristic_has_power(tmp_path: Path, leak: str, name: str) -> None:
    rng = np.random.default_rng(3)
    items = [_tangle_item(tmp_path, i, rng, leak) for i in range(30)]
    assign_targets(items)
    preds, _ = heuristic_predictions(items, fold_ids(items, 5), name)
    assert len(preds) == 30
    assert _acc(items, preds) >= 0.9 > 0.2 + 0.05  # chance = 1/5


def test_nearest_fails_without_leak(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    items = [_tangle_item(tmp_path, i, rng, "color") for i in range(40)]
    assign_targets(items)
    preds, _ = heuristic_predictions(items, fold_ids(items, 5), "nearest")
    assert _acc(items, preds) < 0.5


def _plain_item(tmp: Path, i: int, target: str, text: str, shade: int) -> Item:
    d = tmp / f"g{i}"
    d.mkdir(parents=True)
    Image.new("RGB", (64, 64), (shade, shade, shade)).save(d / "beauty.png")
    return Item(
        bundle_id=f"kw-{i:020d}",
        plugin="gauge",
        family="G2_ZONE",
        scene_id=f"{i:064x}",
        dir=d,
        text=text,
        options=["green", "amber", "red"],
        answer={"type": "zone", "value": target, "canonical": target},
        manifest={"objects": [], "terminals": [], "gauges": []},
        steps=[],
        program=[],
        difficulty={},
    )


def test_gbdt_has_power_on_global_leak(tmp_path: Path) -> None:
    zones = ["green", "amber", "red"]
    items = [
        _plain_item(tmp_path, i, zones[i % 3], "Which zone?", 60 + 60 * (i % 3) + (i % 7))
        for i in range(60)
    ]
    assign_targets(items)
    feats = gbdt.load_features([it.dir / "beauty.png" for it in items])
    assert feats.shape == (60, 56)
    y = [it.target for it in items]

    def fit(tr: list[int], te: list[int], seed: int) -> list[str]:
        return gbdt.fit_predict(feats[tr], [y[i] for i in tr], feats[te], seed)

    preds = learned_predictions(items, fold_ids(items, 5), 0, "gbdt_image_stats", fit)
    assert _acc(items, preds) >= 0.9 > 1 / 3 + 0.05


def test_text_only_has_power_on_text_leak(tmp_path: Path) -> None:
    zones = ["green", "amber", "red"]
    items = [
        _plain_item(tmp_path, i, zones[i % 3], f"Hint {zones[i % 3]} item {i}.", 128)
        for i in range(60)
    ]
    assign_targets(items)
    texts = [it.text_with_options for it in items]
    y = [it.target for it in items]

    def fit(tr: list[int], te: list[int], seed: int) -> list[str]:
        return text_only.fit_predict(
            [texts[i] for i in tr], [y[i] for i in tr], [texts[i] for i in te], seed
        )

    preds = learned_predictions(items, fold_ids(items, 5), 0, "text_only", fit)
    assert _acc(items, preds) >= 0.9


def test_text_only_near_chance_without_leak(tmp_path: Path) -> None:
    zones = ["green", "amber", "red"]
    rng = np.random.default_rng(9)
    items = [
        _plain_item(tmp_path, i, zones[int(rng.integers(3))], "Which zone is the needle in?", 128)
        for i in range(90)
    ]
    assign_targets(items)
    texts = [it.text_with_options for it in items]
    y = [it.target for it in items]

    def fit(tr: list[int], te: list[int], seed: int) -> list[str]:
        return text_only.fit_predict(
            [texts[i] for i in tr], [y[i] for i in tr], [texts[i] for i in te], seed
        )

    preds = learned_predictions(items, fold_ids(items, 5), 0, "text_only", fit)
    assert _acc(items, preds) < 0.6


def test_folds_group_by_scene(tmp_path: Path) -> None:
    items = [_plain_item(tmp_path, i, "red", "Which zone?", 100) for i in range(12)]
    for i, it in enumerate(items):
        it.scene_id = f"{i // 3:064x}"  # 4 scenes x 3 items
    f = fold_ids(items, 5)
    for s in range(4):
        assert len({int(f[3 * s + j]) for j in range(3)}) == 1


def test_wilson_reference_values() -> None:
    assert wilson_upper(5, 10) == pytest.approx(0.76340, abs=1e-4)
    assert wilson_upper(0, 10) == pytest.approx(0.27753, abs=1e-4)
    assert wilson_upper(50, 100) == pytest.approx(0.59617, abs=1e-4)
    assert wilson_upper(0, 0) == 1.0


def test_chi2_reference_values() -> None:
    r = chi2_uniform([10, 10, 10])
    assert r["chi2"] == 0.0 and r["p"] == pytest.approx(1.0) and r["dof"] == 2
    r = chi2_uniform([20, 10, 0])
    assert r["chi2"] == pytest.approx(20.0)
    assert r["p"] == pytest.approx(math.exp(-10.0), rel=1e-6)


def test_phash_and_dedup(tmp_path: Path) -> None:
    a = Image.new("RGB", (256, 256), BG)
    ImageDraw.Draw(a).rectangle([20, 20, 120, 200], fill=(10, 10, 10))
    b = Image.new("RGB", (256, 256), BG)
    ImageDraw.Draw(b).ellipse([100, 30, 250, 140], fill=(10, 10, 10))
    a.save(tmp_path / "a.png")
    a.save(tmp_path / "a2.png")
    b.save(tmp_path / "b.png")
    bits = np.vstack([phash_bits(tmp_path / f) for f in ("a.png", "a2.png", "b.png")])
    assert bits.shape == (3, 64)
    assert near_duplicate_pairs(bits, 4) == 1
    assert near_duplicate_pairs(bits[:1], 4) == 0


def test_value_bins_and_majority() -> None:
    assert value_bin(0, 0, 80) == 0 and value_bin(80, 0, 80) == 9 and value_bin(41, 10, 90) == 3
    assert majority_freq(["a", "b", "a", "c"]) == 0.5 and majority_freq([]) == 0.0


def test_chance_levels(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    items = [_tangle_item(tmp_path, i, rng, "straight", n=4) for i in range(4)]
    assign_targets(items)
    assert all(it.chance == 0.25 and it.target.startswith("idx") for it in items)
    num: list[Any] = []
    for i, v in enumerate([5.0, 6.0, 50.0, 75.0]):
        it = _plain_item(tmp_path, 100 + i, "red", "Read it.", 1)
        it.family, it.answer = "G1_READ", {"type": "float", "value": v, "canonical": str(v)}
        it.manifest = {
            "objects": [{"obj_id": 1, "label_text": "00"}, {"obj_id": 2, "label_text": "80"}],
            "gauges": [
                {"gauge_id": "gauge/gauge:g1", "ticks": [{"label_obj": 1}, {"label_obj": 2}]}
            ],
        }
        it.steps = [{"op": "READ", "inputs": ["gauge/gauge:g1"], "output": [v, 0.3]}]
        num.append(it)
    assign_targets(num)
    assert [it.target for it in num] == ["bin0", "bin0", "bin6", "bin9"]
    assert all(it.chance == 0.5 for it in num)
    assert heuristics.straight_across(num[0]) is None
