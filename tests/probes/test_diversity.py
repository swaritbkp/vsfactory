"""Task 3.5: Vendi score (SPEC 11.4) and the recolor-only baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from knotwork.diversity import parse_path, recolor, run_diversity, tokens, vendi


def test_vendi_identical_items_is_one() -> None:
    e = np.tile(np.array([[0.6, 0.8, 0.0]]), (50, 1))
    assert vendi(e) == pytest.approx(1.0, abs=1e-6)


def test_vendi_orthonormal_items_is_n() -> None:
    for n in (3, 17, 64):
        assert vendi(np.eye(n)) == pytest.approx(float(n), abs=1e-6)


def test_vendi_between_bounds() -> None:
    rng = np.random.default_rng(0)
    e = rng.normal(size=(40, 8))
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    assert 1.0 < vendi(e) < 8.0 + 1e-9


def test_recolor_changes_only_colours() -> None:
    pay = {"colors": [[10, 20, 30], [40, 50, 60]], "render": {"background_rgb": [200, 200, 200]},
           "dial": {"bezel_rgb": [100, 100, 100], "value": 42.0, "center": [1.0, 2.0]}, "n": 3}  # fmt: skip
    out = recolor(pay, np.random.default_rng(1))
    assert out["n"] == 3 and out["dial"]["value"] == 42.0 and out["dial"]["center"] == [1.0, 2.0]
    assert out["render"]["background_rgb"] != [200, 200, 200]
    assert len(out["colors"]) == 2 and pay["colors"] == [[10, 20, 30], [40, 50, 60]]


def test_tokens_and_paths() -> None:
    assert tokens("T1_TRACE", {"n_strands": 5, "layout": "strip", "style": "harness"}) == (
        "T1_TRACE", "n=5", "layout=strip", "style=harness")  # fmt: skip
    assert tokens("G1_READ", {"scale": "linear", "radius": 200.0})[1:] == (
        "n=1",
        "layout=linear",
        "style=-",
    )
    assert parse_path("s:tangle/s:slot/i:3/s:attempt/i:0") == ("tangle", "slot", 3, "attempt", 0)


def test_run_diversity_report(small_split: Path, tmp_path: Path) -> None:
    out = tmp_path / "div.json"
    assert run_diversity(small_split, out, n=16, seed=0) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    for plugin in ("tangle", "gauge", "panel"):
        row = rep["plugins"][plugin]
        assert row["n"] == min(16, {"tangle": 12, "gauge": 10, "panel": 14}[plugin])
        assert 1.0 <= row["vendi_recolor"] < row["vendi_ours"] <= row["n"]
        assert row["ratio"] == pytest.approx(row["vendi_ours"] / row["vendi_recolor"], rel=1e-3)
