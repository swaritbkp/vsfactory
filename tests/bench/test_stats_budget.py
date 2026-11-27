"""Task 5.4: statistics recover known parameters; budget preflight and stop rule (SPEC 12.3, 12.5)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from knotwork.bench import stats
from knotwork.bench.budget import Budget, actual_cost, est_call_usd, preflight
from knotwork.bench.client_base import BudgetExhausted
from knotwork.bench.run import Item, run_staircase

REPO = Path(__file__).resolve().parents[2]


def simulate(p: float, n: int, rng: np.random.Generator) -> list[dict[str, object]]:
    k = rng.integers(1, 13, size=n)
    m = rng.integers(3, 11, size=n).astype(float)
    s = (1 - p) ** k
    correct = rng.random(n) < s + (1 - s) / m
    return [
        {"k": int(a), "m": float(b), "correct": bool(c)}
        for a, b, c in zip(k, m, correct, strict=True)
    ]


@pytest.mark.parametrize("p", [0.04, 0.08, 0.15])
def test_mle_recovers_p(p: float) -> None:
    trials = simulate(p, 2000, np.random.default_rng(11))
    k = [t["k"] for t in trials]
    m = [t["m"] for t in trials]
    c = [t["correct"] for t in trials]
    assert abs(stats.mle_p(k, m, c) - p) < 0.02


def test_bootstrap_seeded_and_covering() -> None:
    trials = simulate(0.08, 400, np.random.default_rng(3))
    a = stats.per_crossing(trials, np.random.default_rng(7))
    b = stats.per_crossing(trials, np.random.default_rng(7))
    assert a == b and a["n"] == 400
    lo, hi = a["ci"]
    assert lo < a["p_hat"] < hi and lo <= 0.08 + 0.015 and hi >= 0.08 - 0.015


def test_abbott() -> None:
    assert stats.abbott(0.5, 0.25) == pytest.approx(1 / 3)
    assert stats.abbott(0.25, 0.25) == 0.0 and stats.abbott(1.0, 0.2) == 1.0


def test_composition_product_law_holds_for_independent_errors() -> None:
    rng = np.random.default_rng(5)
    rows = []
    for _ in range(400):
        trace_ok, read_ok = rng.random() < 0.7, rng.random() < 0.8
        k = 3
        rows.append({"trace": trace_ok or rng.random() < 1 / k, "read": read_ok, "comp": trace_ok and read_ok,
                     "c_trace": 1 / k, "c_read": 0.0, "c_comp": 0.0})  # fmt: skip
    res = stats.composition(rows, np.random.default_rng(1))
    lo, hi = res["abbott"]["gap_ci"]
    assert lo <= 0.0 <= hi
    assert res["abbott"]["trace"] == pytest.approx(0.7, abs=0.07)


def test_small_blocks() -> None:
    assert stats.color_ablation([(True, False), (True, True)]) == {
        "n": 2,
        "acc_unique": 1.0,
        "acc_mono": 0.5,
        "delta": 0.5,
    }
    assert (
        stats.track_switch([("3", {"3", "5"}), ("4", {"5"})])["single_switch_explained_frac"] == 0.5
    )
    t = stats.twins([(True, True, "A", "A"), (True, False, "B", "C")])
    assert t == {"n": 2, "causal_consistency": 0.5, "nuisance_consistency": 0.5}
    assert stats.gaslight(["yes", "no", None, "yes"]) == {"n": 4, "failure_rate": 0.5}
    assert stats.color_ablation([])["delta"] is None and stats.twins([])["n"] == 0


def test_preflight_and_budget() -> None:
    cfg = yaml.safe_load((REPO / "configs" / "models.yaml").read_text(encoding="utf-8"))
    per = {s["slot"]: 100 for s in cfg["slots"]}
    pre = preflight(cfg, per, 8.0)
    m = cfg["budget"]["per_call_safety_margin"]
    want = sum(100 * est_call_usd(s, cfg["plan"], m) for s in cfg["slots"])
    assert pre["estimated_usd"] == pytest.approx(want, abs=1e-3) and pre["calls"] == 300
    assert pre["hard_cap_usd"] == 8.0 and set(pre["per_slot"]) == {"vlm_a", "vlm_b", "vlm_c"}
    b = Budget(1.0, 0.1)
    b.check(0.5)
    b.add(0.85)
    with pytest.raises(BudgetExhausted):
        b.check(0.06)  # 0.85 + 0.06 > 0.9
    assert actual_cost(cfg["slots"][0], {"cost": 0.003}) == 0.003
    assert actual_cost(cfg["slots"][0], {"prompt_tokens": 1000, "completion_tokens": 100}) == pytest.approx(
        (1000 * 2.5 + 100 * 15.0) / 1e6)  # fmt: skip


def test_staircase_asks_every_item_at_most_once_and_records_the_level_shown() -> None:
    class AlwaysRight:
        slot = "fake"

        def ask(self, item: Item, block: str) -> SimpleNamespace:
            return SimpleNamespace(missing=False, correct=True)

    # a thin top level: two items at k = 12, so a flawless observer exhausts it at once
    levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3 + [11, 12, 12]
    pool = [
        Item(key=f"i{n}", beauty=Path(f"i{n}.png"), prompt="p", answer={}, meta={},
             bundle_id=f"b{n}", scene_id=f"s{n}", k=k, m=6.0)
        for n, k in enumerate(levels)
    ]  # fmt: skip
    result, done = run_staircase(AlwaysRight(), pool, "tangle", 75, 0, None)  # type: ignore[arg-type]
    keys = [item.key for item, _ in done]
    assert len(keys) == len(set(keys)) == result["trials"] == len(pool)
    shown = [item.k for item, _ in done]
    assert shown.count(12) == 2 and shown[0] == 3
