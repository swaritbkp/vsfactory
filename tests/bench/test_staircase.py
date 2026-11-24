"""Task 5.1: 1-up-2-down staircase converges to the 70.7 % level (SPEC 12.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from knotwork.bench.staircase import TARGET, Staircase, expected_level, run_simulated


def observer(k: float, centre: float = 6.0, slope: float = 0.8) -> float:
    """Known psychometric curve: 4-option guess rate, logistic in the level."""
    return 0.25 + 0.75 / (1.0 + math.exp((k - centre) / slope))


def test_threshold_within_1_5_levels_in_90pct_of_200_runs() -> None:
    k_star = brentq(lambda x: observer(x) - TARGET, 1, 12)
    hits = 0
    for seed in range(200):
        st = run_simulated(observer, np.random.default_rng(seed), 1, 12)
        hits += abs(st.threshold() - k_star) <= 1.5
    assert hits >= 180, hits


def test_rules() -> None:
    st = Staircase(1, 12)
    assert st.level == 3
    st.update(True)
    assert st.level == 3
    st.update(True)
    assert st.level == 4  # two consecutive correct -> harder
    st.update(False)
    assert st.level == 3 and st.reversals == [4]  # one wrong -> easier (a reversal)
    st.update(True)
    st.update(True)
    assert st.level == 4 and st.reversals == [4, 3]
    assert st.threshold() == pytest.approx(3.5)


def test_stops_at_10_reversals_or_max_trials() -> None:
    st = Staircase(1, 12, max_trials=1000)
    while not st.done:
        st.update(len(st.history) % 3 != 2)  # correct, correct, wrong, ...
    assert len(st.reversals) == 10
    st2 = Staircase(1, 12, max_trials=12)
    while not st2.done:
        st2.update(True)
    assert len(st2.history) == 12 and st2.level == 9
    assert st2.threshold() == pytest.approx(np.mean([lv for lv, _ in st2.history]))


def test_bounds_and_expected_level() -> None:
    st = Staircase(1, 4)
    for _ in range(20):
        st.update(True)
    assert st.level == 4
    for _ in range(10):
        st.update(False)
    assert st.level == 1
    p, m = 0.08, 6.0
    k = expected_level(p, lambda _: m, 1, 12)
    assert (1 - p) ** k + (1 - (1 - p) ** k) / m == pytest.approx(TARGET, abs=1e-9)
    assert expected_level(0.04, lambda _: 3.0, 1, 5) == 5.0  # beyond reach -> clipped
    assert expected_level(0.9, lambda _: 3.0, 1, 5) == 1.0
