"""Adaptive 1-up-2-down staircase with a maximum-likelihood threshold (SPEC 12.2, D-39).

Start at level 3; two consecutive correct answers -> harder, one wrong -> easier. The step is 2 levels
until the third reversal, then 1. Stop at `MAX_REVERSALS` reversals or ``max_trials`` (75). The
threshold ``k*`` (P(correct) = sqrt(0.5)) comes from a logistic psychometric function fitted by
maximum likelihood to every trial; the mean of the last reversals is kept as a cross-check.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, minimize

TARGET = math.sqrt(0.5)
MAX_REVERSALS = 20
LAST_REVERSALS = 6
MAX_TRIALS = 75
COARSE_STEP, FINE_STEP = 2, 1
COARSE_REVERSALS = 3
#: starting points (threshold fraction of the level range, slope) of the likelihood search
STARTS = ((0.2, 0.5), (0.5, 1.0), (0.8, 2.0))


@dataclass
class Staircase:
    lo: int
    hi: int
    start: int = 3
    max_trials: int = MAX_TRIALS
    level: int = field(init=False)
    streak: int = field(default=0, init=False)
    direction: int = field(default=0, init=False)
    reversals: list[int] = field(default_factory=list, init=False)
    history: list[tuple[int, bool]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.hi < self.lo:
            raise ValueError("empty level range")
        self.level = min(max(self.start, self.lo), self.hi)

    @property
    def done(self) -> bool:
        return len(self.reversals) >= MAX_REVERSALS or len(self.history) >= self.max_trials

    def update(self, correct: bool, shown: int | None = None) -> None:
        """Record the answer and move. `shown` is the level of the item actually asked when no
        unused item was left at the current level; the history keeps what was shown."""
        self.history.append((self.level if shown is None else shown, bool(correct)))
        step = 0
        if correct:
            self.streak += 1
            if self.streak == 2:
                self.streak, step = 0, 1
        else:
            self.streak, step = 0, -1
        if step == 0:
            return
        if self.direction != 0 and step != self.direction:
            self.reversals.append(self.level)
        self.direction = step
        size = COARSE_STEP if len(self.reversals) < COARSE_REVERSALS else FINE_STEP
        self.level = min(max(self.level + step * size, self.lo), self.hi)

    def threshold(self) -> float:
        if self.reversals:
            return float(np.mean(self.reversals[-LAST_REVERSALS:]))
        return float(np.mean([lv for lv, _ in self.history])) if self.history else float(self.level)


def mle_threshold(
    history: list[tuple[int, bool]],
    lo: int,
    hi: int,
    guess: Callable[[float], float] | None = None,
) -> float:
    """``k*`` of the logistic curve P(k) = g(k) + (1 - g(k)) / (1 + exp((k - mu) / s)) fitted to all
    trials by maximum likelihood, clipped to [lo, hi]; ``g`` is the guess rate per level. A weak
    quadratic prior on the slope keeps separable data finite."""
    if not history:
        return float(lo)
    levels = np.array([k for k, _ in history], dtype=np.float64)
    correct = np.array([c for _, c in history], dtype=np.float64)
    floor = np.array([guess(k) if guess else 0.0 for k in levels], dtype=np.float64)

    def curve(k: np.ndarray, g: np.ndarray, mu: float, s: float) -> np.ndarray:
        return g + (1.0 - g) / (1.0 + np.exp(np.clip((k - mu) / s, -30.0, 30.0)))

    def loss(x: np.ndarray) -> float:
        mu, log_s = float(x[0]), float(np.clip(x[1], -3.0, 3.0))
        prob = np.clip(curve(levels, floor, mu, math.exp(log_s)), 1e-6, 1 - 1e-6)
        likelihood = correct * np.log(prob) + (1 - correct) * np.log(1 - prob)
        return float(-likelihood.sum() + 0.05 * log_s**2)

    bounds = [(lo - (hi - lo) - 1.0, hi + (hi - lo) + 1.0), (-3.0, 3.0)]
    fits = [
        minimize(
            loss, np.array([lo + f * (hi - lo), math.log(s)]), method="L-BFGS-B", bounds=bounds
        )
        for f, s in STARTS
    ]
    best = min(fits, key=lambda r: float(r.fun))
    mu, s = float(best.x[0]), math.exp(float(best.x[1]))

    def f(k: float) -> float:
        g = guess(k) if guess else 0.0
        return float(curve(np.array(k), np.array(g), mu, s) - TARGET)

    if f(lo) <= 0:
        return float(lo)
    if f(hi) >= 0:
        return float(hi)
    return float(brentq(f, lo, hi))


def expected_level(p: float, m_bar: Callable[[float], float], lo: int, hi: int) -> float:
    """Real k* solving (1-p)^k + (1 - (1-p)^k)/m(k) = sqrt(0.5), clipped to [lo, hi] (the range the
    staircase can reach; D-33)."""

    def f(k: float) -> float:
        s = (1.0 - p) ** k
        return float(s + (1.0 - s) / m_bar(k) - TARGET)

    if f(lo) <= 0:
        return float(lo)
    if f(hi) >= 0:
        return float(hi)
    return float(brentq(f, lo, hi))


def run_simulated(
    p_correct: Callable[[int], float],
    rng: np.random.Generator,
    lo: int,
    hi: int,
    max_trials: int = MAX_TRIALS,
) -> Staircase:
    """Staircase driven by an observer with a known psychometric curve (tests / calibration)."""
    st = Staircase(lo, hi, max_trials=max_trials)
    while not st.done:
        st.update(bool(rng.random() < p_correct(st.level)))
    return st


__all__ = ["MAX_TRIALS", "TARGET", "Staircase", "expected_level", "mle_threshold", "run_simulated"]
