"""Adaptive 1-up-2-down staircase (SPEC 12.2).

Start at level 3; two consecutive correct answers -> level + 1 (harder); one wrong -> level - 1.
Stop at 10 reversals or ``max_trials``. Threshold = mean level of the last 6 reversals (all reversals
if fewer than 6; the mean visited level if there were none). Converges to P(correct) = sqrt(0.5).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

TARGET = math.sqrt(0.5)
MAX_REVERSALS = 10
LAST_REVERSALS = 6


@dataclass
class Staircase:
    lo: int
    hi: int
    start: int = 3
    max_trials: int = 50
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

    def update(self, correct: bool) -> None:
        """Record the answer at the current level and move."""
        self.history.append((self.level, bool(correct)))
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
        self.level = min(max(self.level + step, self.lo), self.hi)

    def threshold(self) -> float:
        if self.reversals:
            return float(np.mean(self.reversals[-LAST_REVERSALS:]))
        return float(np.mean([lv for lv, _ in self.history])) if self.history else float(self.level)


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
    max_trials: int = 50,
) -> Staircase:
    """Staircase driven by an observer with a known psychometric curve (tests / calibration)."""
    st = Staircase(lo, hi, max_trials=max_trials)
    while not st.done:
        st.update(bool(rng.random() < p_correct(st.level)))
    return st


__all__ = ["TARGET", "Staircase", "expected_level", "run_simulated"]
