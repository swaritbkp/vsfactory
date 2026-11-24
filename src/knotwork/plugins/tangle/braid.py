"""Braid words for tangle scenes (SPEC 8.1). Untrusted plugin code.

A word is a list of letters ``(i, eps)`` (generator sigma_i with sign eps in {+1, -1}) or ``None``
for an identity (ID) slot. Positions 0..n-1 run top to bottom. Composition starts from
``P = [0..n-1]`` (position -> strand); letter ``(i, eps)`` crosses strands ``a = P[i]`` and
``b = P[i+1]`` with **over = a if eps = +1 else b**, then swaps ``P[i], P[i+1]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: (i, eps), or (i, eps, 1) for a letter filling an ID slot (keeps the slot's barrier columns), or
#: None for an empty ID slot.
Letter = tuple[int, ...] | None


@dataclass(frozen=True)
class BraidCrossing:
    letter: int  # index of the letter in the word
    pos: int  # generator index i (positions i, i+1)
    a: int  # strand at position i before the crossing
    b: int  # strand at position i+1 before the crossing
    eps: int

    @property
    def over(self) -> int:
        return self.a if self.eps == 1 else self.b

    @property
    def under(self) -> int:
        return self.b if self.eps == 1 else self.a


def compose(n: int, word: list[Letter]) -> tuple[list[int], list[BraidCrossing]]:
    """Final arrangement ``P`` (position -> strand) and every crossing in word order."""
    p = list(range(n))
    out: list[BraidCrossing] = []
    for k, letter in enumerate(word):
        if letter is None:
            continue
        i, eps = letter[0], letter[1]
        if not 0 <= i < n - 1 or eps not in (1, -1):
            raise ValueError(f"bad letter {letter} for n={n}")
        out.append(BraidCrossing(k, i, p[i], p[i + 1], eps))
        p[i], p[i + 1] = p[i + 1], p[i]
    return p, out


def end_position(n: int, word: list[Letter]) -> dict[int, int]:
    """strand -> final position."""
    p, _ = compose(n, word)
    return {s: j for j, s in enumerate(p)}


def crossing_counts(n: int, word: list[Letter]) -> list[int]:
    """Number of crossings each strand takes part in."""
    counts = [0] * n
    for c in compose(n, word)[1]:
        counts[c.a] += 1
        counts[c.b] += 1
    return counts


def pair_crossings(n: int, word: list[Letter], a: int, b: int) -> list[BraidCrossing]:
    """Crossings between strands a and b in word (= left-to-right path) order."""
    return [c for c in compose(n, word)[1] if {c.a, c.b} == {a, b}]


def reduced_word(perm: list[int], rng: np.random.Generator) -> list[Letter]:
    """Adjacent swaps of an odd-even transposition (parallel bubble) sort realising ``perm``.

    Applying the returned word to the identity arrangement yields ``P == perm``. Each swap removes
    one inversion, so the word is reduced; random signs.
    """
    arr = list(perm)
    swaps: list[int] = []
    n = len(arr)
    for rnd in range(n):
        for i in range(rnd % 2, n - 1, 2):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swaps.append(i)
    # sorting perm -> identity used swaps s1..sk; identity -> perm is the reverse sequence
    return [(i, int(rng.choice([1, -1]))) for i in reversed(swaps)]


def sample_word(
    n: int,
    rng: np.random.Generator,
    length: tuple[int, int],
    r2_pairs: tuple[int, int],
    id_slots: tuple[int, int],
) -> tuple[list[int], list[Letter]]:
    """(target permutation, word) following SPEC 8.1 steps (1)-(5)."""
    perm = [int(x) for x in rng.permutation(n)]
    word = reduced_word(perm, rng)
    target = int(rng.integers(length[0], length[1] + 1))
    while len(word) + 2 <= target:
        i, eps, k = (
            int(rng.integers(0, n - 1)),
            int(rng.choice([1, -1])),
            int(rng.integers(0, len(word) + 1)),
        )
        word[k:k] = [(i, eps), (i, eps)]
    for _ in range(int(rng.integers(r2_pairs[0], r2_pairs[1] + 1))):
        i, eps, k = (
            int(rng.integers(0, n - 1)),
            int(rng.choice([1, -1])),
            int(rng.integers(0, len(word) + 1)),
        )
        word[k:k] = [(i, eps), (i, -eps)]
    for _ in range(int(rng.integers(id_slots[0], id_slots[1] + 1))):
        word.insert(int(rng.integers(0, len(word) + 1)), None)
    return perm, word


def columns(n: int, word: list[Letter]) -> list[list[int]]:
    """Greedy packing of letters into diagram columns (lists of indices into ``word``).

    Letters sharing no position (|i - j| >= 2) commute and may share a column; letters on a shared
    position keep their word order. An ID slot is a barrier occupying two empty columns, each
    holding the slot's word index.
    """
    cols: list[list[int]] = []
    last = [-1] * n  # last column index used by each position
    for k, letter in enumerate(word):
        if letter is None or len(letter) == 3:
            cols.extend([[k], [k]] if letter is None else [[k]])
            last = [len(cols) - 1] * n
            continue
        i = letter[0]
        c = max(last[i], last[i + 1]) + 1
        if c == len(cols):
            cols.append([])
        cols[c].append(k)
        last[i] = last[i + 1] = c
    return cols
