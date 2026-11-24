"""Task 2.1: braid algebra (SPEC 8.1)."""

from __future__ import annotations

import numpy as np
from scipy.stats import chisquare

from knotwork.plugins.tangle import braid


def test_sigma1_sigma2_hand_example() -> None:
    p, cs = braid.compose(3, [(0, 1), (1, 1)])
    # (0,+1): a=0 b=1 over=0 -> P=[1,0,2]; (1,+1): a=0 b=2 over=0 -> P=[1,2,0]
    assert p == [1, 2, 0]
    assert [(c.a, c.b, c.over) for c in cs] == [(0, 1, 0), (0, 2, 0)]
    assert braid.end_position(3, [(0, 1), (1, 1)]) == {1: 0, 2: 1, 0: 2}
    _, neg = braid.compose(3, [(0, -1)])
    assert neg[0].over == 1 and neg[0].under == 0  # over = a iff eps = +1


def test_crossing_counts_and_pairs() -> None:
    word = [(0, 1), (1, -1), None, (0, 1)]
    assert braid.crossing_counts(3, word) == [2, 2, 2]
    assert len(braid.pair_crossings(3, word, 0, 1)) == 1


def test_reduced_word_realises_permutation() -> None:
    rng = np.random.default_rng(0)
    for n in range(3, 11):
        for _ in range(30):
            perm = [int(x) for x in rng.permutation(n)]
            assert braid.compose(n, braid.reduced_word(perm, rng))[0] == perm


def test_sampled_permutations_uniform() -> None:
    rng = np.random.default_rng(1)
    counts: dict[tuple[int, ...], int] = {}
    for _ in range(6000):
        perm, word = braid.sample_word(4, rng, (4, 18), (0, 3), (2, 4))
        assert braid.compose(4, word)[0] == perm
        counts[tuple(perm)] = counts.get(tuple(perm), 0) + 1
    assert len(counts) == 24
    assert chisquare(list(counts.values())).pvalue > 0.01


def test_pure_and_r2_pairs_preserve_permutation() -> None:
    rng = np.random.default_rng(2)
    for _ in range(200):
        n = int(rng.integers(3, 11))
        perm = [int(x) for x in rng.permutation(n)]
        word: list[braid.Letter] = list(braid.reduced_word(perm, rng))
        i, k = int(rng.integers(n - 1)), int(rng.integers(len(word) + 1))
        word[k:k] = [(i, 1), (i, 1)] if rng.random() < 0.5 else [(i, 1), (i, -1)]
        assert braid.compose(n, word)[0] == perm


def test_identity_slots_counted_and_columns() -> None:
    rng = np.random.default_rng(3)
    for _ in range(100):
        _, word = braid.sample_word(5, rng, (4, 18), (0, 3), (2, 4))
        slots = sum(x is None for x in word)
        assert 2 <= slots <= 4
        cols = braid.columns(5, word)
        assert sum(1 for c in cols if len(c) == 1 and word[c[0]] is None) == 2 * slots
        letters = sorted(k for c in cols for k in c if word[k] is not None)
        assert letters == [k for k, x in enumerate(word) if x is not None]
