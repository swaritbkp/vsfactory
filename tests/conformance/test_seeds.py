"""Hierarchical seed determinism and SHA-256 content-addressing stability."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from knotwork.knotcore import seeds as S

ROOT = 0x1234_5678_9ABC_DEF0_0FED_CBA9_8765_4321


def test_known_answer_vector_matches_pinned_constants() -> None:
    assert S.selftest_vector() == S.SELFTEST_EXPECTED


def test_known_answer_vector_identical_in_fresh_process() -> None:
    code = "import json;from knotwork.knotcore.seeds import selftest_vector as v;print(json.dumps(v(),sort_keys=True))"
    a = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    b = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert a.stdout == b.stdout and a.stdout.strip()


def test_same_path_same_stream() -> None:
    a = S.SeedTree(ROOT).child("tangle", 3, "jitter").rng().integers(0, 1 << 30, 64)
    b = S.SeedTree(ROOT).child("tangle").child(3).child("jitter").rng().integers(0, 1 << 30, 64)
    assert np.array_equal(a, b)


def test_siblings_and_order_differ() -> None:
    t = S.SeedTree(ROOT)
    draws = {
        p: tuple(t.child(*p).rng().integers(0, 1 << 30, 8))
        for p in [("a", "b"), ("b", "a"), ("a", 1), ("a", "1"), ("a",)]
    }
    assert len(set(draws.values())) == len(draws)


def test_root_changes_everything() -> None:
    a = S.SeedTree(ROOT).child("x").int_seed()
    b = S.SeedTree(ROOT + 1).child("x").int_seed()
    assert a != b


@pytest.mark.parametrize("bad", [-1, 2**128, True, 1.5, "7"])
def test_root_validation(bad: object) -> None:
    with pytest.raises(ValueError):
        S.SeedTree(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [1.0, None, True, b"x"])
def test_path_component_validation(bad: object) -> None:
    with pytest.raises(TypeError):
        S.SeedTree(ROOT).child(bad)  # type: ignore[arg-type]


def test_child_requires_parts() -> None:
    with pytest.raises(ValueError):
        S.SeedTree(ROOT).child()


def test_int_seed_widths() -> None:
    t = S.SeedTree(ROOT).child("w")
    assert t.int_seed(32) < 2**32 <= 2**64
    assert t.int_seed(64) < 2**64
    assert t.int_seed(128) < 2**128
    assert t.int_seed(128) >> 64 == t.int_seed(64)


def test_tree_equality_and_hash() -> None:
    assert S.SeedTree(ROOT).child("a") == S.SeedTree(ROOT, ("a",))
    assert len({S.SeedTree(ROOT).child("a"), S.SeedTree(ROOT, ("a",))}) == 1


def test_canonical_json_is_order_independent_and_strict() -> None:
    assert S.canonical_json({"b": 1, "a": [1, 2]}) == S.canonical_json({"a": [1, 2], "b": 1})
    assert S.canonical_json({"k": "é"}) == '{"k":"é"}'.encode()
    with pytest.raises(ValueError):
        S.canonical_json({"x": float("nan")})


def test_array_digest_sensitivity() -> None:
    a = np.arange(12, dtype=np.uint16).reshape(3, 4)
    assert S.array_digest(a) == S.array_digest(a.copy())
    assert S.array_digest(a) == S.array_digest(np.asfortranarray(a))  # layout-independent
    assert S.array_digest(a) == S.array_digest(a.astype(">u2"))  # endianness-normalised
    assert S.array_digest(a) != S.array_digest(a.astype(np.uint32))  # dtype matters
    assert S.array_digest(a) != S.array_digest(a.reshape(4, 3))  # shape matters
    b = a.copy()
    b[0, 0] = 1
    assert S.array_digest(a) != S.array_digest(b)


def test_bundle_id_format_and_stability() -> None:
    files = {"beauty": "a" * 64, "id_buffer": "b" * 64}
    bid = S.bundle_id(files, {"q": 1}, {"a": 2})
    assert bid.startswith("kw-") and len(bid) == 23
    assert bid == S.bundle_id(dict(reversed(list(files.items()))), {"q": 1}, {"a": 2})
    assert bid != S.bundle_id(files, {"q": 1}, {"a": 3})
    with pytest.raises(ValueError):
        S.content_id({})


def test_derive_root_seed() -> None:
    r = S.derive_root_seed(S.offline_randomness(0), "0" * 64)
    assert 0 <= r < 2**128
    assert r != S.derive_root_seed(S.offline_randomness(1), "0" * 64)
    assert r != S.derive_root_seed(S.offline_randomness(0), "1" * 64)
    with pytest.raises(ValueError):
        S.derive_root_seed("abc", "x")
    with pytest.raises(ValueError):
        S.offline_randomness(-1)


def test_canary_format() -> None:
    c = S.canary_for(ROOT)
    assert c.startswith("KNOTWORK-CANARY-") and len(c) == len("KNOTWORK-CANARY-") + 32


@settings(max_examples=60, deadline=None)
@given(
    root=st.integers(min_value=0, max_value=2**128 - 1),
    path=st.lists(st.one_of(st.text(max_size=8), st.integers(0, 10**9)), max_size=4),
)
def test_property_determinism(root: int, path: list[str | int]) -> None:
    t1, t2 = S.SeedTree(root, tuple(path)), S.SeedTree(root, tuple(path))
    assert t1.int_seed() == t2.int_seed()
    assert np.array_equal(t1.rng().random(4), t2.rng().random(4))
    assert t1.digest() == t2.digest()
