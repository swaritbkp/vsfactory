"""Task 1.8: COUNT, CMP, ARGMAX (SPEC 7.8)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fixtures import CFG, Dial, canvas, draw_dial, write_claims

from knotwork.knotcore.contracts import ErrorCode
from knotwork.knotcore.io import KernelFault, load_bundle
from knotwork.knotcore.logic import argmax, cmp, count


def test_cmp_is_tolerance_aware() -> None:
    assert cmp([47.0, 0.3], [47.5, 0.3])[0] == "equal"
    assert cmp([48.0, 0.3], [47.0, 0.3])[0] == "greater"
    assert cmp([46.0, 0.3], [47.0, 0.3])[0] == "less"


def test_argmax_and_tie() -> None:
    assert argmax([["A", 30.0, 0.3], ["B", 62.0, 0.3], ["C", 50.0, 0.3]])[0] == "B"
    with pytest.raises(KernelFault) as e:
        argmax([["A", 61.6, 0.3], ["B", 62.0, 0.3]])
    assert e.value.code is ErrorCode.E_AMBIGUOUS_ARGMAX


@pytest.fixture(scope="module")
def two_dials(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """g1 reads 71 (red); g2 reads 74 (red) but its needle is hidden by an occluder."""
    out = tmp_path_factory.mktemp("dials")
    c = canvas()
    g1 = draw_dial(c, Dial(value=71.0, center=(280.0, 512.0), radius=200.0, name="g1"))
    g2 = draw_dial(c, Dial(value=74.0, center=(760.0, 512.0), radius=200.0, name="g2"))
    occ = c.new_object("fx/occluder:cover", "occluder", (200, 200, 200))
    c.disk(occ, (760.0, 512.0), 190)
    c.finish(out, "fx", gauges=[g1, g2])
    write_claims(out, "fx", [])
    return out


def test_count_visible_and_zone(two_dials: Path) -> None:
    b = load_bundle(two_dials)
    assert count(b, "needle", "any", CFG)[0] == 1  # g2's needle has < min_visible_px visible
    assert count(b, "needle", "zone:red", CFG)[0] == 1
    assert count(b, "needle", "zone:green", CFG)[0] == 0
    assert count(b, "dial", "any", CFG)[0] == 2
