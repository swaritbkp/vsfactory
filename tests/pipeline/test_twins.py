"""Task 2.8: twin builder (SPEC 8.4)."""

from __future__ import annotations

from pathlib import Path

import pytest
from plugin_helpers import PROFILE

from knotwork.knotcore.contracts import Family
from knotwork.knotcore.seeds import SeedTree
from knotwork.pipeline.certify import run_candidate
from knotwork.pipeline.twins import TWIN_RETRIES, build_twin, pixel_diff_frac
from knotwork.plugins.registry import get_plugin


@pytest.mark.parametrize(
    ("plugin", "family"),
    [("tangle", Family.T2_CROSS), ("gauge", Family.G1_READ), ("panel", Family.T_GASLIGHT)],
)
def test_causal_and_nuisance_twins(tmp_path: Path, plugin: str, family: Family) -> None:
    p = get_plugin(plugin)
    at = SeedTree(41).child(plugin, "twin")
    scene = p.sample_scene(at.child("scene"), dict(PROFILE, _family=family.value))
    q = p.questions(scene, at.child("question"), family)
    base = run_candidate(p, scene, q, tmp_path / "base", PROFILE)
    assert base.certified, base.certificate.codes
    causal = build_twin("causal", p, base, at, tmp_path, PROFILE, family.value)
    nuis = build_twin("nuisance", p, base, at, tmp_path, PROFILE, family.value)
    assert causal.candidate is not None and nuis.candidate is not None
    assert causal.n_candidates <= TWIN_RETRIES and nuis.n_candidates <= TWIN_RETRIES
    assert causal.candidate.answer.canonical != base.answer.canonical
    assert causal.diff is not None and causal.diff < 0.05
    assert causal.diff == pixel_diff_frac(base.dir, causal.candidate.dir)
    assert nuis.candidate.answer.canonical == base.answer.canonical
    assert nuis.diff is not None and nuis.diff >= 0.02
    if family is Family.T_GASLIGHT:
        # moving the needle into the claimed zone makes the premise true: twin family flips
        assert causal.candidate.question.family is Family.P5_PREMISE_CONTROL
        assert (
            causal.candidate.question.premise is not None
            and causal.candidate.question.premise.holds
        )
