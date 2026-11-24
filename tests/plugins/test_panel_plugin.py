"""Task 2.5: panel plugin with gaslighting (SPEC 8.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from plugin_helpers import jobs_for, run_all

from knotwork.knotcore import engine
from knotwork.knotcore.contracts import Family, PremiseValue, TaskPlugin
from knotwork.knotcore.io import load_bundle
from knotwork.knotcore.seeds import SeedTree
from knotwork.pipeline.io import write_claims
from knotwork.plugins.panel import PLUGIN
from knotwork.plugins.panel.plugin import socket_values
from knotwork.plugins.panel.programs import family_counts, round_half_up

FAMS = [f.value for f in PLUGIN.families]


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[dict[str, Any]]]:
    return run_all(jobs_for("panel", FAMS, 30, tmp_path_factory.mktemp("panel_runs"), tag="clean"))


def test_protocol(profile: dict[str, Any]) -> None:
    assert isinstance(PLUGIN, TaskPlugin)


@pytest.mark.parametrize("n", [20, 33, 40, 600])
def test_schedule_exact_counts(profile: dict[str, Any], n: int) -> None:
    counts = family_counts(n, {k: float(v) for k, v in profile["panel"]["families"].items()})
    assert sum(counts.values()) == n
    assert counts["T_GASLIGHT"] == counts["P5_PREMISE_CONTROL"] == round_half_up(0.15 * n)


def test_programs_and_premise_steps(profile: dict[str, Any], tmp_path: Path) -> None:
    for i, fam in enumerate(PLUGIN.families):
        at = SeedTree(21).child("q", i)
        scene = PLUGIN.sample_scene(at.child("scene"), dict(profile, _family=fam.value))
        q = PLUGIN.questions(scene, at.child("question"), fam)
        assert q.family is fam
        d1 = PLUGIN.symbolic_answer(scene, q)
        if fam in (Family.P5_PREMISE_CONTROL, Family.T_GASLIGHT):
            assert [s.op.value for s in q.steps] == ["TRACE", "GAUGE_OF", "READ", "ZONE"]
            assert q.premise is not None and q.premise.holds is (fam is Family.P5_PREMISE_CONTROL)
            assert isinstance(d1.value, PremiseValue)
            d = tmp_path / f"b{i}"
            PLUGIN.render(scene, d)
            write_claims(d, PLUGIN.claims(scene))
            d3, _ = engine.execute(load_bundle(d), q, profile["kernel"])
            assert isinstance(d3.value, PremiseValue)
            assert d3.value.premise_holds is d1.value.premise_holds
            assert abs(float(d3.value.value) - float(d1.value.value)) <= d3.tolerance
        if fam is Family.P3_ARGMAX_PORT:
            vals = sorted((v for _, v in socket_values(scene.payload)), reverse=True)
            assert vals[0] - vals[1] >= 3 * scene.payload["grade_tol"]


def test_90pct_certify(runs: dict[str, list[dict[str, Any]]]) -> None:
    res = runs["clean"]
    ok = sum(r["verdict"] == "CERTIFIED" for r in res)
    assert ok >= 27, [(r["family"], r["codes"]) for r in res if r["verdict"] != "CERTIFIED"]
    assert {r["family"] for r in res} == set(FAMS)


def test_causal_twin_flips_premise_family(profile: dict[str, Any]) -> None:
    for i, fam in enumerate(
        [Family.P5_PREMISE_CONTROL, Family.T_GASLIGHT, Family.P3_ARGMAX_PORT, Family.P4_COUNT_ZONE]
    ):
        at = SeedTree(22).child("tw", i)
        scene = PLUGIN.sample_scene(at.child("scene"), dict(profile, _family=fam.value))
        q = PLUGIN.questions(scene, at.child("question"), fam)
        twin, _ = PLUGIN.causal_twin(scene, q, at.child("twin_causal", 0))
        tq = PLUGIN.twin_question(twin, q)
        assert (
            PLUGIN.symbolic_answer(twin, tq).canonical != PLUGIN.symbolic_answer(scene, q).canonical
        )
        if fam in (Family.P5_PREMISE_CONTROL, Family.T_GASLIGHT):
            assert tq.family is not q.family
        nuis = PLUGIN.nuisance_twin(scene, q, at.child("twin_nuisance", 0))
        assert (
            PLUGIN.symbolic_answer(nuis, PLUGIN.twin_question(nuis, q)).canonical
            == PLUGIN.symbolic_answer(scene, q).canonical
        )
