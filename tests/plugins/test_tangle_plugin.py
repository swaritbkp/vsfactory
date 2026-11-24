"""Task 2.3: tangle plugin end to end (SPEC 6.1, 8.1, 10)."""

from __future__ import annotations

from typing import Any

import pytest
from plugin_helpers import jobs_for, run_all

from knotwork.knotcore.contracts import FAULT_EXPECTED, FAULT_PLUGIN, Family, FaultCode, TaskPlugin
from knotwork.knotcore.seeds import SeedTree
from knotwork.plugins.tangle import PLUGIN, braid, fault_effective
from knotwork.plugins.tangle.plugin import PROMPTS, word_of

FAMS = [f.value for f in PLUGIN.families]
TANGLE_FAULTS = [f for f in FaultCode if FAULT_PLUGIN[f] == "tangle"]


def test_protocol_styles_and_colour_regimes(profile: dict[str, Any]) -> None:
    assert isinstance(PLUGIN, TaskPlugin)
    seen: dict[str, set[str]] = {"style": set(), "color_regime": set()}
    for i in range(60):
        pay = PLUGIN.sample_scene(SeedTree(5).child("s", i), profile).payload
        seen["style"].add(pay["style"])
        seen["color_regime"].add(pay["color_regime"])
    assert seen == {
        "style": {"schematic", "harness"},
        "color_regime": {"unique", "partial", "mono"},
    }
    test_split = PLUGIN.sample_scene(SeedTree(5).child("t"), dict(profile, _split="test")).payload
    assert test_split["color_regime"] == "mono"


def test_exact_prompts_and_claims(profile: dict[str, Any]) -> None:
    for i, fam in enumerate(PLUGIN.families):
        at = SeedTree(6).child("p", i)
        scene = PLUGIN.sample_scene(at.child("scene"), dict(profile, _family=fam.value))
        q = PLUGIN.questions(scene, at.child("question"), fam)
        head = PROMPTS[fam].split("{")[0]
        assert q.text.startswith(head) and "ANSWER:" in q.text
        pay = scene.payload
        n, word = pay["n"], word_of(pay)
        claims = PLUGIN.claims(scene).claims
        n_cross = len(braid.compose(n, word)[1])
        assert sum(c.op.value == "TRACE" for c in claims) == n
        assert sum(c.op.value == "CROSS" for c in claims) == n
        assert sum(c.op.value == "ONTOP" for c in claims) == n_cross


def test_t2_targets_cover_range(profile: dict[str, Any]) -> None:
    vals = set()
    for i in range(40):
        at = SeedTree(7).child("t2", i)
        scene = PLUGIN.sample_scene(at.child("scene"), dict(profile, _family="T2_CROSS"))
        q = PLUGIN.questions(scene, at.child("question"), Family.T2_CROSS)
        vals.add(int(PLUGIN.symbolic_answer(scene, q).value))  # type: ignore[arg-type]
    assert vals <= set(range(7)) and len(vals) >= 5


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[dict[str, Any]]]:
    """40 clean scenes + 4 fault injections per tangle fault, certified in one process pool."""
    work = tmp_path_factory.mktemp("tangle_runs")
    jobs = jobs_for("tangle", FAMS, 40, work, tag="clean")
    for f in TANGLE_FAULTS:
        jobs += jobs_for("tangle", FAMS, 4, work, fault=f.value, tag=f.value)
    return run_all(jobs)


def test_95pct_of_40_clean_scenes_certify(runs: dict[str, list[dict[str, Any]]]) -> None:
    res = runs["clean"]
    ok = sum(r["verdict"] == "CERTIFIED" for r in res)
    assert ok >= 38, [r["codes"] for r in res if r["verdict"] != "CERTIFIED"]


@pytest.mark.parametrize("fault", TANGLE_FAULTS)
def test_fault_hooks_raise_expected_code(
    runs: dict[str, list[dict[str, Any]]], fault: FaultCode
) -> None:
    hit = [r for r in runs[fault.value] if r["effective"]]
    assert hit, f"{fault} never effective"
    for r in hit:
        assert FAULT_EXPECTED[fault].value in r["codes"], r


def test_fault_effective_defined(profile: dict[str, Any]) -> None:
    scene = PLUGIN.sample_scene(SeedTree(8).child("fe"), profile)
    for f in FaultCode:
        assert isinstance(fault_effective(scene, f), bool)
    assert not fault_effective(scene, FaultCode.M8)


def test_twins_flip_and_preserve(profile: dict[str, Any]) -> None:
    for i, fam in enumerate(PLUGIN.families):
        at = SeedTree(9).child("tw", i)
        scene = PLUGIN.sample_scene(at.child("scene"), dict(profile, _family=fam.value))
        q = PLUGIN.questions(scene, at.child("question"), fam)
        base = PLUGIN.symbolic_answer(scene, q).canonical
        causal, edit = PLUGIN.causal_twin(scene, q, at.child("twin_causal", 0))
        assert PLUGIN.symbolic_answer(causal, q).canonical != base and edit
        nuis = PLUGIN.nuisance_twin(scene, q, at.child("twin_nuisance", 0))
        assert PLUGIN.symbolic_answer(nuis, q).canonical == base
        assert nuis.payload["style"] != scene.payload["style"]
