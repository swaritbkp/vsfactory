"""Task 2.4: gauge plugin (SPEC 8.2)."""

from __future__ import annotations

from typing import Any

import pytest
from plugin_helpers import jobs_for, run_all

from knotwork.knotcore.contracts import FAULT_EXPECTED, Family, FaultCode, TaskPlugin
from knotwork.knotcore.seeds import SeedTree
from knotwork.plugins.gauge import PLUGIN, fault_effective
from knotwork.plugins.gauge.scale import angle, zone_of

FAMS = [f.value for f in PLUGIN.families]


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[dict[str, Any]]]:
    work = tmp_path_factory.mktemp("gauge_runs")
    jobs = jobs_for("gauge", FAMS, 40, work, tag="clean")
    for f in (FaultCode.M8, FaultCode.M9):
        jobs += jobs_for("gauge", FAMS, 6, work, fault=f.value, tag=f.value)
    return run_all(jobs)


def test_protocol_scales_and_angle_formula(profile: dict[str, Any]) -> None:
    assert isinstance(PLUGIN, TaskPlugin)
    scales = {
        PLUGIN.sample_scene(SeedTree(3).child("g", i), profile).payload["dial"]["scale"]
        for i in range(30)
    }
    assert scales == {"linear", "offset"}
    assert angle(0, 0, 80) == 225 and angle(80, 0, 80) == -45 and angle(40, 0, 80) == 90


def test_needle_values_avoid_zone_boundaries(profile: dict[str, Any]) -> None:
    zones_seen = set()
    for i in range(80):
        fam = "G2_ZONE" if i % 2 else "G1_READ"
        d = PLUGIN.sample_scene(SeedTree(4).child("v", i), dict(profile, _family=fam)).payload[
            "dial"
        ]
        tol = 0.02 * (d["vmax"] - d["vmin"])
        v = d["value"]
        assert float(v).is_integer()
        for b in (d["vmin"], *d["zones"], d["vmax"]):
            assert abs(v - b) >= 3 * tol
        assert (d["zones"][0] - d["vmin"]) % 5 == 0 and (d["zones"][1] - d["vmin"]) % 5 == 0
        zones_seen.add(zone_of(v, tuple(d["zones"])))
    assert zones_seen == {"green", "amber", "red"}


@pytest.mark.tier2
def test_labels_ocr_and_95pct_certify(runs: dict[str, list[dict[str, Any]]]) -> None:
    res = runs["clean"]
    assert not any("E_LABEL_MISMATCH" in r["codes"] or "E_OCR_FAIL" in r["codes"] for r in res)
    assert all(r["ocr_engine"] == "rapidocr" for r in res)
    assert sum(r["verdict"] == "CERTIFIED" for r in res) >= 38, [r["codes"] for r in res]


@pytest.mark.tier2
@pytest.mark.parametrize("fault", [FaultCode.M8, FaultCode.M9])
def test_fault_hooks(runs: dict[str, list[dict[str, Any]]], fault: FaultCode) -> None:
    hit = [r for r in runs[fault.value] if r["effective"]]
    assert hit, f"{fault} never effective"
    for r in hit:
        assert FAULT_EXPECTED[fault].value in r["codes"], r


def test_twins(profile: dict[str, Any]) -> None:
    for i, fam in enumerate([Family.G1_READ, Family.G2_ZONE] * 3):
        at = SeedTree(5).child("tw", i)
        scene = PLUGIN.sample_scene(at.child("scene"), dict(profile, _family=fam.value))
        q = PLUGIN.questions(scene, at.child("question"), fam)
        base = PLUGIN.symbolic_answer(scene, q)
        causal, _ = PLUGIN.causal_twin(scene, q, at.child("twin_causal", 0))
        new = PLUGIN.symbolic_answer(causal, q)
        assert new.canonical != base.canonical
        if fam is Family.G1_READ:
            assert abs(float(new.value) - float(base.value)) >= 5 * scene.payload["grade_tol"]  # type: ignore[arg-type]
        nuis = PLUGIN.nuisance_twin(scene, q, at.child("twin_nuisance", 0))
        assert PLUGIN.symbolic_answer(nuis, q).canonical == base.canonical
        assert nuis.payload["render"]["background_rgb"] != scene.payload["render"]["background_rgb"]
        assert isinstance(fault_effective(scene, FaultCode.M9), bool)
