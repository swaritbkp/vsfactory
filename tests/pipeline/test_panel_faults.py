"""Panel fault injections M11-M14 on one fixed scene, certified with the deterministic template OCR
so the check stays fast (`knot mutate` runs the full matrix with RapidOCR)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from knotwork.knotcore.contracts import FAULT_EXPECTED, FAULT_PLUGIN, Family, FaultCode
from knotwork.knotcore.seeds import SeedTree
from knotwork.pipeline.certify import run_candidate
from knotwork.plugins.registry import fault_effective, get_plugin

REPO = Path(__file__).resolve().parents[2]
PROFILE: dict[str, Any] = yaml.safe_load(
    (REPO / "configs" / "default_profile.yaml").read_text(encoding="utf-8")
)
PANEL_FAULTS = (FaultCode.M11, FaultCode.M12, FaultCode.M13, FaultCode.M14)


def test_panel_faults_are_registered_and_effective() -> None:
    plugin = get_plugin("panel")
    at = SeedTree(3).child("panel", "faults")
    for i in range(20):
        scene = plugin.sample_scene(at.child(i), dict(PROFILE, _family="P1_READ_TRACE"))
        if all(fault_effective("panel", scene, f) for f in PANEL_FAULTS):
            break
    else:
        pytest.fail("no scene admits every panel fault")
    assert all(FAULT_PLUGIN[f] == "panel" and f in FAULT_EXPECTED for f in PANEL_FAULTS)
    assert not fault_effective("panel", scene, FaultCode.M8)


@pytest.mark.parametrize("fault", [FaultCode.M13, FaultCode.M14])
def test_panel_fault_is_caught_with_its_code(tmp_path: Path, fault: FaultCode) -> None:
    profile = copy.deepcopy(PROFILE)
    profile["kernel"]["ocr"]["engine"] = "template"
    plugin = get_plugin("panel")
    at = SeedTree(3).child("panel", "faults", 0)
    scene = plugin.sample_scene(at, dict(profile, _family="P1_READ_TRACE"))
    question = plugin.questions(scene, at.child("question"), Family.P1_READ_TRACE)
    bad = run_candidate(plugin, scene, question, tmp_path / "faulty", profile, fault)
    assert FAULT_EXPECTED[fault] in bad.certificate.codes, bad.certificate.codes
    assert not bad.certified
