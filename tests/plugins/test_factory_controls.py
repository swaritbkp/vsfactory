"""Difficulty dials, typed payloads, OOD holdouts, lineage and telemetry (sampling only, no OCR)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from knotwork.freshness import lineage
from knotwork.knotcore.contracts import DifficultySpec, Family
from knotwork.knotcore.seeds import SeedTree
from knotwork.pipeline import ood, telemetry
from knotwork.plugins.payloads import validate_payload
from knotwork.plugins.registry import NAMES, get_plugin

REPO = Path(__file__).resolve().parents[2]
PROFILE: dict[str, Any] = yaml.safe_load(
    (REPO / "configs" / "default_profile.yaml").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("name", NAMES)
def test_dials_are_orthogonal_and_payloads_are_typed(name: str) -> None:
    plugin = get_plugin(name)
    family = plugin.families[0]
    at = SeedTree(9).child(name, "dials")
    prof = dict(PROFILE, _family=family.value)
    spec = DifficultySpec(reasoning_load=0.0, visual_stress=0.8)
    scene = plugin.sample_scene(at, prof, spec)
    question = plugin.questions(scene, at.child("q"), Family(family))
    assert question.difficulty_spec == spec
    assert scene.payload["surface"]["noise"] > 0 and scene.payload["surface"]["dpi_scale"] < 1
    if name == "tangle":
        assert scene.payload["n"] == PROFILE["tangle"]["strands"][0]
        assert scene.payload["color_regime"] == "mono"
    if name == "panel":
        assert scene.payload["k"] == PROFILE["panel"]["gauges"][0]
    plain = plugin.sample_scene(at, prof)
    assert "difficulty_spec" not in plain.payload
    broken = dict(scene.payload, surprise=1)
    with pytest.raises(ValidationError):
        validate_payload(name, broken)


def test_ood_split_and_its_complement_are_disjoint() -> None:
    held = ood.resolve("tangle", "ood", PROFILE, None, ["T3_ONTOP"], ["strands=9:10"])
    rest = ood.resolve("tangle", "dev", PROFILE, None, ["T3_ONTOP"], ["strands=9:10"])
    assert held.families == ["T3_ONTOP"] and held.profile["tangle"]["strands"] == [9, 10]
    assert rest.families == ["T1_TRACE", "T2_CROSS"] and rest.profile["tangle"]["strands"] == [3, 8]
    assert PROFILE["tangle"]["strands"] == [3, 10]
    with pytest.raises(ValueError, match="one end"):
        ood.resolve("tangle", "dev", PROFILE, None, None, ["strands=5:6"])
    with pytest.raises(ValueError, match="needs"):
        ood.resolve("tangle", "ood", PROFILE, None)


def test_lineage_detects_edits_and_telemetry_flags_regressions() -> None:
    task = {
        "bundle_id": "kw-1", "scene_id": "a" * 64, "seed_path": "p", "files": {"beauty.png": "x"},
        "certificate": {"kernel_sha256": "k", "verdict": "CERTIFIED"}, "twins": {"role": "base"},
    }  # fmt: skip
    nodes = lineage.task_nodes(task, {})
    assert [n["kind"] for n in nodes] == ["scene", "render", "certificate"]
    assert lineage.verify(nodes) == []
    edited = [nodes[0], dict(nodes[1], digest="y"), nodes[2]]
    assert lineage.verify(edited)

    def run(ok: bool) -> dict[str, Any]:
        tally = telemetry.Telemetry()
        quarantine = [] if ok else [{"codes": ["E_ZORDER"]}]
        tally.add(
            "T1_TRACE", {"ok": ok, "n_candidates": 1, "bundles": [], "quarantine": quarantine}
        )
        return tally.report(1.0)

    diff = telemetry.compare(run(True), run(False))
    assert any("false_quarantine_rate" in line for line in diff["regressions"])
    assert run(False)["rejection_pareto"][0]["code"] == "E_ZORDER"
