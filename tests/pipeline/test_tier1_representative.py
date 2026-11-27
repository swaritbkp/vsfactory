"""Fast representative coverage of generation, verification, export, certification and faults.

The exhaustive versions (every family, many scenes, process pools) are tier2 and run nightly; this
file keeps one real path through each stage in tier1, about ten certifications in total.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image

from knotwork.export.manifest import check_manifest
from knotwork.export.run import run_export
from knotwork.freshness import lineage
from knotwork.knotcore.contracts import FAULT_EXPECTED, FaultCode
from knotwork.knotcore.grading import grade_prediction
from knotwork.pipeline import telemetry
from knotwork.pipeline.certify import certify_job
from knotwork.pipeline.generate import run_generate
from knotwork.pipeline.io import read_json, read_jsonl
from knotwork.pipeline.verify import run_verify
from shared_builds import shared_build

REPO = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPO / "configs" / "default_profile.yaml"
PROFILE: dict[str, Any] = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))
ATTEMPTS = 6


@pytest.fixture(scope="module")
def mini(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One gauge item with both twins, rendered once per session."""

    def build(root: Path) -> None:
        rc = run_generate("gauge", 1, root, 5, "offline", PROFILE_PATH, "dev", True, None, 1)
        assert rc == 0

    return shared_build(tmp_path_factory, "tier1_mini", build)


def test_generate_writes_index_telemetry_and_lineage(mini: Path) -> None:
    index = read_jsonl(mini / "index.jsonl")
    assert [r["role"] for r in index] == ["base", "causal", "nuisance"]
    assert telemetry.load(mini)["aggregate"]["certified"] == 1
    assert lineage.check_dataset(mini) == []
    assert index[0]["answer_canonical"] != index[1]["answer_canonical"]


def test_verify_accepts_clean_and_rejects_tampered(mini: Path, tmp_path: Path) -> None:
    row = read_jsonl(mini / "index.jsonl")[0]
    clean = tmp_path / "clean"
    shutil.copytree(mini / row["dir"], clean)
    assert run_verify(clean, tmp_path / "ok.json") == 0
    ids = np.asarray(Image.open(clean / "id_buffer.png"), dtype=np.uint16).copy()
    ids[512, 512] ^= 1
    Image.fromarray(ids).save(clean / "id_buffer.png")
    assert run_verify(clean, tmp_path / "bad.json") == 1
    codes = read_json(tmp_path / "bad.json")["failures"][0]["codes"]
    assert "E_DIGEST_MISMATCH" in codes


def test_export_all_formats_with_manifest_and_oracle(mini: Path, tmp_path: Path) -> None:
    out = tmp_path / "export"
    assert run_export(mini, "all", out, None) == 0
    assert check_manifest(out) == []
    lines = (out / "rlvr" / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        answer = json.loads(line)["answer"]
        assert grade_prediction(f"ANSWER: {answer['canonical']}", answer) == 1.0


def _first(plugin: str, family: str, work: Path, fault: FaultCode | None) -> dict[str, Any]:
    """The first effective (and, for clean runs, certified) candidate of a fixed seed path, as the
    pipeline would find it by retrying."""
    result: dict[str, Any] = {}
    for i in range(ATTEMPTS):
        result = certify_job(
            {"plugin": plugin, "root": 2026, "path": [plugin, "tier1", i], "family": family,
             "fault": fault.value if fault else None, "work": str(work), "profile": PROFILE}
        )  # fmt: skip
        if result["effective"] and (fault is not None or result["verdict"] == "CERTIFIED"):
            return result
    return result


@pytest.mark.parametrize(("plugin", "family"), [("tangle", "T3_ONTOP"), ("panel", "P2_ZONE_TRACE")])
def test_clean_scene_certifies(tmp_path: Path, plugin: str, family: str) -> None:
    result = _first(plugin, family, tmp_path, None)
    assert result["verdict"] == "CERTIFIED", result["codes"]
    assert result["d1"] == result["d3"]


@pytest.mark.parametrize(
    ("plugin", "family", "fault"),
    [
        ("tangle", "T1_TRACE", FaultCode.M6),
        ("gauge", "G1_READ", FaultCode.M9),
        ("panel", "P1_READ_TRACE", FaultCode.M12),
    ],
)
def test_fault_is_caught_with_its_code(
    tmp_path: Path, plugin: str, family: str, fault: FaultCode
) -> None:
    result = _first(plugin, family, tmp_path, fault)
    assert result["effective"] and result["verdict"] == "QUARANTINED"
    assert FAULT_EXPECTED[fault].value in result["codes"], result["codes"]
