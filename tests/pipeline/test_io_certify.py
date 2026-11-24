"""Task 2.7: bundle I/O + certification orchestrator (SPEC 3, 7.13)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from plugin_helpers import PROFILE

from knotwork.knotcore import seeds
from knotwork.knotcore.contracts import ErrorCode, Family, FaultCode, primary_code
from knotwork.knotcore.seeds import SeedTree
from knotwork.pipeline.certify import quarantine_of, run_candidate, stage_of
from knotwork.pipeline.io import bundle_id_for, file_digests, read_json
from knotwork.plugins.tangle import PLUGIN


def _scene(i: int = 0):  # type: ignore[no-untyped-def]
    at = SeedTree(31).child("io", i)
    scene = PLUGIN.sample_scene(at.child("scene"), dict(PROFILE, _family="T1_TRACE"))
    return scene, PLUGIN.questions(scene, at.child("question"), Family.T1_TRACE)


def test_bundle_files_digests_and_id(tmp_path: Path) -> None:
    scene, q = _scene()
    c = run_candidate(PLUGIN, scene, q, tmp_path / "b", PROFILE)
    assert c.certified, c.certificate.codes
    for f in ("beauty.png", "id_buffer.png", "manifest.json", "claims.json"):
        assert (c.dir / f).is_file()
    assert {k for k in c.files if k.startswith("masks/")} == {
        f"masks/{p.name}" for p in (c.dir / "masks").glob("*.png")
    }
    beauty = np.asarray(Image.open(c.dir / "beauty.png").convert("RGB"), dtype=np.uint8)
    assert c.files["beauty.png"] == seeds.array_digest(beauty)
    assert c.files["manifest.json"] == seeds.json_digest(read_json(c.dir / "manifest.json"))
    assert c.files == file_digests(c.dir)
    assert c.bundle_id == bundle_id_for(c.files, q, c.answer) == c.certificate.bundle_id
    assert c.bundle_id.startswith("kw-") and len(c.bundle_id) == 23


def test_quarantine_stage_and_primary(tmp_path: Path) -> None:
    scene, q = _scene(1)
    cases = [(FaultCode.M7, "witness"), (FaultCode.M5, "witness"), (FaultCode.M2, "kernel")]
    for fault, want in cases:
        c = run_candidate(PLUGIN, scene, q, tmp_path / fault.value, PROFILE, fault)
        rec = quarantine_of("x", "tangle", q.family, "p", c.certificate, fault=fault)
        assert rec.codes and rec.primary_code == primary_code(rec.codes)
        assert stage_of(c.certificate) in ("witness", "legibility", "kernel")
        if fault is FaultCode.M2:
            assert ErrorCode.E_ZORDER in rec.codes and rec.stage == want
        if fault is FaultCode.M7:
            assert rec.stage == want and ErrorCode.E_UNEXPLAINED_INK in rec.codes
    wrong = PLUGIN.symbolic_answer(scene, q).model_copy(update={"value": "99", "canonical": "99"})
    from knotwork.knotcore import engine

    c = run_candidate(PLUGIN, scene, q, tmp_path / "ok", PROFILE)
    cert = engine.certify(c.dir, q, wrong, None, PROFILE["kernel"], c.bundle_id)
    rec = quarantine_of("y", "tangle", q.family, "p", cert)
    assert rec.stage == "certify" and rec.primary_code is ErrorCode.E_D1_D3_MISMATCH
