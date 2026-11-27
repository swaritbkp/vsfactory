"""Certification orchestrator (SPEC 2.3, 7.13): render a candidate, write claims, compute D1/D2,
certify with the kernel, and turn failures into QuarantineRecords.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knotwork.knotcore import engine
from knotwork.knotcore.contracts import (
    Answer,
    AnswerType,
    Certificate,
    ErrorCode,
    Family,
    FaultCode,
    QuarantineRecord,
    QuestionProgram,
    SceneIR,
    TaskPlugin,
    Verdict,
    primary_code,
)
from knotwork.pipeline.io import bundle_id_for, file_digests, write_claims

WITNESS_CHECKS = {
    "contrast",
    "crossing_color",
    "needle_tip_color",
    "unexplained_ink",
    "oob_mask",
    "oob_box",
    "cvd_strands",
    "cvd_zones",
    "oracle_path",
    "oracle_crossing",
    "oracle_tip",
    "oracle_object",
}
LEGIBILITY_CHECKS = {
    "stroke_768",
    "crossing_angle",
    "crossing_spacing",
    "terminal_clearance",
    "ocr_768",
}
PROVISIONAL_ID = "kw-" + "0" * 20


@dataclass
class Candidate:
    dir: Path
    scene: SceneIR
    question: QuestionProgram
    d1: Answer
    d2: Answer | None
    answer: Answer
    bundle_id: str
    certificate: Certificate
    files: dict[str, str]

    @property
    def certified(self) -> bool:
        return self.certificate.verdict is Verdict.CERTIFIED


def emitted_answer(
    d1: Answer, d3: Answer, question: QuestionProgram, profile: dict[str, Any]
) -> Answer:
    """D1 with the SPEC 13.3 float tolerance max(d3.tolerance, grade_tolerance_frac x span)."""
    if d1.type not in (AnswerType.FLOAT, AnswerType.PREMISE):
        return d1
    span = float(question.difficulty.get("span", 0.0))
    tol = max(float(d3.tolerance), float(profile["gauge"]["grade_tolerance_frac"]) * span)
    return d1.model_copy(update={"tolerance": round(tol, 6)})


def run_candidate(
    plugin: TaskPlugin,
    scene: SceneIR,
    question: QuestionProgram,
    out_dir: Path,
    profile: dict[str, Any],
    fault: FaultCode | None = None,
) -> Candidate:
    """Render (optionally with a fault), write claims, certify. D1/D2 always describe the clean scene."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plugin.render(scene, out_dir, fault)
    write_claims(out_dir, plugin.claims(scene))
    d1 = plugin.symbolic_answer(scene, question)
    d2 = plugin.geometric_answer(scene, question)
    cert = engine.certify(out_dir, question, d1, d2, profile["kernel"], PROVISIONAL_ID)
    answer = emitted_answer(d1, cert.d3, question, profile)
    files = file_digests(out_dir)
    bid = bundle_id_for(files, question, answer)
    cert = cert.model_copy(update={"bundle_id": bid})
    return Candidate(out_dir, scene, question, d1, d2, answer, bid, cert, files)


def stage_of(cert: Certificate) -> str:
    """Earliest stage that produced a code: schema < witness < legibility < kernel < certify."""
    failed = {c.name for c in cert.checks if not c.passed}
    if ErrorCode.E_SCHEMA in cert.codes:
        return "schema"
    if failed & WITNESS_CHECKS:
        return "witness"
    if failed & LEGIBILITY_CHECKS:
        return "legibility"
    if cert.claims_failed or any(s.codes for s in cert.steps):
        return "kernel"
    return "certify"


def quarantine_of(
    candidate_id: str,
    plugin: str,
    family: Family | None,
    seed_path: str,
    cert: Certificate,
    detail: str = "",
    fault: FaultCode | None = None,
) -> QuarantineRecord:
    codes = list(cert.codes) or [ErrorCode.E_D1_D3_MISMATCH]
    return QuarantineRecord(
        candidate_id=candidate_id, plugin=plugin, family=family, seed_path=seed_path,
        stage=stage_of(cert), codes=codes, primary_code=primary_code(codes),  # type: ignore[arg-type]
        detail=detail or ", ".join(f"{c.name}:{c.obj_id}" for c in cert.checks if not c.passed)[:500],
        injected_fault=fault,
    )  # fmt: skip


def twin_quarantine(
    candidate_id: str, plugin: str, family: Family, seed_path: str, detail: str
) -> QuarantineRecord:
    return QuarantineRecord(
        candidate_id=candidate_id, plugin=plugin, family=family, seed_path=seed_path, stage="twin",
        codes=[ErrorCode.E_TWIN_INVALID], primary_code=ErrorCode.E_TWIN_INVALID, detail=detail[:500],
    )  # fmt: skip


def certify_job(job: dict[str, Any]) -> dict[str, Any]:
    """Sample, render and certify one scene (picklable for process pools; used by tests and tools).

    job: plugin, root, path (seed path components), family, fault (or None), work, profile.
    """
    from knotwork.knotcore.seeds import SeedTree
    from knotwork.plugins.registry import fault_effective, get_plugin

    plugin = get_plugin(job["plugin"])
    at = SeedTree(int(job["root"])).child(*job["path"])
    fam = Family(job["family"])
    scene = plugin.sample_scene(at.child("scene"), dict(job["profile"], _family=fam.value))
    q = plugin.questions(scene, at.child("question"), fam)
    fault = FaultCode(job["fault"]) if job.get("fault") else None
    effective = fault is None or fault_effective(job["plugin"], scene, fault)
    if not effective:
        return {"effective": False, "path": at.path_str}
    cand = run_candidate(
        plugin, scene, q, Path(job["work"]) / at.digest()[:16], job["profile"], fault
    )
    return {
        "effective": True, "path": at.path_str, "verdict": cand.certificate.verdict.value,
        "codes": [c.value for c in cand.certificate.codes], "d1": cand.d1.canonical, "d3": cand.certificate.d3.canonical,
        "family": cand.question.family.value, "dir": str(cand.dir), "ocr_engine": cand.certificate.ocr_engine,
    }  # fmt: skip
