"""Kernel engine: run question programs, re-check plugin claims, issue certificates (SPEC 7.12).

Order: load -> witness -> legibility -> claims -> execute -> compare. Every stage runs and codes
accumulate (only a load failure stops early). CERTIFIED iff D1 == D3 (and D2 == D1 when given), all
claims hold and no check failed. Premise answers are built here from READ and ZONE, never from the
plugin's `premise.holds`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from knotwork.knotcore import contracts as C
from knotwork.knotcore import gauge, logic, topo
from knotwork.knotcore.io import Bundle, KernelFault, load_bundle
from knotwork.knotcore.legibility import legibility
from knotwork.knotcore.witness import witness

A, Op = C.AnswerType, C.Op
OPS: dict[C.Op, Callable[[Bundle, list[Any], dict[str, Any]], tuple[Any, dict[str, Any]]]] = {
    Op.TRACE: lambda b, a, c: topo.trace(b, str(a[0]), c),
    Op.CROSS: lambda b, a, c: topo.cross(b, str(a[0]), str(a[1])),
    Op.ONTOP: lambda b, a, c: topo.ontop(b, str(a[0]), str(a[1]), int(a[2]), c),
    Op.READ: lambda b, a, c: gauge.read(b, str(a[0]), c),
    Op.ZONE: lambda b, a, c: gauge.zone(b, str(a[0]), c),
    Op.GAUGE_OF: lambda b, a, c: gauge.gauge_of(b, str(a[0]), c),
    Op.READ_ALL: lambda b, a, c: gauge.read_all(b, [str(x) for x in a], c),
    Op.COUNT: lambda b, a, c: logic.count(b, str(a[0]), str(a[1]), c),
    Op.CMP: lambda b, a, c: logic.cmp(a[0], a[1]),
    Op.ARGMAX: lambda b, a, c: logic.argmax(a[0]),
}


def kernel_sha256() -> str:
    """Digest of the kernel source (same algorithm as devtools.kernel_audit.kernel_sha256)."""
    root, h = Path(__file__).resolve().parent, hashlib.sha256()
    for path in sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts):
        h.update(path.relative_to(root).as_posix().encode() + b"\0")
        h.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()


def to_answer(t: C.AnswerType, out: Any) -> C.Answer:
    if t is A.FLOAT:
        v, tol = float(out[0]), float(out[1])
        return C.Answer(type=t, value=v, tolerance=tol, canonical=format(v, "g"))
    if t is A.INT:
        return C.Answer(type=t, value=int(out), canonical=str(int(out)))
    return C.Answer(type=t, value=str(out), canonical=str(out))


def agree(x: C.Answer, y: C.Answer, tol: float) -> bool:
    """Exact canonical equality; floats (and premise values) within `tol`."""
    vx, vy = x.value, y.value
    if x.type is not y.type:
        return False
    if isinstance(vx, C.PremiseValue) and isinstance(vy, C.PremiseValue):
        same = vx.premise_holds is vy.premise_holds
        return same and abs(float(vx.value) - float(vy.value)) <= tol
    if x.type is A.FLOAT and isinstance(vx, int | float) and isinstance(vy, int | float):
        return abs(float(vx) - float(vy)) <= tol
    return x.canonical == y.canonical


def execute(
    b: Bundle, q: C.QuestionProgram, cfg: dict[str, Any]
) -> tuple[C.Answer, list[C.StepTrace]]:
    out: dict[str, Any] = {}
    steps: list[C.StepTrace] = []
    for st in q.steps:
        args = [out[a[1:]] if isinstance(a, str) and a.startswith("$") else a for a in st.args]
        trace: dict[str, Any] = {"step_id": st.step_id, "op": st.op, "inputs": args}
        try:
            out[st.step_id], ev = OPS[st.op](b, args, cfg)
        except KernelFault as f:
            steps.append(C.StepTrace(**trace, output=None, codes=[f.code]))
            return C.Answer(type=q.answer_type, value="", canonical=""), steps
        steps.append(C.StepTrace(**trace, output=out[st.step_id], evidence=ev))
    if q.premise is None:
        return to_answer(q.answer_type, out[q.steps[-1].step_id]), steps
    (reading, tol), holds = out["s2"], out["s3"] == q.premise.claimed_zone.value
    value = C.PremiseValue(premise_holds=holds, value=float(reading))
    canonical = f"{'yes' if holds else 'no'}; {format(reading, 'g')}"
    return C.Answer(type=A.PREMISE, value=value, tolerance=float(tol), canonical=canonical), steps


def check_claims(
    b: Bundle, claims: C.ClaimSet, cfg: dict[str, Any]
) -> tuple[int, list[C.ErrorCode]]:
    """(number of failed claims, codes); a disagreeing claim yields CLAIM_MISMATCH_CODE[op]."""
    codes: list[C.ErrorCode] = []
    for c in claims.claims:
        try:
            got = to_answer(c.expected.type, OPS[c.op](b, list(c.args), cfg)[0])
            if not agree(c.expected, got, got.tolerance):
                codes.append(C.CLAIM_MISMATCH_CODE[c.op])
        except KernelFault as f:
            codes.append(f.code)
    return len(codes), codes


def certify(
    bundle_dir: Path, q: C.QuestionProgram, d1: C.Answer, d2: C.Answer | None, cfg: dict[str, Any], bundle_id: str
) -> C.Certificate:  # fmt: skip
    base: dict[str, Any] = {
        "bundle_id": bundle_id,
        "kernel_sha256": kernel_sha256(),
        "d1": d1,
        "d2": d2,
    }
    base |= {"d3": C.Answer(type=q.answer_type, value="", canonical=""), "steps": [], "checks": []}
    base |= {"agree": False, "claims_checked": 0, "claims_failed": 0, "ocr_engine": "rapidocr"}
    try:
        b = load_bundle(Path(bundle_dir))
    except KernelFault as f:
        return C.Certificate(**base, verdict=C.Verdict.QUARANTINED, codes=[f.code])
    checks = witness(b, cfg) + legibility(b, cfg)
    codes = [c.code for c in checks if not c.passed]
    n_failed, claim_codes = check_claims(b, b.claims, cfg)
    d3, steps = execute(b, q, cfg)
    step_codes = [c for s in steps for c in s.codes]
    ok = not step_codes and agree(d1, d3, d3.tolerance)
    codes += claim_codes + step_codes + ([] if ok or step_codes else [C.ErrorCode.E_D1_D3_MISMATCH])
    if d2 is not None and not agree(d2, d1, max(d1.tolerance, d3.tolerance)):
        ok, codes = False, [*codes, C.ErrorCode.E_D1_D2_MISMATCH]
    codes = sorted(set(codes), key=C.CODE_PRIORITY.index)
    template = any(e == "template" for _, e in b.memo.get("ocr", {}).values())
    base |= {"d3": d3, "steps": steps, "checks": checks, "agree": ok, "claims_failed": n_failed}
    base |= {
        "claims_checked": len(b.claims.claims),
        "ocr_engine": "template" if template else "rapidocr",
    }
    verdict = C.Verdict.CERTIFIED if ok and not codes and not n_failed else C.Verdict.QUARANTINED
    return C.Certificate(**base, verdict=verdict, codes=codes)
