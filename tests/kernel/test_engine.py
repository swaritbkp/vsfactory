"""Task 1.11: engine execute / check_claims / certify (SPEC 7.12)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fixtures import (
    BG,
    CFG,
    GID,
    Dial,
    gauge_bundle,
    int_answer,
    label_answer,
    panel_bundle,
    tid,
    wires_bundle,
    x_scene,
)

from knotwork.devtools import kernel_audit
from knotwork.knotcore import engine
from knotwork.knotcore.contracts import (
    Answer,
    AnswerType,
    Claim,
    ErrorCode,
    Family,
    Premise,
    PremiseValue,
    QuestionProgram,
    Verdict,
    ZoneName,
)
from knotwork.knotcore.io import load_bundle

BID = "kw-" + "0" * 20


def trace_q(start: str = "A") -> QuestionProgram:
    return QuestionProgram(
        program_id="q", family=Family.T1_TRACE, text="Follow the cable that starts at A.",
        steps=[{"step_id": "s0", "op": "TRACE", "args": [tid("s", start)]}], answer_type=AnswerType.LABEL,
    )  # fmt: skip


def premise_q(claimed: str, holds: bool) -> QuestionProgram:
    return QuestionProgram(
        program_id="p", family=Family.P5_PREMISE_CONTROL if holds else Family.T_GASLIGHT,
        text="Since the gauge connected to socket S is in the zone, what does it read?",
        steps=[
            {"step_id": "s0", "op": "TRACE", "args": ["fx/terminal:s_S"]},
            {"step_id": "s1", "op": "GAUGE_OF", "args": ["$s0"]},
            {"step_id": "s2", "op": "READ", "args": ["$s1"]},
            {"step_id": "s3", "op": "ZONE", "args": ["$s1"]},
        ],
        answer_type=AnswerType.PREMISE,
        premise=Premise(text="in the zone", claimed_zone=ZoneName(claimed), holds=holds),
    )  # fmt: skip


def test_clean_fixture_certifies(x_dir: Path) -> None:
    cert = engine.certify(x_dir, trace_q(), label_answer("1"), label_answer("1"), CFG, BID)
    assert cert.verdict is Verdict.CERTIFIED, cert.codes
    assert cert.claims_checked == 5 and cert.claims_failed == 0 and cert.agree
    assert cert.d3.canonical == "1" and cert.steps[0].evidence["ocr_text"] == "1"


def test_clean_gauge_certifies(gauge_dir: Path) -> None:
    q = QuestionProgram(
        program_id="g", family=Family.G1_READ, text="What value does the gauge needle point to?",
        steps=[{"step_id": "s0", "op": "READ", "args": [GID]}], answer_type=AnswerType.FLOAT,
    )  # fmt: skip
    d1 = Answer(type=AnswerType.FLOAT, value=47.0, tolerance=0.0, canonical="47")
    cert = engine.certify(gauge_dir, q, d1, None, CFG, BID)
    assert cert.verdict is Verdict.CERTIFIED, (cert.codes, [c for c in cert.checks if not c.passed])


@pytest.mark.parametrize(("claimed", "holds"), [("red", True), ("green", False)])
def test_premise_program(tmp_path: Path, claimed: str, holds: bool) -> None:
    b = load_bundle(panel_bundle(tmp_path))  # socket S -> port G2 -> gauge reading 62 (red)
    ans, steps = engine.execute(b, premise_q(claimed, holds), CFG)
    assert [s.op.value for s in steps] == ["TRACE", "GAUGE_OF", "READ", "ZONE"]
    assert ans.type is AnswerType.PREMISE and isinstance(ans.value, PremiseValue)
    assert ans.value.premise_holds is holds
    assert abs(float(ans.value.value) - 62.0) <= ans.tolerance
    assert ans.canonical == f"{'yes' if holds else 'no'}; {format(ans.value.value, 'g')}"


def test_three_step_substitution(tmp_path: Path) -> None:
    b = load_bundle(panel_bundle(tmp_path))
    q = QuestionProgram(
        program_id="p1", family=Family.P1_READ_TRACE, text="What does the gauge on socket T read?",
        steps=[
            {"step_id": "s0", "op": "TRACE", "args": ["fx/terminal:s_T"]},
            {"step_id": "s1", "op": "GAUGE_OF", "args": ["$s0"]},
            {"step_id": "s2", "op": "READ", "args": ["$s1"]},
        ],
        answer_type=AnswerType.FLOAT,
    )  # fmt: skip
    ans, steps = engine.execute(b, q, CFG)
    assert steps[1].inputs == ["G1"] and steps[2].inputs == ["fx/gauge:g1"]
    assert abs(float(ans.value) - 30.0) <= ans.tolerance  # type: ignore[arg-type]


def test_wrong_d1(x_dir: Path) -> None:
    cert = engine.certify(x_dir, trace_q(), label_answer("2"), None, CFG, BID)
    assert cert.verdict is Verdict.QUARANTINED and ErrorCode.E_D1_D3_MISMATCH in cert.codes


def test_d2_disagrees(x_dir: Path) -> None:
    cert = engine.certify(x_dir, trace_q(), label_answer("1"), label_answer("2"), CFG, BID)
    assert cert.codes == [ErrorCode.E_D1_D2_MISMATCH] and not cert.agree


def test_wrong_ontop_claim_is_zorder(tmp_path: Path) -> None:
    sc = x_scene(top="A")
    sc.claims[-1] = Claim(
        claim_id="o_0",
        op="ONTOP",
        args=[tid("s", "A"), tid("s", "B"), 0],
        expected=label_answer("B"),
    )
    cert = engine.certify(wires_bundle(tmp_path, sc), trace_q(), label_answer("1"), None, CFG, BID)
    assert ErrorCode.E_ZORDER in cert.codes and cert.claims_failed == 1


def test_wrong_read_claim_is_scale_mismatch(tmp_path: Path) -> None:
    claims = [
        Claim(
            claim_id="r",
            op="READ",
            args=[GID],
            expected=Answer(type=AnswerType.FLOAT, value=30.0, canonical="30"),
        )
    ]
    d = gauge_bundle(tmp_path, Dial(value=47.0), claims=claims)
    n_failed, codes = engine.check_claims(load_bundle(d), load_bundle(d).claims, CFG)
    assert n_failed == 1 and codes == [ErrorCode.E_SCALE_MISMATCH]


def test_all_stages_run_and_codes_accumulate(tmp_path: Path) -> None:
    sc = x_scene(top="A")
    sc.wires[1].beauty_rgb = tuple(
        int(0.08 * c + 0.92 * g) for c, g in zip(sc.wires[1].color, BG, strict=True)
    )  # type: ignore[assignment]
    sc.claims[2] = Claim(
        claim_id="c_a", op="CROSS", args=[tid("s", "A"), "*"], expected=int_answer(4)
    )
    cert = engine.certify(wires_bundle(tmp_path, sc), trace_q(), label_answer("2"), None, CFG, BID)
    for code in (
        ErrorCode.E_BEAUTY_MISMATCH,
        ErrorCode.E_CLAIM_MISMATCH,
        ErrorCode.E_D1_D3_MISMATCH,
    ):
        assert code in cert.codes, cert.codes
    assert cert.verdict is Verdict.QUARANTINED


def test_load_failure_is_schema(tmp_path: Path) -> None:
    cert = engine.certify(tmp_path, trace_q(), label_answer("1"), None, CFG, BID)
    assert cert.codes == [ErrorCode.E_SCHEMA] and cert.verdict is Verdict.QUARANTINED


def test_kernel_sha_parity() -> None:
    assert engine.kernel_sha256() == kernel_audit.kernel_sha256()
