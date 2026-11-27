"""Contract (schema) conformance: validators, enums, fault map and JSON-schema export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from knotwork.knotcore import contracts as C

H = "a" * 64
BID = "kw-" + "0" * 20
CANARY = "KNOTWORK-CANARY-" + "f" * 32


def _obj(i: int, kind: C.Kind = C.Kind.STRAND, **kw: object) -> C.ObjectEntry:
    return C.ObjectEntry(
        obj_id=i,
        name=f"tangle/{kind.value}:{i}",
        kind=kind,
        mask=f"masks/obj_{i:05d}.png",
        color_rgb=(10, 20, 30),
        **kw,  # type: ignore[arg-type]
    )


def _manifest(**kw: object) -> C.RenderManifest:
    base: dict[str, object] = {
        "plugin": "tangle",
        "background_rgb": (250, 250, 245),
        "font_sha256": H,
        "objects": [_obj(1), _obj(2, C.Kind.LABEL, label_text="A")],
        "terminals": [
            C.TerminalEntry(
                terminal_id="tangle/terminal:s0",
                role="start",
                box=C.PixelBox(x0=0, y0=0, x1=10, y1=10),
                label_obj=2,
            )
        ],
    }
    base.update(kw)
    return C.RenderManifest(**base)  # type: ignore[arg-type]


def _answer(v: object = "B") -> C.Answer:
    return C.Answer(type=C.AnswerType.LABEL, value=v, canonical=str(v))  # type: ignore[arg-type]


def _program(**kw: object) -> C.QuestionProgram:
    base: dict[str, object] = {
        "program_id": "p0",
        "family": C.Family.T1_TRACE,
        "steps": [C.ProgramStep(step_id="s0", op=C.Op.TRACE, args=["tangle/strand:1"])],
        "text": "Follow the cable from terminal A. Which numbered terminal does it reach?",
        "answer_type": C.AnswerType.LABEL,
        "options": ["1", "2", "3"],
    }
    base.update(kw)
    return C.QuestionProgram(**base)  # type: ignore[arg-type]


def _cert(**kw: object) -> C.Certificate:
    base: dict[str, object] = {
        "bundle_id": BID,
        "kernel_sha256": H,
        "ocr_engine": "rapidocr",
        "d1": _answer(),
        "d3": _answer(),
        "agree": True,
        "steps": [],
        "claims_checked": 3,
        "claims_failed": 0,
        "checks": [],
        "verdict": C.Verdict.CERTIFIED,
    }
    base.update(kw)
    return C.Certificate(**base)  # type: ignore[arg-type]


def test_fault_map_is_complete_and_exact() -> None:
    assert [f.value for f in C.FaultCode] == [f"M{i}" for i in range(1, 15)]
    assert set(C.FAULT_EXPECTED) == set(C.FaultCode)
    first_ten = [C.FAULT_EXPECTED[f] for f in list(C.FaultCode)[:10]]
    assert len(set(first_ten)) == 10
    assert C.FAULT_EXPECTED[C.FaultCode.M10] is C.ErrorCode.E_BEAUTY_MISMATCH
    assert {C.FAULT_PLUGIN[f] for f in (C.FaultCode.M8, C.FaultCode.M9)} == {"gauge"}
    assert {C.FAULT_PLUGIN[f] for f in list(C.FaultCode)[10:]} == {"panel"}
    assert set(C.FAULT_PLUGIN) == set(C.FaultCode)


def test_code_priority_is_a_total_order() -> None:
    assert sorted(C.CODE_PRIORITY, key=str) == sorted(C.ErrorCode, key=str)
    assert len(set(C.CODE_PRIORITY)) == len(C.ErrorCode)
    assert C.primary_code([C.ErrorCode.E_ZORDER, C.ErrorCode.E_OOB]) is C.ErrorCode.E_OOB


def test_claim_mismatch_codes() -> None:
    assert C.CLAIM_MISMATCH_CODE[C.Op.ONTOP] is C.ErrorCode.E_ZORDER
    assert C.CLAIM_MISMATCH_CODE[C.Op.READ] is C.ErrorCode.E_SCALE_MISMATCH
    assert C.CLAIM_MISMATCH_CODE[C.Op.TRACE] is C.ErrorCode.E_CLAIM_MISMATCH


def test_pixel_box_rejects_empty_but_allows_out_of_bounds() -> None:
    with pytest.raises(ValidationError):
        C.PixelBox(x0=5, y0=0, x1=5, y1=3)
    C.PixelBox(x0=-4, y0=1000, x1=3, y1=1030)  # OOB is a kernel finding (E_OOB), not a schema error


def test_manifest_referential_integrity() -> None:
    m = _manifest()
    assert m.obj(2).label_text == "A"
    with pytest.raises(ValidationError):
        _manifest(objects=[_obj(1), _obj(1)])
    with pytest.raises(ValidationError):
        _manifest(objects=[_obj(1)])  # terminal label_obj=2 missing
    with pytest.raises(ValidationError):
        _obj(0)
    with pytest.raises(ValidationError):
        C.ObjectEntry(
            obj_id=3,
            name="Bad Name",
            kind=C.Kind.DIAL,
            mask="masks/obj_00003.png",
            color_rgb=(0, 0, 0),
        )


def test_manifest_gauge_refs() -> None:
    objs = [
        _obj(1, C.Kind.DIAL),
        _obj(2, C.Kind.NEEDLE),
        _obj(3, C.Kind.TICK),
        _obj(4, C.Kind.LABEL, label_text="00"),
        _obj(5, C.Kind.TICK),
        _obj(6, C.Kind.LABEL, label_text="20"),
    ]
    g = C.GaugeEntry(
        gauge_id="gauge/gauge:g0",
        dial_obj=1,
        needle_obj=2,
        ticks=[C.TickEntry(tick_obj=3, label_obj=4), C.TickEntry(tick_obj=5, label_obj=6)],
        claimed_center=(512.0, 512.0),
        claimed_radius=300.0,
    )
    _manifest(plugin="gauge", objects=objs, terminals=[], gauges=[g])
    bad = g.model_copy(update={"needle_obj": 99})
    with pytest.raises(ValidationError):
        _manifest(plugin="gauge", objects=objs, terminals=[], gauges=[bad])


def _premise_steps(last: C.Op = C.Op.ZONE) -> list[C.ProgramStep]:
    return [
        C.ProgramStep(step_id="s0", op=C.Op.TRACE, args=["panel/terminal:sA"]),
        C.ProgramStep(step_id="s1", op=C.Op.GAUGE_OF, args=["$s0"]),
        C.ProgramStep(step_id="s2", op=C.Op.READ, args=["$s1"]),
        C.ProgramStep(step_id="s3", op=last, args=["$s1"]),
    ]


def test_program_reference_and_gaslight_rules() -> None:
    _program()
    with pytest.raises(ValidationError):
        _program(steps=[C.ProgramStep(step_id="s0", op=C.Op.READ, args=["$s1"])])
    false_premise = C.Premise(
        text="Since it is green...", claimed_zone=C.ZoneName.GREEN, holds=False
    )
    true_premise = C.Premise(text="Since it is red...", claimed_zone=C.ZoneName.RED, holds=True)
    prem = {"steps": _premise_steps(), "answer_type": C.AnswerType.PREMISE, "options": None}
    _program(family=C.Family.T_GASLIGHT, premise=false_premise, **prem)
    _program(family=C.Family.P5_PREMISE_CONTROL, premise=true_premise, **prem)
    with pytest.raises(ValidationError):
        _program(family=C.Family.T_GASLIGHT, premise=true_premise, **prem)
    with pytest.raises(ValidationError):
        _program(family=C.Family.T1_TRACE, premise=false_premise, **prem)
    with pytest.raises(ValidationError):
        _program(family=C.Family.P5_PREMISE_CONTROL, premise=false_premise, **prem)


def test_premise_programs_need_exact_four_steps() -> None:
    false_premise = C.Premise(
        text="Since it is green...", claimed_zone=C.ZoneName.GREEN, holds=False
    )
    assert C.PREMISE_OPS == (C.Op.TRACE, C.Op.GAUGE_OF, C.Op.READ, C.Op.ZONE)
    ok = _program(
        family=C.Family.T_GASLIGHT,
        premise=false_premise,
        steps=_premise_steps(),
        answer_type=C.AnswerType.PREMISE,
    )
    assert [s.op for s in ok.steps] == list(C.PREMISE_OPS)
    with pytest.raises(ValidationError):  # P1-style 3 steps: ZONE never evaluated
        _program(
            family=C.Family.T_GASLIGHT,
            premise=false_premise,
            steps=_premise_steps()[:3],
            answer_type=C.AnswerType.PREMISE,
        )
    with pytest.raises(ValidationError):  # wrong final op
        _program(
            family=C.Family.T_GASLIGHT,
            premise=false_premise,
            steps=_premise_steps(C.Op.READ),
            answer_type=C.AnswerType.PREMISE,
        )
    with pytest.raises(ValidationError):  # premise answers must be typed PREMISE
        _program(
            family=C.Family.T_GASLIGHT,
            premise=false_premise,
            steps=_premise_steps(),
            answer_type=C.AnswerType.FLOAT,
        )


def test_answer_strict_scalars_and_premise() -> None:
    assert _answer(3).value == 3 and isinstance(_answer(3).value, int)
    assert _answer(True).value is True
    pv = C.PremiseValue(premise_holds=False, value=47.0)
    a = C.Answer(type=C.AnswerType.PREMISE, value=pv, tolerance=1.5, canonical="no;47")
    assert isinstance(a.value, C.PremiseValue)
    with pytest.raises(ValidationError):
        C.Answer(type=C.AnswerType.FLOAT, value=1.0, tolerance=-1, canonical="1")


def test_certificate_verdict_consistency() -> None:
    _cert()
    with pytest.raises(ValidationError):
        _cert(agree=False)
    with pytest.raises(ValidationError):
        _cert(codes=[C.ErrorCode.E_OOB])
    _cert(agree=False, verdict=C.Verdict.QUARANTINED, codes=[C.ErrorCode.E_D1_D3_MISMATCH])


def test_quarantine_primary_in_codes() -> None:
    C.QuarantineRecord(
        candidate_id="c0",
        plugin="tangle",
        seed_path="s:tangle/i:0",
        stage="witness",
        codes=[C.ErrorCode.E_BEAUTY_MISMATCH],
        primary_code=C.ErrorCode.E_BEAUTY_MISMATCH,
        detail="contrast 0.02 < 0.12",
        injected_fault=C.FaultCode.M10,
    )
    with pytest.raises(ValidationError):
        C.QuarantineRecord(
            candidate_id="c0",
            plugin="tangle",
            seed_path="x",
            stage="witness",
            codes=[C.ErrorCode.E_OOB],
            primary_code=C.ErrorCode.E_ZORDER,
            detail="",
        )


def test_task_bundle_requires_certified_and_matching_ids() -> None:
    receipt = C.FreshnessReceipt(
        receipt_id=H,
        protocol_commit=H,
        source="offline",
        randomness_hex=H,
        root_seed=5,
        emission_mask_commit=H,
        canary=CANARY,
    )
    kwargs: dict[str, object] = {
        "bundle_id": BID,
        "plugin": "tangle",
        "family": C.Family.T1_TRACE,
        "split": C.Split.DEV,
        "seed_path": "s:tangle/i:0",
        "scene_id": H,
        "question": _program(),
        "answer": _answer(),
        "certificate": _cert(),
        "freshness_receipt_id": receipt.receipt_id,
        "twins": C.TwinRefs(role="base", causal_id=BID, nuisance_id=BID),
        "canary": CANARY,
        "files": {"beauty.png": H},
    }
    tb = C.TaskBundle(**kwargs)  # type: ignore[arg-type]
    again = C.TaskBundle.model_validate_json(tb.model_dump_json())
    assert again == tb
    with pytest.raises(ValidationError):
        C.TaskBundle(**{**kwargs, "bundle_id": "kw-" + "1" * 20})  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        C.TaskBundle(**{**kwargs, "canary": "nope"})  # type: ignore[arg-type]


def test_models_are_frozen_and_forbid_extra() -> None:
    box = C.PixelBox(x0=0, y0=0, x1=1, y1=1)
    with pytest.raises(ValidationError):
        box.x0 = 3  # type: ignore[misc]
    with pytest.raises(ValidationError):
        C.PixelBox(x0=0, y0=0, x1=1, y1=1, z=2)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "model",
    [
        C.SceneIR,
        C.RenderManifest,
        C.RenderBundle,
        C.QuestionProgram,
        C.StepTrace,
        C.Certificate,
        C.QuarantineRecord,
        C.FreshnessReceipt,
        C.TaskBundle,
        C.HarborTask,
        C.ClaimSet,
    ],
)
def test_json_schema_export(model: type[BaseModel], tmp_path: Path) -> None:
    schema = model.model_json_schema()
    (tmp_path / f"{model.__name__}.json").write_text(json.dumps(schema))
    assert schema["type"] == "object" and schema.get("additionalProperties") is False


def test_task_plugin_protocol_is_runtime_checkable() -> None:
    class Dummy:
        name, version, families = "dummy", "0", (C.Family.T1_TRACE,)

        def sample_scene(self, seeds, profile): ...  # type: ignore[no-untyped-def]
        def render(self, scene, out_dir, fault=None): ...  # type: ignore[no-untyped-def]
        def claims(self, scene): ...  # type: ignore[no-untyped-def]
        def questions(self, scene, seeds, family): ...  # type: ignore[no-untyped-def]
        def symbolic_answer(self, scene, program): ...  # type: ignore[no-untyped-def]
        def geometric_answer(self, scene, program): ...  # type: ignore[no-untyped-def]
        def causal_twin(self, scene, program, seeds): ...  # type: ignore[no-untyped-def]
        def nuisance_twin(self, scene, program, seeds): ...  # type: ignore[no-untyped-def]

    assert isinstance(Dummy(), C.TaskPlugin)
    assert not isinstance(object(), C.TaskPlugin)
