"""Knotwork data contracts (Pydantic v2).

Single source of truth for every artifact that crosses the trusted-kernel boundary.
The kernel reads ONLY: beauty.png, id_buffer.png, masks/*.png, manifest.json (RenderManifest),
claims.json (ClaimSet) and task.json (TaskBundle.question). It never reads SceneIR.payload.
See SPEC.md sections 3-4 for field semantics. Any change here requires a SCHEMA_VERSION bump.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

if TYPE_CHECKING:
    from knotwork.knotcore.seeds import SeedTree

SCHEMA_VERSION = "1.0.0"
KERNEL_VERSION = "1.0.0"
CANVAS_PX = 1024
CONSUMPTION_PX = 768
SUPERSAMPLE = 2
HEX64 = r"^[0-9a-f]{64}$"
BUNDLE_ID = r"^kw-[0-9a-f]{20}$"
CANARY = r"^KNOTWORK-CANARY-[0-9a-f]{32}$"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class ErrorCode(StrEnum):
    """Declared in priority order: index 0 wins QuarantineRecord.primary_code."""

    E_SCHEMA = "E_SCHEMA"
    E_DIGEST_MISMATCH = "E_DIGEST_MISMATCH"
    E_FONT = "E_FONT"
    E_OOB = "E_OOB"
    E_LABEL_MISMATCH = "E_LABEL_MISMATCH"
    E_UNEXPLAINED_INK = "E_UNEXPLAINED_INK"
    E_BEAUTY_MISMATCH = "E_BEAUTY_MISMATCH"
    E_OCCLUDED_CROSSING = "E_OCCLUDED_CROSSING"
    E_WITNESS_OCCLUDED = "E_WITNESS_OCCLUDED"
    E_TANGENT_CROSSING = "E_TANGENT_CROSSING"
    E_ILLEGIBLE_RES = "E_ILLEGIBLE_RES"
    E_NEEDLE_OFFCENTER = "E_NEEDLE_OFFCENTER"
    E_ZORDER = "E_ZORDER"
    E_SCALE_MISMATCH = "E_SCALE_MISMATCH"
    E_CROSSING_SPACING = "E_CROSSING_SPACING"
    E_TERMINAL_CLEARANCE = "E_TERMINAL_CLEARANCE"
    E_OCR_FAIL = "E_OCR_FAIL"
    E_TRACE_NO_TERMINAL = "E_TRACE_NO_TERMINAL"
    E_TRACE_AMBIGUOUS = "E_TRACE_AMBIGUOUS"
    E_ZONE_UNRESOLVED = "E_ZONE_UNRESOLVED"
    E_AMBIGUOUS_ARGMAX = "E_AMBIGUOUS_ARGMAX"
    E_CLAIM_MISMATCH = "E_CLAIM_MISMATCH"
    E_D1_D3_MISMATCH = "E_D1_D3_MISMATCH"
    E_D1_D2_MISMATCH = "E_D1_D2_MISMATCH"
    E_TWIN_INVALID = "E_TWIN_INVALID"


CODE_PRIORITY: tuple[ErrorCode, ...] = tuple(ErrorCode)


class FaultCode(StrEnum):
    M1, M2, M3, M4, M5 = "M1", "M2", "M3", "M4", "M5"
    M6, M7, M8, M9, M10 = "M6", "M7", "M8", "M9", "M10"


_E = ErrorCode
#: The exact ErrorCode each injected fault MUST produce (it must appear in QuarantineRecord.codes).
FAULT_EXPECTED: dict[FaultCode, ErrorCode] = dict(
    zip(
        FaultCode,
        (
            _E.E_OOB,
            _E.E_ZORDER,
            _E.E_LABEL_MISMATCH,
            _E.E_TANGENT_CROSSING,
            _E.E_ILLEGIBLE_RES,
            _E.E_OCCLUDED_CROSSING,
            _E.E_UNEXPLAINED_INK,
            _E.E_SCALE_MISMATCH,
            _E.E_NEEDLE_OFFCENTER,
            _E.E_BEAUTY_MISMATCH,
        ),
        strict=True,
    )
)

#: Which plugin each fault is injected into by `knot mutate`.
FAULT_PLUGIN: dict[FaultCode, str] = {
    f: ("gauge" if f in (FaultCode.M8, FaultCode.M9) else "tangle") for f in FaultCode
}


def primary_code(codes: list[ErrorCode]) -> ErrorCode:
    """Highest-priority code in `codes` according to CODE_PRIORITY."""
    return min(codes, key=CODE_PRIORITY.index)


class Op(StrEnum):
    TRACE = "TRACE"
    CROSS = "CROSS"
    ONTOP = "ONTOP"
    READ = "READ"
    COUNT = "COUNT"
    CMP = "CMP"
    ARGMAX = "ARGMAX"
    ZONE = "ZONE"
    READ_ALL = "READ_ALL"
    GAUGE_OF = "GAUGE_OF"


#: Exact op sequence of premise programs (P5_PREMISE_CONTROL, T_GASLIGHT); see SPEC 7.12 / 8.3.
PREMISE_OPS: tuple[Op, ...] = (Op.TRACE, Op.GAUGE_OF, Op.READ, Op.ZONE)

#: ErrorCode emitted when a plugin claim (D1) disagrees with the kernel (D3) for a given op.
CLAIM_MISMATCH_CODE: dict[Op, ErrorCode] = {
    op: {Op.ONTOP: ErrorCode.E_ZORDER, Op.READ: ErrorCode.E_SCALE_MISMATCH}.get(
        op, ErrorCode.E_CLAIM_MISMATCH
    )
    for op in Op
}


class Family(StrEnum):
    T1_TRACE = "T1_TRACE"
    T2_CROSS = "T2_CROSS"
    T3_ONTOP = "T3_ONTOP"
    G1_READ = "G1_READ"
    G2_ZONE = "G2_ZONE"
    P1_READ_TRACE = "P1_READ_TRACE"
    P2_ZONE_TRACE = "P2_ZONE_TRACE"
    P3_ARGMAX_PORT = "P3_ARGMAX_PORT"
    P4_COUNT_ZONE = "P4_COUNT_ZONE"
    P5_PREMISE_CONTROL = "P5_PREMISE_CONTROL"
    T_GASLIGHT = "T_GASLIGHT"


class Kind(StrEnum):
    STRAND = "strand"
    TERMINAL = "terminal"
    LABEL = "label"
    DIAL = "dial"
    NEEDLE = "needle"
    TICK = "tick"
    TICK_MINOR = "tick_minor"
    ZONE_ARC = "zone_arc"
    OCCLUDER = "occluder"
    DECOR = "decor"


class ZoneName(StrEnum):
    GREEN = "green"
    AMBER = "amber"
    RED = "red"


class AnswerType(StrEnum):
    LABEL = "label"
    INT = "int"
    FLOAT = "float"
    ZONE = "zone"
    PREMISE = "premise"


class Verdict(StrEnum):
    CERTIFIED = "CERTIFIED"
    QUARANTINED = "QUARANTINED"


class Split(StrEnum):
    DEV = "dev"
    TEST = "test"
    SEALED = "sealed"


class PixelBox(_Model):
    """Half-open integer pixel box [x0, x1) x [y0, y1) in 1024-px canvas coordinates."""

    x0: StrictInt
    y0: StrictInt
    x1: StrictInt
    y1: StrictInt

    @model_validator(mode="after")
    def _non_empty(self) -> PixelBox:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("PixelBox must have x1 > x0 and y1 > y0")
        return self


class ObjectEntry(_Model):
    obj_id: int = Field(ge=1, le=65535)
    name: str = Field(pattern=r"^[a-z]+/[a-z_]+:[A-Za-z0-9_]+$")
    kind: Kind
    mask: str = Field(pattern=r"^masks/obj_\d{5}\.png$")
    color_rgb: tuple[int, int, int]
    parent: str | None = None
    label_text: str | None = None
    box: PixelBox | None = None
    font_px: int | None = Field(default=None, ge=6, le=96)


class TerminalEntry(_Model):
    terminal_id: str = Field(pattern=r"^[a-z]+/terminal:[A-Za-z0-9_]+$")
    role: Literal["start", "end", "port"]
    box: PixelBox
    label_obj: int | None = Field(default=None, ge=1, le=65535)


class TickEntry(_Model):
    tick_obj: int = Field(ge=1, le=65535)
    label_obj: int = Field(ge=1, le=65535)


class ZoneEntry(_Model):
    zone_obj: int = Field(ge=1, le=65535)
    name: ZoneName


class GaugeEntry(_Model):
    gauge_id: str = Field(pattern=r"^[a-z]+/gauge:[A-Za-z0-9_]+$")
    dial_obj: int = Field(ge=1, le=65535)
    needle_obj: int = Field(ge=1, le=65535)
    ticks: list[TickEntry] = Field(min_length=2)
    zones: list[ZoneEntry] = Field(default_factory=list)
    port_terminal: str | None = None
    claimed_center: tuple[float, float]
    claimed_radius: float = Field(gt=0)


class RenderManifest(_Model):
    schema_version: Literal["1.0.0"] = "1.0.0"
    plugin: str
    canvas: tuple[int, int] = (CANVAS_PX, CANVAS_PX)
    supersample: int = SUPERSAMPLE
    background_rgb: tuple[int, int, int]
    beauty: Literal["beauty.png"] = "beauty.png"
    id_buffer: Literal["id_buffer.png"] = "id_buffer.png"
    font_sha256: str = Field(pattern=HEX64)
    objects: list[ObjectEntry] = Field(min_length=1)
    terminals: list[TerminalEntry] = Field(default_factory=list)
    gauges: list[GaugeEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _referential_integrity(self) -> RenderManifest:
        ids = [o.obj_id for o in self.objects]
        names = [o.name for o in self.objects]
        if len(set(ids)) != len(ids) or len(set(names)) != len(names):
            raise ValueError("obj_id and name must be unique")
        known = set(ids)
        tids = [t.terminal_id for t in self.terminals]
        if len(set(tids)) != len(tids):
            raise ValueError("terminal_id must be unique")
        refs = [t.label_obj for t in self.terminals if t.label_obj is not None]
        for g in self.gauges:
            refs += [g.dial_obj, g.needle_obj, *(t.tick_obj for t in g.ticks)]
            refs += [t.label_obj for t in g.ticks] + [z.zone_obj for z in g.zones]
            if g.port_terminal is not None and g.port_terminal not in tids:
                raise ValueError(f"unknown port_terminal {g.port_terminal}")
        missing = sorted(set(refs) - known)
        if missing:
            raise ValueError(f"manifest references unknown obj_ids {missing}")
        return self

    def obj(self, obj_id: int) -> ObjectEntry:
        for o in self.objects:
            if o.obj_id == obj_id:
                return o
        raise KeyError(obj_id)


Scalar = StrictBool | StrictInt | StrictFloat | StrictStr


class PremiseValue(_Model):
    premise_holds: StrictBool
    value: StrictFloat | StrictInt | StrictStr


class Answer(_Model):
    type: AnswerType
    value: Scalar | PremiseValue
    tolerance: float = Field(default=0.0, ge=0.0)
    canonical: str


class Claim(_Model):
    claim_id: str
    op: Op
    args: list[Scalar]
    expected: Answer


class ClaimSet(_Model):
    plugin: str
    claims: list[Claim]


class ProgramStep(_Model):
    step_id: str = Field(pattern=r"^s\d+$")
    op: Op
    args: list[Scalar]


class Premise(_Model):
    text: str
    claimed_zone: ZoneName
    holds: bool


class QuestionProgram(_Model):
    program_id: str
    family: Family
    steps: list[ProgramStep] = Field(min_length=1)
    text: str = Field(min_length=8)
    answer_type: AnswerType
    options: list[str] | None = None
    premise: Premise | None = None
    difficulty: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _refs_resolve(self) -> QuestionProgram:
        seen: set[str] = set()
        for step in self.steps:
            for a in step.args:
                if isinstance(a, str) and a.startswith("$") and a[1:] not in seen:
                    raise ValueError(f"step {step.step_id} references unknown {a}")
            seen.add(step.step_id)
        gaslight = self.family is Family.T_GASLIGHT
        if gaslight != (self.premise is not None and not self.premise.holds):
            raise ValueError("T_GASLIGHT iff premise present and premise.holds is False")
        if self.family is Family.P5_PREMISE_CONTROL and not (self.premise and self.premise.holds):
            raise ValueError("P5_PREMISE_CONTROL requires a true premise")
        if self.premise is not None:
            ops = [st.op for st in self.steps]
            if ops != list(PREMISE_OPS) or self.answer_type is not AnswerType.PREMISE:
                raise ValueError(
                    "premise programs are exactly TRACE, GAUGE_OF, READ, ZONE -> premise"
                )
        return self


class StepTrace(_Model):
    step_id: str
    op: Op
    inputs: list[Any]
    output: Any
    evidence: dict[str, Any] = Field(default_factory=dict)
    codes: list[ErrorCode] = Field(default_factory=list)


class WitnessCheck(_Model):
    name: str
    obj_id: int | None = None
    passed: bool
    metric: float
    threshold: float
    code: ErrorCode


class Certificate(_Model):
    bundle_id: str = Field(pattern=BUNDLE_ID)
    kernel_version: str = KERNEL_VERSION
    kernel_sha256: str = Field(pattern=HEX64)
    ocr_engine: Literal["rapidocr", "template"]
    d1: Answer
    d2: Answer | None = None
    d3: Answer
    agree: bool
    steps: list[StepTrace]
    claims_checked: int = Field(ge=0)
    claims_failed: int = Field(ge=0)
    checks: list[WitnessCheck]
    verdict: Verdict
    codes: list[ErrorCode] = Field(default_factory=list)

    @model_validator(mode="after")
    def _verdict_consistent(self) -> Certificate:
        clean = self.agree and not self.codes and self.claims_failed == 0
        if (self.verdict is Verdict.CERTIFIED) != clean:
            raise ValueError("CERTIFIED iff agree and no codes and no failed claims")
        return self


class QuarantineRecord(_Model):
    candidate_id: str
    plugin: str
    family: Family | None = None
    seed_path: str
    stage: Literal["schema", "render", "witness", "legibility", "kernel", "certify", "twin"]
    codes: list[ErrorCode] = Field(min_length=1)
    primary_code: ErrorCode
    detail: str
    injected_fault: FaultCode | None = None

    @model_validator(mode="after")
    def _primary_in_codes(self) -> QuarantineRecord:
        if self.primary_code not in self.codes:
            raise ValueError("primary_code must be one of codes")
        return self


class FreshnessReceipt(_Model):
    receipt_id: str = Field(pattern=HEX64)
    protocol_commit: str = Field(pattern=HEX64)
    source: Literal["drand", "nist", "offline"]
    round: int | None = None
    randomness_hex: str = Field(pattern=HEX64)
    root_seed: int = Field(ge=0, lt=2**128)
    beacon_url: str | None = None
    fetched_at_utc: str | None = None
    emission_mask_commit: str = Field(pattern=HEX64)
    canary: str = Field(pattern=CANARY)


class TwinRefs(_Model):
    role: Literal["base", "causal", "nuisance"]
    base_id: str | None = None
    causal_id: str | None = None
    nuisance_id: str | None = None
    edit: str | None = None
    pixel_diff_frac: float | None = Field(default=None, ge=0.0, le=1.0)


class TaskBundle(_Model):
    schema_version: Literal["1.0.0"] = "1.0.0"
    bundle_id: str = Field(pattern=BUNDLE_ID)
    plugin: str
    family: Family
    split: Split
    seed_path: str
    scene_id: str = Field(pattern=HEX64)
    question: QuestionProgram
    answer: Answer
    certificate: Certificate
    freshness_receipt_id: str = Field(pattern=HEX64)
    twins: TwinRefs
    canary: str = Field(pattern=CANARY)
    files: dict[str, str]

    @model_validator(mode="after")
    def _ids_agree(self) -> TaskBundle:
        if self.certificate.bundle_id != self.bundle_id:
            raise ValueError("certificate.bundle_id != bundle_id")
        if self.certificate.verdict is not Verdict.CERTIFIED:
            raise ValueError("only CERTIFIED bundles may be emitted as TaskBundle")
        return self


class HarborTask(_Model):
    task_name: str = Field(pattern=r"^knotwork/[a-z0-9-]+$")
    bundle_id: str = Field(pattern=BUNDLE_ID)
    instruction_md: str
    image_relpath: Literal["environment/image.png"] = "environment/image.png"
    expected: Answer
    difficulty: Literal["easy", "medium", "hard"]
    category: Literal["visual-reasoning"] = "visual-reasoning"
    tags: list[str]
    agent_timeout_sec: float = 300.0
    verifier_timeout_sec: float = 120.0
    canary: str = Field(pattern=CANARY)


class SceneIR(_Model):
    """Plugin-side scene description. `payload` is opaque to knotcore and never read by it."""

    plugin: str
    plugin_version: str
    seed_path: str
    params: dict[str, Any]
    payload: dict[str, Any]


@runtime_checkable
class TaskPlugin(Protocol):
    """Contract every untrusted plugin implements. Plugins may import knotcore; never the reverse."""

    name: str
    version: str
    families: tuple[Family, ...]

    def sample_scene(self, seeds: SeedTree, profile: dict[str, Any]) -> SceneIR: ...

    def render(
        self, scene: SceneIR, out_dir: Path, fault: FaultCode | None = None
    ) -> RenderManifest: ...

    def claims(self, scene: SceneIR) -> ClaimSet: ...

    def questions(self, scene: SceneIR, seeds: SeedTree, family: Family) -> QuestionProgram: ...

    def symbolic_answer(self, scene: SceneIR, program: QuestionProgram) -> Answer: ...

    def geometric_answer(self, scene: SceneIR, program: QuestionProgram) -> Answer | None: ...

    def causal_twin(
        self, scene: SceneIR, program: QuestionProgram, seeds: SeedTree
    ) -> tuple[SceneIR, str]: ...

    def nuisance_twin(
        self, scene: SceneIR, program: QuestionProgram, seeds: SeedTree
    ) -> SceneIR: ...


class RenderBundle(_Model):
    """Paths of one rendered candidate on disk (all relative to `root`)."""

    root: str
    manifest: RenderManifest
    file_digests: dict[str, str]
