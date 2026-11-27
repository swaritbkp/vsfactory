"""Typed scene payloads, validated where a plugin hands a scene to the pipeline.

`SceneIR.payload` stays an opaque mapping in the contracts (the kernel never reads it); these models
make a malformed or misspelled payload fail at the plugin boundary instead of deep in rendering.
Validation never rewrites the mapping, so content digests are those of the plugin's own values.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RGB = tuple[int, int, int]
ColorRegime = Literal["unique", "partial", "mono"]
Style = Literal["schematic", "harness", "flat"]
ZoneColour = Literal["green", "amber", "red"]


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Surface(_Payload):
    seed: int
    light_angle: float
    light: float = Field(ge=0.0)
    grain: float = Field(ge=0.0)
    brush: float = Field(ge=0.0)
    brush_dir: Literal[0, 1]
    smudges: int = Field(ge=0)
    dust: int = Field(ge=0)
    finish: str
    noise: float = Field(default=0.0, ge=0.0, le=1.0)
    dpi_scale: float = Field(default=1.0, ge=0.5, le=1.0)


class Render(_Payload):
    background_rgb: list[int] = Field(min_length=3, max_length=3)
    font_path: str
    font_sha256: str
    canvas_px: int
    supersample: int


class Dial(_Payload):
    name: str
    center: list[float] = Field(min_length=2, max_length=2)
    radius: float = Field(gt=0)
    scale: str
    labels: list[str] = Field(min_length=2)
    vmin: float
    vmax: float
    zones: list[float] = Field(min_length=2, max_length=2)
    bezel_rgb: list[int]
    band_inner: float
    face_rgb: list[int]
    unit: str
    serial: str
    minor_per_major: int = Field(ge=0)
    surface: Surface
    label_frac: float
    value: float
    zone_order: list[ZoneColour] | None = None

    @model_validator(mode="after")
    def _value_on_scale(self) -> Dial:
        if not self.vmin <= self.value <= self.vmax or self.vmin >= self.vmax:
            raise ValueError("dial value must lie on an increasing scale")
        if self.zone_order is not None and len(set(self.zone_order)) != 3:
            raise ValueError("zone_order must name each zone once")
        return self


class TanglePayload(_Payload):
    n: int = Field(ge=2)
    word: list[list[int] | None]
    perm: list[int]
    stroke: float = Field(gt=0)
    layout: dict[str, Any]
    style: Style
    t2_target: int = Field(ge=0)
    render: Render
    surface: Surface
    start_labels: list[str]
    end_labels: list[str]
    colors: list[list[int]]
    color_regime: ColorRegime
    relief: list[list[float]]
    answer_slot: int | None = None
    difficulty_spec: dict[str, Any] | None = None
    plain: bool = False  # no screws, enclosure furniture or relief (diagnostic twins)

    @model_validator(mode="after")
    def _one_per_strand(self) -> TanglePayload:
        sizes = {len(self.perm), len(self.start_labels), len(self.end_labels), len(self.colors)}
        if sizes != {self.n}:
            raise ValueError("perm, labels and colors need one entry per strand")
        return self


class RackPatchPayload(TanglePayload):
    """A tangle loom drawn as a rack patch bay: jacket style and per-port link LEDs."""

    jacket: Literal["mono", "dual", "mixed"]
    leds: list[list[int]] = Field(min_length=2, max_length=2)


class GaugePayload(_Payload):
    dial: Dial
    render: Render
    grade_tol: float = Field(gt=0)
    surface: Surface
    stem: dict[str, Any] | None = None
    plate: dict[str, Any] | None = None
    relief: list[list[float]]
    difficulty_spec: dict[str, Any] | None = None
    plain: bool = False  # no screws, enclosure furniture or relief (diagnostic twins)


class PanelPayload(_Payload):
    n: int = Field(ge=1)
    k: int = Field(ge=1)
    word: list[list[int]]
    perm: list[int]
    stroke: float = Field(gt=0)
    braid: dict[str, Any]
    style: Style
    colors: list[list[int]]
    color_regime: ColorRegime
    sockets: dict[str, str]
    dests: list[list[float]]
    dest_kind: list[Literal["port", "pad"]]
    dest_gauge: list[int]
    dials: list[Dial]
    render: Render
    grade_tol: float = Field(gt=0)
    surface: Surface
    panel_id: int
    argmax_margin: float
    target: str | None
    claimed: ZoneColour | None
    mirror: bool
    relief: list[list[float]]
    ordinal_hint: int | None = Field(default=None, alias="_ordinal")
    difficulty_spec: dict[str, Any] | None = None
    plain: bool = False  # no screws, enclosure furniture or relief (diagnostic twins)

    @model_validator(mode="after")
    def _wiring(self) -> PanelPayload:
        if len(self.dials) != self.k or len(self.sockets) != self.k:
            raise ValueError("one dial and one socket per gauge")
        if {len(self.dests), len(self.dest_kind), len(self.dest_gauge), len(self.colors)} != {
            self.n
        }:
            raise ValueError("destinations and colors need one entry per cable")
        return self


MODELS: dict[str, type[BaseModel]] = {
    "tangle": TanglePayload,
    "gauge": GaugePayload,
    "panel": PanelPayload,
    "rack_patch": RackPatchPayload,
}


def validate_payload(plugin: str, payload: dict[str, Any]) -> BaseModel:
    """The typed view of a payload; raises `pydantic.ValidationError` when it is malformed."""
    return MODELS[plugin].model_validate(payload)
