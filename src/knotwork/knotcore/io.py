"""Bundle loader: rasters + declarative JSON only (SPEC 7, 3).

Invariant: nothing here interprets plugin geometry; it only decodes files and validates schemas.
Kernel primitives signal a failed check by raising `KernelFault(code)`; `engine` converts every
fault into an `ErrorCode` on the certificate (errors are values at the kernel boundary).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image
from pydantic import ValidationError

from knotwork.knotcore import contracts as C

BoolArr = npt.NDArray[np.bool_]


class KernelFault(Exception):
    """A kernel check failed with `code`; converted to a certificate code by the engine."""

    def __init__(self, code: C.ErrorCode, detail: str = "") -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code


@dataclass
class Bundle:
    root: Path
    manifest: C.RenderManifest
    claims: C.ClaimSet
    beauty: npt.NDArray[np.uint8]
    lum: npt.NDArray[np.float64]
    ids: npt.NDArray[np.uint16]
    masks: dict[int, BoolArr]
    beauty768: npt.NDArray[np.uint8]
    memo: dict[str, Any] = field(default_factory=dict)

    def terminal(self, tid: str) -> C.TerminalEntry:
        for t in self.manifest.terminals:
            if t.terminal_id == tid:
                return t
        raise KernelFault(C.ErrorCode.E_TRACE_NO_TERMINAL, f"unknown terminal {tid}")

    def kind(self, kind: str) -> list[int]:
        return [o.obj_id for o in self.manifest.objects if o.kind.value == kind]


def sl(box: C.PixelBox, pad: int = 0) -> tuple[slice, slice]:
    """Numpy (row, col) slices of a pixel box grown by `pad`, clipped to the canvas."""
    y0, x0 = max(box.y0 - pad, 0), max(box.x0 - pad, 0)
    return slice(y0, max(box.y1 + pad, y0)), slice(x0, max(box.x1 + pad, x0))


def load_bundle(d: Path) -> Bundle:
    """Decode one bundle directory; raises KernelFault(E_SCHEMA) on invalid JSON or rasters."""
    try:
        man = C.RenderManifest.model_validate(json.loads((d / "manifest.json").read_text("utf-8")))
        claims = C.ClaimSet.model_validate(json.loads((d / "claims.json").read_text("utf-8")))
        beauty = np.asarray(Image.open(d / "beauty.png").convert("RGB"), dtype=np.uint8)
        raw = np.asarray(Image.open(d / "id_buffer.png"), dtype=np.int64)
        masks = {o.obj_id: np.asarray(Image.open(d / o.mask)) > 0 for o in man.objects}
    except (OSError, ValueError, ValidationError) as exc:
        raise KernelFault(C.ErrorCode.E_SCHEMA, str(exc)[:300]) from None
    shape = (C.CANVAS_PX, C.CANVAS_PX)
    bad = [k for k, m in masks.items() if m.shape != shape]
    if beauty.shape != (*shape, 3) or raw.shape != shape or bad or raw.max() > 65535:
        raise KernelFault(C.ErrorCode.E_SCHEMA, f"raster shapes invalid (masks {bad})")
    lum = beauty.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722]) / 255.0
    small = Image.fromarray(beauty).resize((C.CONSUMPTION_PX,) * 2, Image.Resampling.LANCZOS)
    b768 = np.asarray(small, dtype=np.uint8)
    return Bundle(d, man, claims, beauty, lum, raw.astype(np.uint16), masks, b768)
