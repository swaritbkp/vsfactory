"""Fault proof pairs: one scene rendered with and without an injected fault, both certified.

A committed default pair under `samples/fault_proof/` lets the Forensics page show a proof the
moment it opens; any other fault and seed is built on demand (and cached by the page).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from knotwork.knotcore.contracts import FAULT_EXPECTED, FAULT_PLUGIN, FaultCode
from knotwork.render.canvas import REPO_ROOT
from knotwork.studio import factory

SAMPLE_DIR = REPO_ROOT / "samples" / "fault_proof"
DEFAULT_FAULT = FaultCode.M3  # two end labels swapped: the kernel's OCR cross-check catches it
DEFAULT_SEED = 3
SAMPLE_PX = 640
LABELS = ("Clean control", "Faulty")


@dataclass(frozen=True)
class ProofSide:
    """One half of a proof strip, ready to draw (no paths into a temporary run folder)."""

    label: str
    verdict: str
    codes: tuple[str, ...]
    image: Image.Image


def expected_code(fault: FaultCode) -> str:
    return FAULT_EXPECTED[fault].value


def caught(fault: FaultCode, side: ProofSide) -> bool:
    return side.label == LABELS[1] and expected_code(fault) in side.codes


def build(fault: FaultCode, seed: int, out: Path) -> list[ProofSide]:
    """Render the faulty item first, then the same attempt without the fault as the clean control."""
    run = out / uuid.uuid4().hex[:8]
    plugin = FAULT_PLUGIN[fault]
    for family in factory.families_of(plugin):
        try:
            bad = factory.generate_one(plugin, family, seed, run / "faulty", fault=fault)
        except RuntimeError:
            continue
        good = factory.generate_one(
            plugin, family, seed, run / "clean", first_attempt=bad.attempts - 1
        )
        sides = []
        for label, made in zip(LABELS, (good, bad), strict=True):
            cert = made.candidate.certificate.model_dump(mode="json")
            overlay = factory.witness_overlay(made.candidate.dir, cert)
            sides.append(
                ProofSide(label, str(cert["verdict"]), tuple(cert["codes"]), _small(overlay))
            )
        return sides
    raise RuntimeError(f"no scene admits {fault.value} at seed {seed}")


def _small(img: Image.Image) -> Image.Image:
    return img.resize((SAMPLE_PX, SAMPLE_PX), Image.Resampling.LANCZOS)


def save_sample(fault: FaultCode, seed: int, sides: list[ProofSide], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"fault": fault.value, "seed": seed, "sides": []}
    for k, side in enumerate(sides):
        name = f"{k}_{side.label.split()[0].lower()}.png"
        side.image.save(folder / name, optimize=True)
        meta["sides"].append(
            {"label": side.label, "verdict": side.verdict, "codes": list(side.codes), "image": name}
        )
    (folder / "proof.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def load_sample(folder: Path = SAMPLE_DIR) -> tuple[FaultCode, int, list[ProofSide]] | None:
    """The committed default pair, or None when it is missing or incomplete."""
    meta_path = folder / "proof.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sides = []
    for entry in meta["sides"]:
        path = folder / entry["image"]
        if not path.is_file():
            return None
        with Image.open(path) as im:
            sides.append(
                ProofSide(
                    entry["label"], entry["verdict"], tuple(entry["codes"]), im.convert("RGB")
                )
            )
    return FaultCode(meta["fault"]), int(meta["seed"]), sides
