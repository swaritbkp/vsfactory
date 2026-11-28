"""Interactive generation: one certified item at a time, overlays, fault proofs and batch commands."""

from __future__ import annotations

import inspect
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from knotwork.freshness.beacon import offline
from knotwork.freshness.receipt import build_receipt
from knotwork.knotcore.contracts import (
    CONSUMPTION_PX,
    DifficultySpec,
    Family,
    FaultCode,
    RenderManifest,
    Split,
    TaskBundle,
    TwinRefs,
)
from knotwork.knotcore.cvd import simulate
from knotwork.knotcore.seeds import SeedTree, json_digest
from knotwork.pipeline.certify import Candidate, run_candidate
from knotwork.pipeline.generate import load_profile
from knotwork.pipeline.io import read_json, write_task
from knotwork.plugins.registry import available, fault_effective, get_plugin
from knotwork.render.canvas import REPO_ROOT

PROFILE_PATH = REPO_ROOT / "configs" / "default_profile.yaml"
ATTEMPTS = 8
OVERLAY_ALPHA = 0.55


@dataclass
class Generated:
    candidate: Candidate
    fault: FaultCode | None
    attempts: int
    task: dict[str, Any] | None = None  # the written task.json of a certified, clean item


def profile() -> dict[str, Any]:
    return load_profile(PROFILE_PATH)


def plugin_names() -> list[str]:
    return list(available())


def families_of(plugin: str) -> list[str]:
    return [f.value for f in get_plugin(plugin).families]


def batch_ready(plugin: str) -> bool:
    """Batch generation needs a profile section with family weights."""
    return "families" in profile().get(plugin, {})


def difficulty_from(dials: dict[str, Any]) -> DifficultySpec | None:
    """A spec from UI values, ignoring the dials left on automatic (None)."""
    chosen = {k: v for k, v in dials.items() if v is not None}
    return DifficultySpec.model_validate(chosen) if chosen else None


def generate_one(
    plugin_name: str,
    family: str,
    seed: int,
    out: Path,
    difficulty: DifficultySpec | None = None,
    fault: FaultCode | None = None,
    certify_attempts: int = 1,
    first_attempt: int = 0,
) -> Generated:
    """Sample, render and certify one item. Resamples (up to `ATTEMPTS`) while the plugin finds no
    legible scene or the requested fault would have no effect, and up to `certify_attempts` times
    while the kernel quarantines a clean item (the pipeline retries slots the same way). The last
    candidate is returned either way; a certified clean one is also written as `task.json`."""
    plugin = get_plugin(plugin_name)
    prof = dict(profile(), _family=family)
    takes_dials = "difficulty" in inspect.signature(plugin.sample_scene).parameters
    last = "no attempt made"
    certified_runs = 0
    result: Generated | None = None
    for attempt in range(first_attempt, ATTEMPTS):
        at = SeedTree(seed).child("studio", plugin_name, family, attempt)
        try:
            kwargs = {"difficulty": difficulty} if takes_dials and difficulty else {}
            scene = plugin.sample_scene(at.child("scene"), prof, **kwargs)
            question = plugin.questions(scene, at.child("question"), Family(family))
        except RuntimeError as exc:
            last = str(exc)
            continue
        if fault is not None and not fault_effective(plugin_name, scene, fault):
            last = f"{fault.value} has no effect on the sampled scenes"
            continue
        directory = out / f"a{attempt}"
        candidate = run_candidate(plugin, scene, question, directory, prof, fault)
        certified_runs += 1
        result = Generated(candidate, fault, attempt + 1)
        if candidate.certified or fault is not None or certified_runs >= certify_attempts:
            break
    if result is None:
        raise RuntimeError(f"no item after {ATTEMPTS} attempts: {last}")
    if result.candidate.certified and fault is None:
        result.task = _write_task(result.candidate, seed, prof)
    return result


def _write_task(candidate: Candidate, seed: int, prof: dict[str, Any]) -> dict[str, Any]:
    """Make a certified item a complete bundle (`knot verify` accepts it)."""
    receipt = build_receipt(offline(seed), prof)
    task = TaskBundle(
        bundle_id=candidate.bundle_id, plugin=candidate.scene.plugin,
        family=candidate.question.family, split=Split.DEV, seed_path=candidate.scene.seed_path,
        scene_id=json_digest(candidate.scene.model_dump(mode="json")), question=candidate.question,
        answer=candidate.answer, certificate=candidate.certificate,
        freshness_receipt_id=receipt.receipt_id, twins=TwinRefs(role="base"),
        canary=receipt.canary, files=candidate.files,
    )  # fmt: skip
    write_task(candidate.dir, task)
    return task.model_dump(mode="json")


# ------------------------------------------------------------------ views of a bundle directory
def beauty(bundle: Path, size: int | None = None) -> Image.Image:
    path = bundle / "beauty.png"
    path = path if path.is_file() else bundle / "image.jpg"
    with Image.open(path) as im:
        rgb = im.convert("RGB")
    return rgb.resize((size, size), Image.Resampling.LANCZOS) if size else rgb


def manifest(bundle: Path) -> RenderManifest:
    return RenderManifest.model_validate(read_json(bundle / "manifest.json"))


def _palette(ids: np.ndarray) -> np.ndarray:
    """A stable bright colour per object id; background stays black."""
    r = (ids * 97 + 53) % 206 + 50
    g = (ids * 57 + 131) % 206 + 50
    b = (ids * 163 + 11) % 206 + 50
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    rgb[ids == 0] = 0
    return rgb


def id_buffer_image(bundle: Path) -> Image.Image:
    ids = np.asarray(Image.open(bundle / "id_buffer.png"), dtype=np.int64)
    return Image.fromarray(_palette(ids))


def mask_overlay(bundle: Path, obj_id: int) -> Image.Image:
    """The beauty image with one object's amodal mask tinted magenta."""
    base = np.asarray(beauty(bundle), dtype=np.float64)
    mask = np.asarray(Image.open(bundle / manifest(bundle).obj(obj_id).mask)) > 0
    base[mask] = (1 - OVERLAY_ALPHA) * base[mask] + OVERLAY_ALPHA * np.array([255.0, 0.0, 255.0])
    return Image.fromarray(base.astype(np.uint8))


WITNESS_AXIS = (236, 0, 140)  # needle principal axis: high-visibility magenta
WITNESS_DIAL = (0, 170, 220)  # dial centre crosshair and rim: cyan
WITNESS_OCR = (245, 158, 11)  # OCR'd tick labels: amber
WITNESS_HALO = (255, 255, 255)


def _gauge_witness(draw: ImageDraw.ImageDraw, read: dict[str, Any]) -> None:
    """Dial crosshair, the needle's principal axis extended across the dial, and its tip."""
    cx, cy = float(read["center"][0]), float(read["center"][1])
    radius = float(read.get("R", 0.0)) or 120.0
    reach = radius * 1.12
    for color, width in ((WITNESS_HALO, 5), (WITNESS_DIAL, 2)):
        draw.line([cx - reach, cy, cx + reach, cy], fill=color, width=width)
        draw.line([cx, cy - reach, cx, cy + reach], fill=color, width=width)
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius], outline=WITNESS_DIAL, width=2
    )
    tx, ty = float(read["tip"][0]), float(read["tip"][1])
    length = max(float(np.hypot(tx - cx, ty - cy)), 1.0)
    ux, uy = (tx - cx) / length, (ty - cy) / length
    axis = [cx - ux * radius * 0.3, cy - uy * radius * 0.3, cx + ux * reach, cy + uy * reach]
    draw.line(axis, fill=WITNESS_HALO, width=10)
    draw.line(axis, fill=WITNESS_AXIS, width=5)
    for x, y, r in ((cx, cy, 9), (tx, ty, 11)):
        draw.ellipse([x - r, y - r, x + r, y + r], outline=WITNESS_HALO, width=6)
        draw.ellipse([x - r, y - r, x + r, y + r], outline=WITNESS_AXIS, width=3)


def _ocr_boxes(draw: ImageDraw.ImageDraw, man: RenderManifest) -> None:
    """Amber boxes around every tick label the kernel reads with OCR."""
    for gauge in man.gauges:
        for tick in gauge.ticks:
            label = man.obj(tick.label_obj)
            if label.box is None:
                continue
            b = label.box
            draw.rectangle([b.x0 - 3, b.y0 - 3, b.x1 + 2, b.y1 + 2], outline=WITNESS_HALO, width=5)
            draw.rectangle([b.x0 - 3, b.y0 - 3, b.x1 + 2, b.y1 + 2], outline=WITNESS_OCR, width=3)


def witness_overlay(bundle: Path, certificate: dict[str, Any]) -> Image.Image:
    """Kernel evidence drawn on the beauty: crossing centres (cyan), terminals (green), for gauges
    the dial crosshair (cyan), the extended needle axis (magenta) and OCR'd tick labels (amber boxes), and
    the objects of failed checks (red boxes)."""
    img = beauty(bundle).copy()
    draw = ImageDraw.Draw(img)
    man = manifest(bundle)

    def ring(xy: Any, r: float, color: tuple[int, int, int]) -> None:
        x, y = float(xy[0]), float(xy[1])
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)

    for t in man.terminals:
        draw.rectangle([t.box.x0, t.box.y0, t.box.x1, t.box.y1], outline=(0, 150, 60), width=2)
    _ocr_boxes(draw, man)
    for step in certificate.get("steps", []):
        evidence = step.get("evidence", {})
        for c in evidence.get("crossings", []):
            ring(c["center"], 14, (0, 190, 210))
        if "top_obj" in evidence:
            ring(evidence["center"], 18, (0, 190, 210))
        if "tip" in evidence and "center" in evidence:
            _gauge_witness(draw, evidence)
    for check in certificate.get("checks", []):
        if check["passed"] or check.get("obj_id") is None:
            continue
        mask = np.asarray(Image.open(bundle / man.obj(check["obj_id"]).mask)) > 0
        ys, xs = np.nonzero(mask)
        if len(xs):
            box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            draw.rectangle(box, outline=(220, 20, 20), width=3)
            draw.text((box[0] + 4, box[1] + 4), check["code"], fill=(220, 20, 20))
    return img


def cvd_preview(bundle: Path, kind: str) -> Image.Image:
    """The beauty image as seen with `kind` ("protanopia" or "deuteranopia")."""
    return Image.fromarray(simulate(np.asarray(beauty(bundle)), kind).round().astype(np.uint8))


def consumption(bundle: Path) -> Image.Image:
    """What a model sees: the image at the consumption resolution."""
    return beauty(bundle, CONSUMPTION_PX)


# ------------------------------------------------------------------ batch builds
def batch_command(
    plugin: str,
    n: int,
    out: Path,
    seed: int = 0,
    split: str = "dev",
    twins: bool = True,
    workers: int = 0,
    shard_size: int = 0,
    holdout_families: list[str] | None = None,
    holdout_params: list[str] | None = None,
    difficulty: DifficultySpec | None = None,
) -> list[str]:
    """Arguments of the equivalent `knot generate` call (the dials the CLI exposes)."""
    args = ["generate", "--plugin", plugin, "--n", str(n), "--out", str(out), "--seed", str(seed),
            "--split", split, "--twins" if twins else "--no-twins", "--workers", str(workers),
            "--shard-size", str(shard_size)]  # fmt: skip
    for family in holdout_families or []:
        args += ["--holdout-family", family]
    for param in holdout_params or []:
        args += ["--holdout-param", param]
    if difficulty is not None:
        flags = {"reasoning_load": "--reasoning-load", "visual_stress": "--visual-stress",
                 "occlusion_level": "--occlusion-level", "color_regime": "--color-regime"}  # fmt: skip
        for name, flag in flags.items():
            value = getattr(difficulty, name)
            if value is not None:
                args += [flag, str(value)]
    return args


def command_line(args: list[str]) -> str:
    return "uv run knot " + " ".join(shlex.quote(a.replace("\\", "/")) for a in args)


def start_batch(args: list[str], log: Path) -> subprocess.Popen[bytes]:
    """Run `knot generate` in its own process (worker pools do not mix with the UI process)."""
    from knotwork.studio import demo_guard

    env = dict(os.environ)
    if demo_guard.is_public_demo():
        sid = demo_guard.get_demo_session_id()
        if sid is not None:
            env[demo_guard.SESSION_ENV] = sid
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as sink:
        return subprocess.Popen(
            [sys.executable, "-m", "knotwork", *args],
            cwd=REPO_ROOT,
            env=env,
            stdout=sink,
            stderr=subprocess.STDOUT,
        )


def batch_progress(out: Path) -> int:
    """Base items written so far."""
    index = out / "index.jsonl"
    if not index.is_file():
        return 0
    return sum('"role": "base"' in line for line in index.read_text(encoding="utf-8").splitlines())
