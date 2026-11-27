"""Deterministic grounded visual chain-of-thought (vCoT) traces from certified bundles.

A trace is derived only from files the bundle already carries: the render manifest (terminal and
label boxes, gauge structure), the amodal object masks, the modal id buffer and the kernel
certificate's step evidence (`task.json`). Crossings are found with the trusted kernel's own
primitives (`knotcore.topo.crossings`: pairwise mask overlaps in geodesic order from the start box;
`knotcore.topo.core_vote`: the id-buffer owner of each crossing core, i.e. over / under). No model
and no network is involved, so the trace cannot hallucinate: every box is the pixel extent of a
real object, and the final answer is the certified one.

Invariants:
* every step carries a non-empty pixel box inside the canvas and its [0, 1000] normalisation;
* step 1 is a perception step, the last step is the conclusion and states the certified answer;
* the same bundle always yields the same trace (no randomness, sorted iteration only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field

from knotwork.knotcore import contracts as C
from knotwork.knotcore import topo
from knotwork.knotcore.io import BoolArr, Bundle, KernelFault

Phase = Literal["perception", "transition", "conclusion"]
Box = tuple[int, int, int, int]
#: at most this many crossing hops are narrated for a TRACE (evenly spaced along the path)
MAX_HOPS = 4
#: at most this many crossings are narrated one by one for a CROSS count
MAX_CROSS_STEPS = 8
NORM = 1000
#: a terminal's label joins its grounding box only when printed within this many pixels of it
LABEL_NEAR_PX = 48
EMERALD = (16, 185, 129)
EMERALD_DARK = (4, 120, 87)
#: step badge radius in 1024-px canvas pixels (legible after the Studio's 560-px downscale)
BADGE_R = 24
FONT_PATH = Path(__file__).resolve().parents[3] / "assets" / "fonts" / "DejaVuSans.ttf"


class VCoTStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_idx: int = Field(ge=1)
    phase: Phase
    evidence_label: str
    bbox_px: list[int] = Field(min_length=4, max_length=4)
    bbox_norm_1000: list[int] = Field(min_length=4, max_length=4)
    text: str


class VCoTTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    domain: str
    family: str
    image_size: list[int] = Field(min_length=2, max_length=2)
    steps: list[VCoTStep] = Field(min_length=1)
    final_answer: str
    formatted_cot: str

    @property
    def boxes_1000(self) -> list[list[int]]:
        return [s.bbox_norm_1000 for s in self.steps]


class VCoTError(ValueError):
    """The bundle lacks what a grounded trace needs (no certificate steps, unknown objects)."""


# ------------------------------------------------------------------ bundle access
class _LazyMasks(dict[int, BoolArr]):
    """Masks decoded on first use: a trace touches a handful of the bundle's objects."""

    def __init__(self, root: Path, manifest: C.RenderManifest) -> None:
        super().__init__()
        self._files = {o.obj_id: root / o.mask for o in manifest.objects}

    def __missing__(self, oid: int) -> BoolArr:
        if oid not in self._files:
            raise KeyError(oid)
        mask = np.asarray(Image.open(self._files[oid]), dtype=np.uint8) > 0
        self[oid] = mask
        return mask


def _bundle(root: Path) -> Bundle:
    """A kernel `Bundle` view for the topology primitives (they read masks and the id buffer only,
    so the beauty raster is not decoded)."""
    man = C.RenderManifest.model_validate(json.loads((root / "manifest.json").read_text("utf-8")))
    claims_path = root / "claims.json"
    claims = C.ClaimSet.model_validate(
        json.loads(claims_path.read_text("utf-8")) if claims_path.is_file() else {"claims": []}
    )
    ids = np.asarray(Image.open(root / "id_buffer.png"), dtype=np.int64).astype(np.uint16)
    empty = np.zeros((1, 1, 3), dtype=np.uint8)
    return Bundle(root, man, claims, empty, np.zeros((1, 1)), ids, _LazyMasks(root, man), empty)


@dataclass
class _Ctx:
    b: Bundle
    task: dict[str, Any]
    size: tuple[int, int]
    steps: list[tuple[Phase, str, Box, str]] = field(default_factory=list)
    port_gauge: dict[str, str] = field(default_factory=dict)  # source label -> gauge id

    def add(self, phase: Phase, label: str, box: Box | None, text: str) -> None:
        if box is None:
            raise VCoTError(f"no pixel evidence for step '{label}'")
        self.steps.append((phase, label, self.clip(box), text))

    def clip(self, box: Box) -> Box:
        w, h = self.size
        x0, y0 = min(max(box[0], 0), w - 1), min(max(box[1], 0), h - 1)
        return x0, y0, max(min(box[2], w), x0 + 1), max(min(box[3], h), y0 + 1)


def _mask_box(mask: BoolArr) -> Box | None:
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not rows.size:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _obj_box(b: Bundle, oid: int | None) -> Box | None:
    if oid is None:
        return None
    entry = b.manifest.obj(oid)
    if entry.box is not None:
        return entry.box.x0, entry.box.y0, entry.box.x1, entry.box.y1
    return _mask_box(b.masks[oid])


def _union(*boxes: Box | None) -> Box | None:
    real = [x for x in boxes if x is not None]
    if not real:
        return None
    return (min(x[0] for x in real), min(x[1] for x in real),
            max(x[2] for x in real), max(x[3] for x in real))  # fmt: skip


def _pad(box: Box | None, p: float) -> Box | None:
    if box is None:
        return None
    q = int(round(p))
    return box[0] - q, box[1] - q, box[2] + q, box[3] + q


def _point_box(xy: Any, r: float) -> Box:
    x, y = float(xy[0]), float(xy[1])
    return int(x - r), int(y - r), int(x + r) + 1, int(y + r) + 1


def normalise(box: Box, size: tuple[int, int]) -> list[int]:
    """Pixel box -> [x1, y1, x2, y2] on the 0-1000 grid used by Qwen-VL style grounding."""
    w, h = size
    scale = (w, h, w, h)
    return [min(NORM, max(0, int(round(v * NORM / s)))) for v, s in zip(box, scale, strict=True)]


# ------------------------------------------------------------------ helpers per object type
def _terminal(ctx: _Ctx, tid: str) -> tuple[Box, str]:
    t = ctx.b.terminal(tid)
    label_box = _obj_box(ctx.b, t.label_obj)
    text = ctx.b.manifest.obj(t.label_obj).label_text if t.label_obj else None
    own: Box = (t.box.x0, t.box.y0, t.box.x1, t.box.y1)
    # a label printed next to the socket joins its box; one printed elsewhere (a panel port is
    # named by its gauge's caption) would only blur the grounding, so it is left out
    near = label_box is not None and _gap(own, label_box) <= LABEL_NEAR_PX
    box = _union(own, label_box if near else None)
    assert box is not None
    return box, text or tid.split(":")[-1]


def _gap(a: Box, b: Box) -> int:
    """Chebyshev gap between two boxes (0 when they touch or overlap)."""
    return max(b[0] - a[2], a[0] - b[2], b[1] - a[3], a[1] - b[3], 0)


def _strand_name(b: Bundle, oid: int) -> str:
    return b.manifest.obj(oid).name.split(":")[-1]


def _crossing_box(b: Bundle, c: dict[str, Any], strand: int) -> Box:
    xs, ys = np.asarray(c["xs"]), np.asarray(c["ys"])
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    padded = _pad(box, max(4.0, 0.5 * topo.stroke(b, strand)))
    assert padded is not None
    return padded


def _over_under(b: Bundle, c: dict[str, Any], strand: int) -> str:
    top = int(topo.core_vote(b, c, strand, int(c["other"]))["top_obj"])
    if top == strand:
        return "over"
    return "under" if top == int(c["other"]) else "past (occluded at)"


def _evenly(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    return sorted({int(round(i * (n - 1) / (k - 1))) for i in range(k)})


def _gauge(b: Bundle, gid: str) -> C.GaugeEntry:
    for g in b.manifest.gauges:
        if g.gauge_id == gid:
            return g
    raise VCoTError(f"unknown gauge {gid}")


def _gauge_of_port(b: Bundle, port: str) -> C.GaugeEntry | None:
    return next((g for g in b.manifest.gauges if g.port_terminal == port), None)


def _end_terminal(b: Bundle, start: str) -> str | None:
    """The end / port terminal reached by the start strand's component (kernel TRACE, no OCR)."""
    try:
        _, comp = topo.trace_component(b, start)
    except KernelFault:
        return None
    for t in b.manifest.terminals:
        box = t.box
        reached = comp[box.y0 : box.y1, box.x0 : box.x1].any()
        if t.terminal_id != start and t.role in ("end", "port") and reached:
            return t.terminal_id
    return None


def _fmt(x: float) -> str:
    return f"{x:.1f}".removesuffix(".0")


# ------------------------------------------------------------------ op narrators
def _trace(ctx: _Ctx, step: dict[str, Any]) -> None:
    b, start = ctx.b, str(step["inputs"][0])
    box, label = _terminal(ctx, start)
    ctx.add("perception", f"start terminal {label}", box,
            f"Locate the start terminal labelled {label}; the traced cable leaves it here.")  # fmt: skip
    try:
        strand = topo.strand_of(b, start)
        cs = topo.crossings(b, start, "*")
    except KernelFault:
        strand, cs = 0, []
    picked = _evenly(len(cs), MAX_HOPS)
    for hop, i in enumerate(picked, 1):
        c = cs[i]
        where = _over_under(b, c, strand)
        other = _strand_name(b, int(c["other"]))
        ctx.add("transition", f"crossing {i + 1}/{len(cs)}", _crossing_box(b, c, strand),
                f"Hop {hop}: at crossing {i + 1} of {len(cs)} along the path the cable passes "
                f"{where} strand {other}; keep following the same cable.")  # fmt: skip
    end = str(step["evidence"].get("end_terminal") or "")
    if not end:
        raise VCoTError("TRACE evidence names no end terminal")
    end_box, end_label = _terminal(ctx, end)
    shown = str(step["evidence"].get("ocr_text") or end_label)
    kind = "port" if b.terminal(end).role == "port" else "terminal"
    ctx.add("transition", f"end {kind} {shown}", end_box,
            f"The cable ends at the {kind} labelled {shown}.")  # fmt: skip


def _cross(ctx: _Ctx, step: dict[str, Any]) -> None:
    b, start, other = ctx.b, str(step["inputs"][0]), str(step["inputs"][1])
    box, label = _terminal(ctx, start)
    ctx.add("perception", f"start terminal {label}", box,
            f"Locate the cable that starts at terminal {label}.")  # fmt: skip
    strand = topo.strand_of(b, start)
    cs = topo.crossings(b, start, other)
    boxes = [_crossing_box(b, c, strand) for c in cs]
    for i in _evenly(len(cs), MAX_CROSS_STEPS):
        ctx.add("transition", f"crossing {i + 1}/{len(cs)}", boxes[i],
                f"Crossing {i + 1}: the cable passes {_over_under(b, cs[i], strand)} strand "
                f"{_strand_name(b, int(cs[i]['other']))}.")  # fmt: skip
    total = _union(*boxes) or box
    ctx.add("transition", "crossing count", total,
            f"Counting every crossing along the cable gives {len(cs)}.")  # fmt: skip


def _ontop(ctx: _Ctx, step: dict[str, Any]) -> None:
    b = ctx.b
    a_id, b_id, idx = str(step["inputs"][0]), str(step["inputs"][1]), int(step["inputs"][2])
    for tid in (a_id, b_id):
        box, label = _terminal(ctx, tid)
        ctx.add("perception", f"start terminal {label}", box,
                f"Locate the cable that starts at terminal {label}.")  # fmt: skip
    cs = topo.crossings(b, a_id, b_id)
    strand = topo.strand_of(b, a_id)
    if 0 <= idx < len(cs):
        cbox: Box | None = _crossing_box(b, cs[idx], strand)
    else:
        cbox = _point_box(step["evidence"]["center"], 12)
    top = int(step["evidence"].get("top_obj", 0))
    owner = _strand_name(b, top) if top else "neither strand"
    ctx.add("transition", f"crossing {idx + 1}", cbox,
            f"At crossing {idx + 1} of the two cables the visible top strand is {owner}.")  # fmt: skip


def _gauge_of(ctx: _Ctx, step: dict[str, Any]) -> None:
    port = str(step["evidence"].get("port") or "")
    g = _gauge_of_port(ctx.b, port) if port else None
    gid = str(step["output"])
    g = g or _gauge(ctx.b, gid)
    ctx.add("transition", f"gauge {g.gauge_id.split(':')[-1]}", _obj_box(ctx.b, g.dial_obj),
            f"Port {step['inputs'][0]} feeds gauge {g.gauge_id.split(':')[-1].upper()}.")  # fmt: skip


def _tick_values(b: Bundle, g: C.GaugeEntry) -> list[tuple[float, int]]:
    out = []
    for t in g.ticks:
        text = b.manifest.obj(t.label_obj).label_text or ""
        try:
            out.append((float(text), t.label_obj))
        except ValueError:
            continue
    return sorted(out)


def _read(ctx: _Ctx, step: dict[str, Any]) -> None:
    b, ev = ctx.b, step["evidence"]
    g = _gauge(b, str(step["inputs"][0]))
    name = g.gauge_id.split(":")[-1].upper()
    cx, cy = ev.get("center", g.claimed_center)
    dial = _obj_box(b, g.dial_obj)
    ctx.add("perception", f"dial {name}", dial,
            f"Locate the dial of gauge {name}; its hub (centre) is at ({cx:.0f}, {cy:.0f}).")  # fmt: skip
    ticks = _tick_values(b, g)
    lo, hi = float(ev.get("lo", ticks[0][0])), float(ev.get("hi", ticks[-1][0]))
    bracket = [oid for v, oid in ticks if v in (lo, hi)] or [oid for _, oid in ticks]
    scale = _union(*(_obj_box(b, oid) for oid in bracket))
    span = f"{_fmt(ticks[0][0])} to {_fmt(ticks[-1][0])}" if ticks else "its printed labels"
    ctx.add("perception", f"scale marks {_fmt(lo)}-{_fmt(hi)}", scale,
            f"The scale runs from {span}; the needle lies between the {_fmt(lo)} and "
            f"{_fmt(hi)} marks.")  # fmt: skip
    tip = ev.get("tip")
    needle = _union(_obj_box(b, g.needle_obj), _point_box(tip, 3) if tip else None)
    tip_text = f" to its tip at ({tip[0]:.0f}, {tip[1]:.0f})" if tip else ""
    ctx.add("transition", f"needle {name}", needle,
            f"The needle runs from the hub{tip_text}.")  # fmt: skip
    value = float(step["output"][0])
    near = _union(_point_box(tip, 14) if tip else None, scale)
    ctx.add("transition", f"reading {name}", near,
            f"Interpolating between the {_fmt(lo)} and {_fmt(hi)} marks, gauge {name} reads "
            f"{value:.1f}.")  # fmt: skip


def _zone(ctx: _Ctx, step: dict[str, Any]) -> None:
    b = ctx.b
    g = _gauge(b, str(step["inputs"][0]))
    zone = str(step["output"])
    zobj = next((z.zone_obj for z in g.zones if z.name.value == zone), None)
    box = _obj_box(b, zobj) if zobj else _obj_box(b, g.dial_obj)
    phi = float(step["evidence"].get("phi_needle", 0.0))
    ctx.add("transition", f"{zone} zone", box,
            f"The needle sits {phi:.0f} degrees along the scale, inside the {zone} zone.")  # fmt: skip


def _read_all(ctx: _Ctx, step: dict[str, Any]) -> None:
    b = ctx.b
    boxes, labels = [], []
    for tid in step["inputs"]:
        box, label = _terminal(ctx, str(tid))
        boxes.append(box)
        labels.append(label)
    ctx.add("perception", "source sockets", _union(*boxes),
            f"Locate the source sockets {', '.join(labels)}.")  # fmt: skip
    readings = {str(r[0]): float(r[1]) for r in step["output"]}
    for tid, label in zip(step["inputs"], labels, strict=True):
        end = _end_terminal(b, str(tid))
        g = _gauge_of_port(b, end) if end else None
        if g is None:
            continue
        ctx.port_gauge[label] = g.gauge_id
        value = readings.get(label)
        reads = f", which reads {value:.1f}" if value is not None else ""
        ctx.add("transition", f"{label} -> {g.gauge_id.split(':')[-1]}", _obj_box(b, g.dial_obj),
                f"Socket {label} is wired to gauge {g.gauge_id.split(':')[-1].upper()}{reads}.")  # fmt: skip


def _argmax(ctx: _Ctx, step: dict[str, Any]) -> None:
    ranked = step["evidence"].get("ranked") or []
    win = str(step["output"])
    gid = ctx.port_gauge.get(win)
    box = _obj_box(ctx.b, _gauge(ctx.b, gid).dial_obj) if gid else None
    order = " > ".join(f"{r[0]} ({float(r[1]):.1f})" for r in ranked)
    ctx.add("transition", f"highest {win}", box or (ctx.steps[-1][2] if ctx.steps else None),
            f"Ranking the readings {order}, socket {win} drives the highest gauge.")  # fmt: skip


def _count(ctx: _Ctx, step: dict[str, Any]) -> None:
    b = ctx.b
    objs = [int(o) for o in step["evidence"].get("objects") or [] if isinstance(o, int)]
    target = str(step["inputs"][1]).split(":")[-1] if len(step["inputs"]) > 1 else "target"
    box = _union(*(_obj_box(b, o) for o in objs))
    if box is None:
        box = _union(*(_obj_box(b, g.dial_obj) for g in b.manifest.gauges))
    ctx.add("transition", f"count {target}", box,
            f"Checking every needle against its {target} zone gives {step['output']}.")  # fmt: skip


NARRATORS = {
    "TRACE": _trace, "CROSS": _cross, "ONTOP": _ontop, "GAUGE_OF": _gauge_of, "READ": _read,
    "ZONE": _zone, "READ_ALL": _read_all, "ARGMAX": _argmax, "COUNT": _count,
}  # fmt: skip


# ------------------------------------------------------------------ public API
def answer_text(answer: dict[str, Any]) -> str:
    return str(answer.get("canonical") or answer.get("value") or "")


def format_cot(steps: list[VCoTStep], final_answer: str) -> str:
    """`<think>` with one grounded line per step, then `\\boxed{answer}` (the SFT target)."""
    lines = [
        f"Step {s.step_idx} ({s.phase}): {s.text} <box>[{', '.join(map(str, s.bbox_norm_1000))}]"
        "</box>"
        for s in steps
    ]
    return "<think>\n" + "\n".join(lines) + "\n</think>\n\\boxed{" + final_answer + "}"


def extract(bundle_dir: Path, task: dict[str, Any] | None = None) -> VCoTTrace:
    """The grounded trace of one certified bundle (raises VCoTError when it cannot be grounded)."""
    root = Path(bundle_dir)
    task = task if task is not None else json.loads((root / "task.json").read_text("utf-8"))
    cert = task.get("certificate") or {}
    steps = cert.get("steps") or []
    if not steps:
        raise VCoTError("certificate has no kernel steps")
    b = _bundle(root)
    canvas = b.manifest.canvas
    ctx = _Ctx(b, task, (int(canvas[0]), int(canvas[1])))
    for step in steps:
        narrator = NARRATORS.get(str(step["op"]))
        if narrator is None:
            raise VCoTError(f"no narrator for op {step['op']}")
        try:
            narrator(ctx, step)
        except (KernelFault, KeyError, IndexError, TypeError) as exc:
            raise VCoTError(f"{step['op']}: {exc}") from None
    final = answer_text(task.get("answer") or {})
    premise = (task.get("question") or {}).get("premise")
    phase, label, box, text = ctx.steps[-1]
    if premise is not None and isinstance((task.get("answer") or {}).get("value"), dict):
        holds = bool(task["answer"]["value"].get("premise_holds"))
        text += f" The stated premise {'holds' if holds else 'is false'}."
    ctx.steps[-1] = ("conclusion", label, box, f"{text} Final answer: {final}.")
    out = [
        VCoTStep(step_idx=i, phase=ph, evidence_label=lab, bbox_px=list(bx),
                 bbox_norm_1000=normalise(bx, ctx.size), text=tx)
        for i, (ph, lab, bx, tx) in enumerate(ctx.steps, 1)
    ]  # fmt: skip
    return VCoTTrace(
        item_id=str(task.get("bundle_id") or root.name), domain=str(task.get("plugin") or ""),
        family=str(task.get("family") or ""), image_size=list(ctx.size), steps=out,
        final_answer=final, formatted_cot=format_cot(out, final),
    )  # fmt: skip


def try_extract(bundle_dir: Path, task: dict[str, Any] | None = None) -> VCoTTrace | None:
    try:
        return extract(bundle_dir, task)
    except (VCoTError, OSError, ValueError):
        return None


def _font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(FONT_PATH), px)
    except OSError:
        return ImageFont.load_default()


def overlay(image: Image.Image, trace: VCoTTrace) -> Image.Image:
    """Numbered emerald step badges and boxes over the image (any size: boxes are rescaled)."""
    img = image.convert("RGB").copy()
    sx, sy = img.width / trace.image_size[0], img.height / trace.image_size[1]
    draw = ImageDraw.Draw(img)
    r = max(10, int(round(BADGE_R * sx)))
    font = _font(int(r * 1.3))
    line = max(2, int(round(5 * sx)))
    for s in trace.steps:
        x0, y0, x1, y1 = (
            s.bbox_px[0] * sx,
            s.bbox_px[1] * sy,
            s.bbox_px[2] * sx,
            s.bbox_px[3] * sy,
        )
        color = EMERALD_DARK if s.phase == "conclusion" else EMERALD
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line)
        # the badge sits just outside the box's top-left corner so it never hides a small crossing
        cx = min(max(x0 - 0.7 * r, r + 1), img.width - r - 1)
        cy = min(max(y0 - 0.7 * r, r + 1), img.height - r - 1)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(255, 255, 255),
                     width=max(2, line // 2))  # fmt: skip
        draw.text((cx, cy), str(s.step_idx), fill=(255, 255, 255), font=font, anchor="mm")
    return img


__all__ = [
    "VCoTError",
    "VCoTStep",
    "VCoTTrace",
    "extract",
    "format_cot",
    "normalise",
    "overlay",
    "try_extract",
]
