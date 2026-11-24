"""Deterministic layered 2D renderer shared by all plugins (SPEC.md section 6).

Three layers are kept in lock-step:

* **beauty** -- RGB, drawn at ``size * supersample`` and Lanczos-downsampled to ``size`` (anti-aliased).
* **masks** -- one ``L`` image per object at ``size``, values {0, 255}, no anti-aliasing, AMODAL.
* **id buffer** -- 32-bit work image at ``size`` (saved as uint16), no anti-aliasing, MODAL: it must
  match what is visible in the beauty layer (halo pixels of an under-wire are 0).

All coordinates are in ``size``-pixel space; the canvas scales beauty drawing internally. The renderer
is an untrusted helper: the kernel re-derives everything from the files it writes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

from knotwork.knotcore.contracts import (
    GaugeEntry,
    Kind,
    ObjectEntry,
    PixelBox,
    RenderManifest,
    TerminalEntry,
)

RGB = tuple[int, int, int]
Shade = Literal["flat", "harness"]
FloatArray = npt.NDArray[np.float64]


class RenderError(RuntimeError):
    """Raised for unrecoverable render problems; the message starts with an ``E_*`` code."""


def font_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _shade_colors(rgb: RGB) -> tuple[RGB, RGB]:
    """(darker rim, lighter core) colours for harness shading; both stay inside the stroke."""
    rim = tuple(int(c * 0.62) for c in rgb)
    core = tuple(int(c + (255 - c) * 0.22) for c in rgb)
    return (rim[0], rim[1], rim[2]), (core[0], core[1], core[2])


@dataclass
class _Obj:
    entry: dict[str, Any]
    mask: Image.Image
    style: dict[str, Any] = field(default_factory=dict)


def as_points(pts: Any) -> FloatArray:
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 1:
        raise ValueError("points must be an (N, 2) array")
    return arr


def _stroke(
    draw: ImageDraw.ImageDraw,
    pts: FloatArray,
    width: float,
    fill: Any,
    scale: float,
    snap: bool,
    caps: bool = True,
) -> None:
    """Exact-width polyline = polygon strip per segment + round joins (disks at vertices).

    ``caps=False`` gives flat (butt) ends: only interior vertices get a round join."""
    p = pts * scale
    # PIL fills pixels whose centres lie on the polygon edge, so 1x layers use (w - 1) / 2 to cover
    # exactly ~w pixels (matching the anti-aliased beauty stroke footprint).
    r = width * scale / 2.0 if not snap else max(width - 1.0, 0.0) / 2.0

    def fix(xy: FloatArray) -> list[tuple[float, float]]:
        if snap:
            return [(float(round(x)), float(round(y))) for x, y in xy]
        return [(float(x), float(y)) for x, y in xy]

    for a, b in zip(p[:-1], p[1:], strict=True):
        d = b - a
        n = float(np.hypot(d[0], d[1]))
        if n < 1e-9:
            continue
        nv = np.array([-d[1], d[0]]) / n * r
        draw.polygon(fix(np.array([a + nv, b + nv, b - nv, a - nv])), fill=fill)
    for x, y in p if caps else p[1:-1]:
        box = np.array([[x - r, y - r], [x + r, y + r]])
        q = fix(box)
        if q[1][0] >= q[0][0] and q[1][1] >= q[0][1]:
            draw.ellipse([q[0], q[1]], fill=fill)


def arc_band_points(
    center: tuple[float, float], r0: float, r1: float, start_deg: float, end_deg: float
) -> FloatArray:
    """Polygon of an annular band between math-convention angles (CCW from +x, y up)."""
    lo, hi = sorted((start_deg, end_deg))
    n = max(8, int((hi - lo) / 1.0) + 2)
    th = np.radians(np.linspace(lo, hi, n))
    cx, cy = center
    outer = np.stack([cx + r1 * np.cos(th), cy - r1 * np.sin(th)], axis=1)
    inner = np.stack([cx + r0 * np.cos(th[::-1]), cy - r0 * np.sin(th[::-1])], axis=1)
    return np.concatenate([outer, inner], axis=0)


class LayeredCanvas:
    """Beauty / amodal masks / modal id buffer kept consistent by construction."""

    def __init__(
        self,
        size: int = 1024,
        supersample: int = 2,
        background_rgb: RGB = (246, 244, 238),
        font_path: str | Path = "assets/fonts/DejaVuSans.ttf",
        expected_font_sha256: str | None = None,
    ) -> None:
        self.size, self.ss = int(size), int(supersample)
        self.background: RGB = (
            int(background_rgb[0]),
            int(background_rgb[1]),
            int(background_rgb[2]),
        )
        self.font_path = Path(font_path)
        if not self.font_path.is_file():
            raise RenderError(f"E_FONT: font file not found: {self.font_path}")
        self.font_digest = font_sha256(self.font_path)
        if expected_font_sha256 is not None and self.font_digest != expected_font_sha256:
            raise RenderError(
                f"E_FONT: font sha256 {self.font_digest} != profile {expected_font_sha256}"
            )
        big = self.size * self.ss
        self.beauty = Image.new("RGB", (big, big), self.background)
        self._bdraw = ImageDraw.Draw(self.beauty)
        self.idimg = Image.new("I", (self.size, self.size), 0)
        self._idraw = ImageDraw.Draw(self.idimg)
        self._idraw.fontmode = "1"
        self._objs: dict[int, _Obj] = {}
        self._fonts: dict[int, ImageFont.FreeTypeFont] = {}

    # ------------------------------------------------------------------ objects
    def new_object(
        self,
        name: str,
        kind: Kind | str,
        color_rgb: RGB,
        parent: str | None = None,
        label_text: str | None = None,
        box: PixelBox | None = None,
        font_px: int | None = None,
    ) -> int:
        oid = len(self._objs) + 1
        if oid > 65535:
            raise RenderError("E_SCHEMA: more than 65535 objects")
        entry = {
            "obj_id": oid,
            "name": name,
            "kind": Kind(kind),
            "mask": f"masks/obj_{oid:05d}.png",
            "color_rgb": tuple(int(c) for c in color_rgb),
            "parent": parent,
            "label_text": label_text,
            "box": box,
            "font_px": font_px,
        }
        self._objs[oid] = _Obj(entry, Image.new("L", (self.size, self.size), 0))
        return oid

    def object_ids(self) -> list[int]:
        return list(self._objs)

    def set_box(self, oid: int, box: PixelBox) -> None:
        self._objs[oid].entry["box"] = box

    def get_box(self, oid: int) -> PixelBox | None:
        box = self._objs[oid].entry["box"]
        return box if isinstance(box, PixelBox) else None

    def font(self, px: int) -> ImageFont.FreeTypeFont:
        if px not in self._fonts:
            self._fonts[px] = ImageFont.truetype(str(self.font_path), px)
        return self._fonts[px]

    def _color(self, oid: int, beauty_rgb: RGB | None) -> RGB:
        if beauty_rgb is not None:
            return beauty_rgb
        c = self._objs[oid].entry["color_rgb"]
        return (int(c[0]), int(c[1]), int(c[2]))

    def _layers(self, oid: int, mask: bool, idbuf: bool) -> list[tuple[ImageDraw.ImageDraw, Any]]:
        out: list[tuple[ImageDraw.ImageDraw, Any]] = []
        if mask:
            out.append((ImageDraw.Draw(self._objs[oid].mask), 255))
        if idbuf:
            out.append((self._idraw, oid))
        return out

    # ------------------------------------------------------------------ primitives
    def polyline(
        self,
        oid: int,
        pts: Any,
        width_px: float,
        beauty: bool = True,
        mask: bool = True,
        idbuf: bool = True,
        beauty_rgb: RGB | None = None,
        shade: Shade = "flat",
    ) -> None:
        p = as_points(pts)
        rgb = self._color(oid, beauty_rgb)
        self._objs[oid].style = {"rgb": rgb, "shade": shade, "width": float(width_px)}
        if beauty:
            self._beauty_stroke(p, float(width_px), rgb, shade)
        for draw, fill in self._layers(oid, mask, idbuf):
            _stroke(draw, p, float(width_px), fill, 1.0, snap=True)

    def _beauty_stroke(
        self, p: FloatArray, width: float, rgb: RGB, shade: Shade, caps: bool = True
    ) -> None:
        if shade == "harness" and width >= 4.0:
            rim, core = _shade_colors(rgb)
            _stroke(self._bdraw, p, width, rim, self.ss, snap=False, caps=caps)
            _stroke(self._bdraw, p, width - 2.0, rgb, self.ss, snap=False, caps=caps)
            _stroke(self._bdraw, p, max(1.0, width * 0.2), core, self.ss, snap=False, caps=caps)
        else:
            _stroke(self._bdraw, p, width, rgb, self.ss, snap=False, caps=caps)

    def raw_beauty_polyline(self, pts: Any, width_px: float, rgb: RGB) -> None:
        """Ink on the beauty layer only: no object, no mask, no id (fault injection M7)."""
        _stroke(self._bdraw, as_points(pts), float(width_px), rgb, self.ss, snap=False)

    def polygon(
        self,
        oid: int,
        pts: Any,
        beauty: bool = True,
        mask: bool = True,
        idbuf: bool = True,
        beauty_rgb: RGB | None = None,
    ) -> None:
        p = as_points(pts)
        if beauty:
            q = [(float(x) * self.ss, float(y) * self.ss) for x, y in p]
            self._bdraw.polygon(q, fill=self._color(oid, beauty_rgb))
        for draw, fill in self._layers(oid, mask, idbuf):
            draw.polygon([(float(round(x)), float(round(y))) for x, y in p], fill=fill)

    def disk(
        self,
        oid: int,
        center: tuple[float, float],
        radius: float,
        beauty: bool = True,
        mask: bool = True,
        idbuf: bool = True,
        beauty_rgb: RGB | None = None,
        outline_rgb: RGB | None = None,
        outline_px: float = 0.0,
    ) -> None:
        cx, cy = float(center[0]), float(center[1])
        if beauty:
            s = self.ss
            box = [(cx - radius) * s, (cy - radius) * s, (cx + radius) * s, (cy + radius) * s]
            self._bdraw.ellipse(box, fill=self._color(oid, beauty_rgb))
            if outline_rgb is not None and outline_px > 0:
                self._bdraw.ellipse(box, outline=outline_rgb, width=max(1, round(outline_px * s)))
        r = float(radius)
        q = [round(cx - r), round(cy - r), round(cx + r), round(cy + r)]
        for draw, fill in self._layers(oid, mask, idbuf):
            draw.ellipse(q, fill=fill)

    def arc_band(
        self,
        oid: int,
        center: tuple[float, float],
        r0: float,
        r1: float,
        start_deg: float,
        end_deg: float,
        beauty: bool = True,
        mask: bool = True,
        idbuf: bool = True,
        beauty_rgb: RGB | None = None,
    ) -> None:
        pts = arc_band_points(center, r0, r1, start_deg, end_deg)
        self.polygon(oid, pts, beauty=beauty, mask=mask, idbuf=idbuf, beauty_rgb=beauty_rgb)

    def text(
        self,
        oid: int,
        xy: tuple[float, float],
        text: str,
        font_px: int,
        anchor: str = "mm",
        beauty: bool = True,
        mask: bool = True,
        idbuf: bool = True,
        beauty_rgb: RGB | None = None,
        box_pad: int = 4,
    ) -> PixelBox:
        """Draw text; beauty glyphs are AA, mask/id glyphs non-AA. Sets the object box if unset."""
        x, y = float(xy[0]), float(xy[1])
        if beauty:
            self._bdraw.text(
                (x * self.ss, y * self.ss),
                text,
                font=self.font(font_px * self.ss),
                fill=self._color(oid, beauty_rgb),
                anchor=anchor,
            )
        font = self.font(font_px)
        for draw, fill in self._layers(oid, mask, idbuf):
            draw.fontmode = "1"
            draw.text((round(x), round(y)), text, font=font, fill=fill, anchor=anchor)
        lft, t, r, b = self._idraw.textbbox((round(x), round(y)), text, font=font, anchor=anchor)
        box = PixelBox(
            x0=int(lft) - box_pad, y0=int(t) - box_pad, x1=int(r) + box_pad, y1=int(b) + box_pad
        )
        if self._objs[oid].entry["box"] is None:
            self._objs[oid].entry["box"] = box
        if self._objs[oid].entry["font_px"] is None:
            self._objs[oid].entry["font_px"] = int(font_px)
        return box

    def halo_patch(
        self,
        top_oid: int,
        pts: Any,
        width_px: float,
        halo_px: float,
        radius_px: float,
        center: tuple[float, float],
    ) -> None:
        """Per-crossing layering (SPEC 6.1): background halo then top-wire redraw (beauty + id)."""
        p = as_points(pts)
        c = np.asarray(center, dtype=np.float64)
        style = self._objs[top_oid].style
        rgb: RGB = style.get("rgb", self._color(top_oid, None))
        shade: Shade = style.get("shade", "flat")
        run = _sub_run(p, c, radius_px)
        if len(run) < 2:
            return
        # Flat-ended background halo, then the top wire redrawn over exactly the same run: nothing
        # outside `radius_px` of this crossing is touched, so nearby crossings keep their layering.
        halo_w = float(width_px + 2 * halo_px)
        _stroke(self._bdraw, run, halo_w, self.background, self.ss, snap=False, caps=False)
        self._beauty_stroke(run, float(width_px), rgb, shade, caps=False)
        _stroke(self._idraw, run, halo_w, 0, 1.0, snap=True, caps=False)
        _stroke(self._idraw, run, float(width_px), top_oid, 1.0, snap=True)

    # ------------------------------------------------------------------ output
    def beauty_array(self) -> npt.NDArray[np.uint8]:
        small = self.beauty.resize((self.size, self.size), Image.Resampling.LANCZOS)
        return np.asarray(small, dtype=np.uint8)

    def id_array(self) -> npt.NDArray[np.uint16]:
        arr = np.asarray(self.idimg, dtype=np.int64)
        if arr.min() < 0 or arr.max() > 65535:
            raise RenderError("E_SCHEMA: id buffer out of uint16 range")
        return arr.astype(np.uint16)

    def mask_array(self, oid: int) -> npt.NDArray[np.bool_]:
        return np.asarray(self._objs[oid].mask, dtype=np.uint8) > 0

    def manifest(
        self,
        plugin: str,
        terminals: list[TerminalEntry] | None = None,
        gauges: list[GaugeEntry] | None = None,
    ) -> RenderManifest:
        return RenderManifest(
            plugin=plugin,
            canvas=(self.size, self.size),
            supersample=self.ss,
            background_rgb=self.background,
            font_sha256=self.font_digest,
            objects=[ObjectEntry(**o.entry) for o in self._objs.values()],
            terminals=terminals or [],
            gauges=gauges or [],
        )

    def finish(
        self,
        out_dir: str | Path,
        plugin: str,
        terminals: list[TerminalEntry] | None = None,
        gauges: list[GaugeEntry] | None = None,
    ) -> RenderManifest:
        """Write beauty.png, id_buffer.png, masks/obj_<id>.png and manifest.json."""
        out = Path(out_dir)
        (out / "masks").mkdir(parents=True, exist_ok=True)
        man = self.manifest(plugin, terminals, gauges)
        Image.fromarray(self.beauty_array(), mode="RGB").save(out / "beauty.png", optimize=False)
        Image.fromarray(self.id_array()).save(out / "id_buffer.png")
        for oid, obj in self._objs.items():
            m = (np.asarray(obj.mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
            Image.fromarray(m, mode="L").save(out / "masks" / f"obj_{oid:05d}.png")
        (out / "manifest.json").write_text(
            json.dumps(man.model_dump(mode="json"), indent=1, sort_keys=True), encoding="utf-8"
        )
        return man


def _sub_run(p: FloatArray, c: FloatArray, radius: float) -> FloatArray:
    """Contiguous run of polyline points within ``radius`` of ``c`` around the nearest vertex.

    The run is extended by linear interpolation to the exact radius so short patches stay
    centred even on coarsely sampled polylines.
    """
    d = np.hypot(p[:, 0] - c[0], p[:, 1] - c[1])
    k = int(np.argmin(d))
    if d[k] > radius:
        return np.zeros((0, 2))
    lo = k
    while lo > 0 and d[lo - 1] <= radius:
        lo -= 1
    hi = k
    while hi < len(p) - 1 and d[hi + 1] <= radius:
        hi += 1
    run = [p[i] for i in range(lo, hi + 1)]
    if lo > 0:
        run.insert(0, _to_radius(p[lo], p[lo - 1], c, radius))
    if hi < len(p) - 1:
        run.append(_to_radius(p[hi], p[hi + 1], c, radius))
    return np.asarray(run, dtype=np.float64)


def _to_radius(inside: FloatArray, outside: FloatArray, c: FloatArray, radius: float) -> FloatArray:
    lo_t, hi_t = 0.0, 1.0
    for _ in range(30):
        t = (lo_t + hi_t) / 2
        q = inside + (outside - inside) * t
        if float(np.hypot(q[0] - c[0], q[1] - c[1])) <= radius:
            lo_t = t
        else:
            hi_t = t
    return np.asarray(inside + (outside - inside) * lo_t, dtype=np.float64)


REPO_ROOT = Path(__file__).resolve().parents[3]


def render_cfg(profile: dict[str, Any]) -> dict[str, Any]:
    """The render settings a plugin stores in its scene payload (background, font, size)."""
    r = profile["render"]
    return {
        "background_rgb": [int(v) for v in r["background_rgb"]],
        "font_path": str(r["font_path"]),
        "font_sha256": str(r["font_sha256"]),
        "canvas_px": int(r["canvas_px"]),
        "supersample": int(r["supersample"]),
    }


def make_canvas(cfg: dict[str, Any]) -> LayeredCanvas:
    """Canvas from `render_cfg` output; relative font paths resolve against the repository root."""
    font = Path(cfg["font_path"])
    if not font.is_absolute():
        font = REPO_ROOT / font
    bg = cfg["background_rgb"]
    return LayeredCanvas(int(cfg["canvas_px"]), int(cfg["supersample"]), (int(bg[0]), int(bg[1]), int(bg[2])),
                         font, cfg["font_sha256"])  # fmt: skip
