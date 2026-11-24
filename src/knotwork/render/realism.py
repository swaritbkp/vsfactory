"""Deterministic photographic finishing for renders (untrusted helper; SPEC 6, decision D-24).

Everything here runs at output resolution as a `LayeredCanvas.post` step and is a pure function of a
seeded parameter dict, so causal twins (same parameters) differ only where the scene changed, and
nuisance twins (new parameters) exercise a genuine domain shift.

Kernel compatibility contract (the kernel is unchanged; these are the rules this module obeys):
* Far (> 4 px) from every id-buffer object the finish may not move a pixel outside ``BG_CLAMP`` (36)
  levels of the manifest background, so surface texture is never "unexplained ink" (kernel threshold:
  40 levels). Pixels that were already outside that band (injected fault ink) are left as they were.
* Effects are attenuated over ``protect`` (numeral/label plates, needles, crossing halos) so OCR and
  the beauty-to-mask colour witnesses see crisp, uniform pixels there.
* Noise uses numpy's platform-independent Generator plus box/Gaussian filtering (no transcendental
  functions on arrays), keeping pixels identical across machines.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import ndimage

F32 = npt.NDArray[np.float32]
U8 = npt.NDArray[np.uint8]
BG_CLAMP = 36.0


def sample_surface(rng: np.random.Generator, finish: str = "powder") -> dict[str, Any]:
    """Seeded surface parameters (stored in scene payloads; twins copy or resample them)."""
    return {
        "seed": int(rng.integers(0, 2**31 - 1)),
        "light_angle": float(rng.integers(0, 360)),
        "light": float(rng.uniform(0.04, 0.08)),
        "grain": float(rng.uniform(1.6, 3.0)),
        "brush": float(rng.uniform(2.0, 5.0))
        if finish == "brushed"
        else float(rng.uniform(0.0, 1.5)),
        "brush_dir": int(rng.integers(0, 2)),
        "smudges": int(rng.integers(2, 6)),
        "dust": int(rng.integers(20, 60)),
        "finish": finish,
    }


def _unit_noise(
    rng: np.random.Generator, shape: tuple[int, int], sigma: tuple[float, float]
) -> F32:
    n = ndimage.gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma)
    return np.asarray(n / (float(n.std()) or 1.0), dtype=np.float32)


def _light_dir(angle_deg: float) -> tuple[float, float]:
    t = math.radians(angle_deg)
    return math.cos(t), -math.sin(t)


def _coords(shape: tuple[int, ...]) -> tuple[F32, F32]:
    h, w = shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    return x, y


def soft(mask: npt.NDArray[np.bool_], grow: int, blur: float) -> F32:
    """Protection weight in [0, 1]: mask grown by ``grow`` px and feathered."""
    m = ndimage.binary_dilation(mask, iterations=grow) if grow else mask
    return np.asarray(
        np.clip(ndimage.gaussian_filter(m.astype(np.float32), blur) * 2.0, 0.0, 1.0),
        dtype=np.float32,
    )


def surface(img: U8, ids: npt.NDArray[np.uint16], bg: tuple[int, int, int], p: dict[str, Any],
            protect: F32) -> U8:  # fmt: skip
    """Uneven overhead lighting, brushed/powder grain, smudges, dust, sensor grain; then the
    background clamp. ``protect`` (0..1) attenuates every effect."""
    rng = np.random.default_rng(int(p["seed"]))
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    x, y = _coords(img.shape)
    lx, ly = _light_dir(float(p["light_angle"]))
    u = ((x - w / 2) * lx + (y - h / 2) * ly) / (w / 2)
    r2 = ((x - w / 2) ** 2 + (y - h / 2) ** 2) / (w / 2) ** 2
    light = 1.0 + float(p["light"]) * (u - 0.6 * r2 + 0.3)
    sig = (0.6, 18.0) if p["brush_dir"] == 0 else (18.0, 0.6)
    brushed = _unit_noise(rng, (h, w), sig) * float(p["brush"])
    small = np.zeros((h // 4, w // 4), np.float32)  # smudges are low-frequency: blur at 1/4 scale
    for _ in range(int(p["smudges"])):
        imp = np.zeros_like(small)
        imp[int(rng.integers(0, h // 4)), int(rng.integers(0, w // 4))] = 1.0
        blob = ndimage.gaussian_filter(imp, float(rng.uniform(4.5, 14.0)))
        small -= blob / float(blob.max() or 1.0) * float(rng.uniform(5.0, 10.0))
    smudge = np.repeat(np.repeat(ndimage.gaussian_filter(small, 1.0), 4, axis=0), 4, axis=1)[:h, :w]
    dust = np.zeros((h, w), np.float32)
    for _ in range(int(p["dust"])):
        cy, cx = int(rng.integers(2, h - 3)), int(rng.integers(2, w - 3))
        dust[cy : cy + int(rng.integers(1, 3)), cx : cx + int(rng.integers(1, 3))] -= float(
            rng.uniform(10, 26)
        )
    grain = rng.standard_normal((h, w)).astype(np.float32) * float(p["grain"])
    keep = (1.0 - 0.85 * protect)[..., None]
    tint = np.array([1.0, 0.96, 0.88], np.float32)  # smudges/grime read slightly warm-brown
    # smooth lighting applies everywhere (protected areas would otherwise read as pasted-on patches);
    # only the high-frequency / blotchy effects are attenuated over protected regions
    lit = f * light[..., None]
    out = lit + keep * ((brushed + grain + dust)[..., None] + smudge[..., None] * tint)
    # Clamp only what the finish adds: far from objects the texture may not push a pixel outside
    # bg +- BG_CLAMP, but pixels that were already outside (e.g. injected fault ink, M7) keep their
    # value, so the finish can never hide a defect from the kernel.
    far = ~ndimage.binary_dilation(ids > 0, iterations=4)
    b = np.asarray(bg, np.float32)
    lo, hi = np.minimum(f, b - BG_CLAMP), np.maximum(f, b + BG_CLAMP)
    out[far] = np.clip(out[far], lo[far], hi[far])
    return np.asarray(np.clip(np.rint(out), 0, 255), dtype=np.uint8)


def dial_finish(img: U8, d: dict[str, Any], r_face: float, r_bezel: float, hub: float,
                protect: F32) -> U8:  # fmt: skip
    """Bevelled multi-ring metal bezel, recess shadow, inner face shadow, aged-enamel vignette, rim
    grime and a curved-glass glare crescent, all inside the dial's own id footprint."""
    s = d.get("surface", {})
    rng = np.random.default_rng(int(s.get("seed", 0)) + 7919)
    cx, cy = (float(v) for v in d["center"])
    f = img.astype(np.float32)
    x, y = _coords(img.shape)
    dx, dy = x - cx, y - cy
    r = np.sqrt(dx * dx + dy * dy)
    nx, ny = dx / np.maximum(r, 1.0), dy / np.maximum(r, 1.0)
    lx, ly = _light_dir(float(s.get("light_angle", 135.0)))
    facing = nx * lx + ny * ly  # +1 where the surface faces the light
    keep = 1.0 - 0.9 * protect
    bw = r_bezel - r_face
    ring = (r >= r_face) & (r < r_bezel)
    t = np.clip((r - r_face) / max(bw, 1.0), 0.0, 1.0)
    bevel = 1.0 + 0.32 * facing * (1.0 - 2.0 * t)  # outer half lit, inner half shaded
    grooves = 1.0 - 0.18 * (np.abs(t - 0.34) < 0.06) - 0.18 * (np.abs(t - 0.68) < 0.06)
    sheen = 1.0 + 0.12 * np.clip(1.0 - ((t - 0.5) / 0.2) ** 2, 0.0, 1.0)
    brushed = 1.0 + 0.03 * _unit_noise(rng, r.shape, (1.0, 6.0))
    mult = np.where(ring, bevel * grooves * sheen * brushed, 1.0)
    face = r < r_face
    depth = np.clip(1.0 - (r_face - r) / max(0.06 * r_face, 6.0), 0.0, 1.0) ** 2
    inner_shadow = 0.24 * depth * (0.6 - 0.4 * facing)
    vignette = 0.05 * (r / r_face) ** 2
    rim = np.clip((r - 0.80 * r_face) / (0.2 * r_face), 0.0, 1.0)
    grime = np.clip(_unit_noise(rng, r.shape, (14.0, 14.0)), 0.0, None) * 0.035 * rim
    mult = np.where(face, mult * (1.0 - keep * (inner_shadow + vignette + grime)), mult)
    out = f * mult[..., None]
    gx, gy = cx - lx * 0.42 * r_face, cy - ly * 0.42 * r_face  # glare centred toward the light
    ex, ey = x - gx, y - gy
    a, b = 0.62 * r_face, 0.30 * r_face
    ux, uy = ex * (-ly) + ey * lx, ex * lx + ey * ly
    bump = np.clip(1.0 - (ux / a) ** 2 - (uy / b) ** 2, 0.0, 1.0) ** 2
    glare = (0.20 * bump * face * keep)[..., None]
    out = out + (255.0 - out) * glare
    hd = np.sqrt((x - cx + lx * 0.3 * hub) ** 2 + (y - cy + ly * 0.3 * hub) ** 2)
    cap = (np.clip(1.0 - (hd / (0.55 * hub)) ** 2, 0.0, 1.0) ** 2 * 0.55 * (r < hub))[..., None]
    out = out + (255.0 - out) * cap
    rec = (r >= r_bezel) & (r < r_bezel + 3.0)
    out[rec] *= 0.72
    return np.asarray(np.clip(np.rint(out), 0, 255), dtype=np.uint8)


def screw(c: Any, prefix: str, name: str, center: tuple[float, float], radius: float) -> int:
    """Hex-socket screw head (a registered decor object: mask + id explain its ink)."""
    cx, cy = center
    oid: int = c.new_object(f"{prefix}/decor:{name}", "decor", (150, 152, 156))
    c.disk(oid, center, radius, beauty_rgb=(118, 120, 126))
    c.disk(oid, center, radius - 2.0, beauty_rgb=(168, 170, 176))
    hexagon = [(cx + 0.45 * radius * math.cos(math.radians(60 * k + 30)),
                cy + 0.45 * radius * math.sin(math.radians(60 * k + 30))) for k in range(6)]  # fmt: skip
    c.polygon(oid, hexagon, beauty_rgb=(70, 72, 78))
    return oid


def corner_screws(c: Any, prefix: str, inset: float = 36.0, radius: float = 11.0) -> list[int]:
    n = float(c.size)
    pts = [(inset, inset), (n - inset, inset), (inset, n - inset), (n - inset, n - inset)]
    return [screw(c, prefix, f"screw{k}", p, radius) for k, p in enumerate(pts)]


def add_surface(c: Any, p: dict[str, Any], protect_kinds: tuple[str, ...] = ("label", "needle"),
                extra: Any = None) -> None:  # fmt: skip
    """Register the seeded surface finish as the canvas's last post-process. Objects of
    ``protect_kinds`` (and the optional ``extra(canvas) -> bool mask``) stay crisp."""

    def run(img: U8, cv: Any) -> U8:
        clean = np.zeros(img.shape[:2], dtype=bool)
        for o in cv.manifest("x").objects:
            if o.kind.value == "label" and "label" in protect_kinds and o.box is not None:
                b = (
                    o.box
                )  # the whole OCR crop region (box + 5 px at 768 ~ 7 px at 1024), not just glyphs
                clean[max(b.y0 - 10, 0) : b.y1 + 10, max(b.x0 - 10, 0) : b.x1 + 10] = True
            elif o.kind.value in protect_kinds:
                clean |= cv.mask_array(o.obj_id)
        if extra is not None:
            clean |= extra(cv)
        return surface(img, cv.id_array(), cv.background, p, soft(clean, 4, 2.0))

    c.post.append(run)


__all__ = [
    "BG_CLAMP",
    "add_surface",
    "corner_screws",
    "dial_finish",
    "sample_surface",
    "screw",
    "soft",
    "surface",
]
