"""OCR of label objects on the 768-px consumption image only, with a crop-hash cache (SPEC 7.9).

Invariants: one OCR pass per label per bundle (memoised on the Bundle); identical preprocessed crops
share one result across bundles (module LRU keyed by `seeds.array_digest`); the RapidOCR engine is a
lazy per-process singleton; template matching with the bundled font is the deterministic fallback.
"""

from __future__ import annotations

import unicodedata
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont
from skimage.feature import match_template

from knotwork.knotcore.contracts import ErrorCode
from knotwork.knotcore.io import Bundle, KernelFault
from knotwork.knotcore.seeds import array_digest

FONT = Path(__file__).resolve().parents[3] / "assets" / "fonts" / "DejaVuSans.ttf"
CACHE_CAPACITY = 4096
_CACHE: OrderedDict[str, tuple[str, float]] = OrderedDict()
_ENGINE: list[Any] = []
U8 = npt.NDArray[np.uint8]


def rapid(img: U8) -> tuple[str, float]:
    """Highest-confidence RapidOCR text of an RGB crop (engine constructed once per process)."""
    if not _ENGINE:
        from rapidocr_onnxruntime import RapidOCR

        _ENGINE.append(RapidOCR())
    res, _ = _ENGINE[0](img, use_det=True, use_cls=False, use_rec=True)
    if not res:
        return "", 0.0
    best = max(res, key=lambda r: float(r[2]))
    return str(best[1]), float(best[2])


def recognise(img: U8) -> tuple[str, float]:
    """Cached `rapid`: key = array_digest of the preprocessed crop, insertion-ordered eviction."""
    key = array_digest(img)
    if key not in _CACHE:
        _CACHE[key] = rapid(img)
        while len(_CACHE) > CACHE_CAPACITY:
            _CACHE.popitem(last=False)
    return _CACHE[key]


def crop768(b: Bundle, oid: int) -> U8:
    box = b.manifest.obj(oid).box
    if box is None:
        raise KernelFault(ErrorCode.E_OCR_FAIL, f"label {oid} has no box")
    x0, y0 = max(int(box.x0 * 0.75) - 5, 0), max(int(box.y0 * 0.75) - 5, 0)
    x1, y1 = min(int(box.x1 * 0.75) + 5, 768), min(int(box.y1 * 0.75) + 5, 768)
    gray = np.asarray(Image.fromarray(b.beauty768[y0:y1, x0:x1]).convert("L"), dtype=np.uint8)
    if gray.size == 0:
        raise KernelFault(ErrorCode.E_OCR_FAIL, f"label {oid} box is empty at 768 px")
    return gray


def preprocess(gray: U8, cfg: dict[str, Any]) -> U8:
    up = int(cfg["ocr"]["upscale"])
    big = Image.fromarray(gray).resize(
        (gray.shape[1] * up, gray.shape[0] * up), Image.Resampling.LANCZOS
    )
    a = np.asarray(big, dtype=np.uint8)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    a = np.pad(a, int(cfg["ocr"]["pad_px"]), constant_values=int(np.median(border)))
    return np.repeat(a[:, :, None], 3, axis=2)


def normalise(text: str, digits: bool) -> str:
    t = "".join(unicodedata.normalize("NFKC", text).split()).upper().replace("×", "X")
    return t.translate(str.maketrans("ODILSB", "001158")) if digits else t


def template_match(gray: U8, font_px: int, candidates: list[str]) -> str | None:
    """Best normalised cross-correlation among candidate glyphs rendered through the beauty chain
    (2x supersample -> 1024 -> 768, Lanczos) with a 4-px background margin; needs a 0.05 lead."""
    font, lz = ImageFont.truetype(str(FONT), 2 * font_px), Image.Resampling.LANCZOS
    scores: list[tuple[float, str]] = [(0.0, "")]
    for cand in sorted(set(candidates)):
        im = Image.new("L", (font_px * 8, font_px * 4), 255)
        ImageDraw.Draw(im).text((16, 16), cand, font=font, fill=0)
        im = im.resize((font_px * 4, font_px * 2), lz).resize(
            (font_px * 3, round(font_px * 1.5)), lz
        )
        t = np.asarray(im, dtype=np.float64)
        ys, xs = np.nonzero(t < 200)
        t = t[max(ys.min() - 4, 0) : ys.max() + 5, max(xs.min() - 4, 0) : xs.max() + 5]
        fits = t.shape[0] <= gray.shape[0] and t.shape[1] <= gray.shape[1]
        ncc = match_template(gray.astype(np.float64), t) if fits else None  # type: ignore[no-untyped-call]
        scores.append((float(np.max(ncc)) if ncc is not None else -1.0, cand))
    scores.sort(reverse=True)
    return scores[0][1] if scores[0][0] - scores[1][0] >= 0.05 else None


def label_text(b: Bundle, oid: int, cfg: dict[str, Any]) -> tuple[str, str]:
    """(normalised OCR text, engine) for label object `oid`; MUST equal the manifest label."""
    memo: dict[int, tuple[str, str]] = b.memo.setdefault("ocr", {})
    if oid not in memo:
        obj = b.manifest.obj(oid)
        want = obj.label_text or ""
        gray = crop768(b, oid)
        text, conf = ("", 0.0)
        if cfg["ocr"].get("engine", "rapidocr") == "rapidocr":
            text, conf = recognise(preprocess(gray, cfg))
        got, engine = normalise(text, want.isdigit()), "rapidocr"
        if not got or conf < float(cfg["ocr"]["min_confidence"]):
            group = [
                o.label_text
                for o in b.manifest.objects
                if o.kind.value == "label" and o.parent == obj.parent and o.label_text
            ]
            got, engine = template_match(gray, obj.font_px or 26, group) or "", "template"
        memo[oid] = (got, engine)
    got, engine = memo[oid]
    if not got:
        raise KernelFault(ErrorCode.E_OCR_FAIL, f"no text for label {oid}")
    if got != (b.manifest.obj(oid).label_text or ""):
        raise KernelFault(ErrorCode.E_LABEL_MISMATCH, f"label {oid}: read {got!r}")
    return got, engine
