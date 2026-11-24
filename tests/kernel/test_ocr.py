"""Task 1.2: OCR at 768 px with crop-hash cache and template fallback (SPEC 7.9)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from fixtures import CFG, PROFILE, canvas, wires_bundle, write_claims, x_scene

from knotwork.knotcore import ocr
from knotwork.knotcore.contracts import ErrorCode
from knotwork.knotcore.io import KernelFault, load_bundle

LABELS = (
    list(PROFILE["tangle"]["start_labels"])
    + [str(i) for i in range(1, 11)]
    + ["G1", "G2", "G3", "G4"]
    + [f"{v:02d}" for v in range(0, 100, 10)]
)


@pytest.fixture(scope="module")
def label_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("labels")
    c = canvas()
    for k, t in enumerate(LABELS):
        group = (
            "ticks"
            if len(t) == 2 and t.isdigit()
            else ("ends" if t.isdigit() else ("letters" if len(t) == 1 else "gauges"))
        )
        o = c.new_object(
            f"fx/label:l{k}", "label", (25, 25, 25), parent=f"fx/labels:{group}", label_text=t
        )
        c.text(o, (70 + (k % 9) * 105, 90 + (k // 9) * 130), t, 26)
    blank = c.new_object(
        "fx/label:blank", "label", (25, 25, 25), parent="fx/labels:blank", label_text="Q"
    )
    c.text(blank, (500, 900), "Q", 26, beauty=False)
    c.finish(out, "fx")
    write_claims(out, "fx", [])
    return out


def _label_ids(d: Path) -> list[int]:
    b = load_bundle(d)
    return [
        o.obj_id
        for o in b.manifest.objects
        if o.label_text in LABELS and o.name != "fx/label:blank"
    ]


def test_rapidocr_reads_every_label_at_768(label_dir: Path) -> None:
    b = load_bundle(label_dir)
    ids = _label_ids(label_dir)
    raw_ok = 0
    for oid in ids:
        want = b.manifest.obj(oid).label_text or ""
        text, _ = ocr.recognise(ocr.preprocess(ocr.crop768(b, oid), CFG))
        raw_ok += ocr.normalise(text, want.isdigit()) == want
        assert ocr.label_text(b, oid, CFG)[0] == want  # 100 % after fallback
    assert raw_ok / len(ids) >= 0.99


def test_cache_hit_skips_engine(label_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def stub(img):  # type: ignore[no-untyped-def]
        calls.append(1)
        return "7", 0.99

    monkeypatch.setattr(ocr, "rapid", stub)
    monkeypatch.setattr(ocr, "_CACHE", type(ocr._CACHE)())
    b = load_bundle(label_dir)
    img = ocr.preprocess(ocr.crop768(b, _label_ids(label_dir)[0]), CFG)
    assert ocr.recognise(img) == ("7", 0.99)
    assert ocr.recognise(img.copy()) == ("7", 0.99)
    assert len(calls) == 1
    from knotwork.knotcore.seeds import array_digest

    assert list(ocr._CACHE) == [array_digest(img)]


def test_cache_capacity_evicts_oldest(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    monkeypatch.setattr(ocr, "rapid", lambda img: ("1", 1.0))
    monkeypatch.setattr(ocr, "_CACHE", type(ocr._CACHE)())
    monkeypatch.setattr(ocr, "CACHE_CAPACITY", 3)
    imgs = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(5)]
    for im in imgs:
        ocr.recognise(im)
    assert len(ocr._CACHE) == 3


def test_forced_template_fallback(label_dir: Path) -> None:
    cfg = copy.deepcopy(CFG)
    cfg["ocr"]["engine"] = "template"
    b = load_bundle(label_dir)
    for oid in _label_ids(label_dir):
        text, engine = ocr.label_text(b, oid, cfg)
        assert engine == "template" and text == b.manifest.obj(oid).label_text


def test_swapped_label_is_mismatch(tmp_path: Path) -> None:
    d = wires_bundle(tmp_path, x_scene(end_label_override={"1": "2", "2": "1"}))
    b = load_bundle(d)
    end = next(o for o in b.manifest.objects if o.name == "fx/label:e_A")
    with pytest.raises(KernelFault) as e:
        ocr.label_text(b, end.obj_id, CFG)
    assert e.value.code is ErrorCode.E_LABEL_MISMATCH


def test_blank_crop_is_ocr_fail(label_dir: Path) -> None:
    b = load_bundle(label_dir)
    blank = next(o for o in b.manifest.objects if o.name == "fx/label:blank")
    with pytest.raises(KernelFault) as e:
        ocr.label_text(b, blank.obj_id, CFG)
    assert e.value.code is ErrorCode.E_OCR_FAIL


def test_engine_is_lazy_singleton(label_dir: Path) -> None:
    b = load_bundle(label_dir)
    ocr.rapid(ocr.preprocess(ocr.crop768(b, _label_ids(label_dir)[1]), CFG))
    first = ocr._ENGINE[0]
    ocr.rapid(ocr.preprocess(ocr.crop768(b, _label_ids(label_dir)[2]), CFG))
    assert len(ocr._ENGINE) == 1 and ocr._ENGINE[0] is first
