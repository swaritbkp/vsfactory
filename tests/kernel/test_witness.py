"""Task 1.9: beauty-to-mask witness (SPEC 7.10)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fixtures import BG, CFG, wires_bundle, x_scene

from knotwork.knotcore.contracts import ErrorCode, PixelBox, WitnessCheck
from knotwork.knotcore.io import load_bundle
from knotwork.knotcore.witness import witness


def _failed(checks: list[WitnessCheck]) -> set[ErrorCode]:
    return {c.code for c in checks if not c.passed}


def test_clean_bundle_passes(x_dir: Path) -> None:
    checks = witness(load_bundle(x_dir), CFG)
    assert all(isinstance(c, WitnessCheck) for c in checks)
    assert _failed(checks) == set(), [c for c in checks if not c.passed]
    assert {c.name for c in checks} >= {
        "contrast",
        "crossing_color",
        "unexplained_ink",
        "oob_mask",
        "oob_box",
    }


def test_low_contrast_wire(tmp_path: Path) -> None:
    sc = x_scene()
    blend = tuple(int(0.08 * c + 0.92 * g) for c, g in zip(sc.wires[0].color, BG, strict=True))
    sc.wires[0].beauty_rgb = blend  # type: ignore[assignment]
    assert ErrorCode.E_BEAUTY_MISMATCH in _failed(
        witness(load_bundle(wires_bundle(tmp_path, sc)), CFG)
    )


def test_beauty_top_differs_from_id_top(tmp_path: Path) -> None:
    sc = x_scene(top="A")
    sc.wires[0].color = (30, 30, 30)  # dark A on top in the id buffer, orange B painted on top

    def extra(c, ids):  # type: ignore[no-untyped-def]
        # repaint B over the crossing on the beauty layer only (id buffer still says A is on top)
        b = sc.wires[1]
        near = np.hypot(b.pts[:, 0] - 510, b.pts[:, 1] - 500) < 18
        c.raw_beauty_polyline(b.pts[near], 9, b.color)

    sc.extra = extra
    checks = witness(load_bundle(wires_bundle(tmp_path, sc)), CFG)
    bad = [c for c in checks if not c.passed]
    assert any(c.name == "crossing_color" and c.code is ErrorCode.E_BEAUTY_MISMATCH for c in bad)


def test_ink_without_id(tmp_path: Path) -> None:
    def extra(c, ids):  # type: ignore[no-untyped-def]
        c.raw_beauty_polyline(np.array([[300.0, 850.0], [700.0, 900.0]]), 6, (40, 40, 40))

    checks = witness(load_bundle(wires_bundle(tmp_path, x_scene(extra=extra))), CFG)
    assert ErrorCode.E_UNEXPLAINED_INK in _failed(checks)


def test_aa_fringe_not_flagged(x_dir: Path) -> None:
    ink = next(c for c in witness(load_bundle(x_dir), CFG) if c.name == "unexplained_ink")
    assert ink.passed and ink.metric <= 30


def test_mask_on_border_and_oob_box(tmp_path: Path) -> None:
    sc = x_scene()
    sc.wires[0].pts = np.concatenate([sc.wires[0].pts, [[1023.0, 760.0]]])
    checks = witness(load_bundle(wires_bundle(tmp_path / "a", sc)), CFG)
    assert any(c.name == "oob_mask" and not c.passed for c in checks)

    def extra(c, ids):  # type: ignore[no-untyped-def]
        o = c.new_object(
            "fx/decor:off", "decor", (0, 0, 0), box=PixelBox(x0=1000, y0=10, x1=1040, y1=30)
        )
        del o

    checks = witness(load_bundle(wires_bundle(tmp_path / "b", x_scene(extra=extra))), CFG)
    assert any(c.name == "oob_box" and not c.passed and c.code is ErrorCode.E_OOB for c in checks)
