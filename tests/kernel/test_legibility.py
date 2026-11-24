"""Task 1.10: 768-px legibility guard (SPEC 7.11)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from fixtures import CFG, ORANGE, Scene, Wire, polyline, wires_bundle, x_scene

from knotwork.knotcore.contracts import ErrorCode, WitnessCheck
from knotwork.knotcore.io import load_bundle
from knotwork.knotcore.legibility import legibility


def _failed(checks: list[WitnessCheck]) -> dict[str, ErrorCode]:
    return {c.name: c.code for c in checks if not c.passed}


def test_clean_x_is_legible(x_dir: Path) -> None:
    checks = legibility(load_bundle(x_dir), CFG)
    assert _failed(checks) == {}, [c for c in checks if not c.passed]
    assert sum(c.name == "ocr_768" for c in checks) == 4  # labels OCR'd at 768


def test_thin_strokes(tmp_path: Path) -> None:
    sc = x_scene()
    for w in sc.wires:
        w.width = 1.0
    assert (
        _failed(legibility(load_bundle(wires_bundle(tmp_path, sc)), CFG)).get("stroke_768")
        is ErrorCode.E_ILLEGIBLE_RES
    )


def test_tangent_crossing(tmp_path: Path) -> None:
    half = math.tan(math.radians(7.5))  # two lines at +-7.5 deg -> 15 deg crossing
    a = polyline(150, 870, lambda x: 500 + (x - 510) * half)
    b = polyline(150, 870, lambda x: 500 - (x - 510) * half)
    sc = Scene(
        wires=[Wire("A", "1", a), Wire("B", "2", b, ORANGE)], crossings=[((510.0, 500.0), "A")]
    )
    assert (
        ErrorCode.E_TANGENT_CROSSING
        in _failed(legibility(load_bundle(wires_bundle(tmp_path, sc)), CFG)).values()
    )


def test_crowded_crossings(tmp_path: Path) -> None:
    # A ripples across horizontal B: crossings ~32 px apart along A (< 4 strokes = 40 px)
    lead = polyline(150, 300, lambda x: 350 + (x - 150) * 164 / 150, 31)
    a = np.concatenate(
        [
            lead,
            polyline(300, 870, lambda x: 500 + 14 * np.cos((x - 300) * 2 * math.pi / 60), 571)[1:],
        ]
    )
    b = polyline(150, 870, lambda x: 0 * x + 500)
    sc = Scene(wires=[Wire("A", "1", a), Wire("B", "2", b, ORANGE)])
    assert (
        _failed(legibility(load_bundle(wires_bundle(tmp_path, sc)), CFG)).get("crossing_spacing")
        is ErrorCode.E_CROSSING_SPACING
    )


def test_foreign_ink_near_terminal(tmp_path: Path) -> None:
    sc = x_scene()
    sc.wires.append(
        Wire("C", "3", np.array([[160.0, 316.0], [400.0, 316.0], [600.0, 900.0]]), ORANGE)
    )
    assert (
        _failed(legibility(load_bundle(wires_bundle(tmp_path, sc)), CFG)).get("terminal_clearance")
        is ErrorCode.E_TERMINAL_CLEARANCE
    )
