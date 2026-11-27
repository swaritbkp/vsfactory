"""Colour-vision gate and beauty-only oracle on tiny synthetic bundles (no OCR)."""

from __future__ import annotations

from pathlib import Path

from fixtures import BG, CFG, wires_bundle, x_scene

from knotwork.knotcore import engine, oracle
from knotwork.knotcore.contracts import ErrorCode
from knotwork.knotcore.cvd import cvd_checks, separation
from knotwork.knotcore.io import load_bundle


def _oracle_failures(out: Path) -> set[int | None]:
    b = load_bundle(out)
    checks = oracle.inspect(out, *engine.oracle_probes(b, CFG), CFG)
    return {c.obj_id for c in checks if not c.passed}


def test_confusable_strand_colours_are_ambiguous(tmp_path: Path) -> None:
    sc = x_scene()
    sc.wires[0].color, sc.wires[1].color = (136, 34, 85), (1, 102, 94)  # collide for deuteranopes
    assert separation(sc.wires[0].color, sc.wires[1].color) < 8.0
    checks = cvd_checks(load_bundle(wires_bundle(tmp_path, sc)), CFG)
    assert [(c.name, c.passed, c.code) for c in checks] == [
        ("cvd_strands", False, ErrorCode.E_CVD_AMBIGUOUS)
    ]


def test_separable_and_mono_colours_pass(tmp_path: Path) -> None:
    clean = cvd_checks(load_bundle(wires_bundle(tmp_path / "a", x_scene())), CFG)
    assert clean and all(c.passed for c in clean)
    sc = x_scene()
    sc.wires[1].color = sc.wires[0].color
    assert cvd_checks(load_bundle(wires_bundle(tmp_path / "b", sc)), CFG) == []


def test_oracle_accepts_visible_and_rejects_ghost_strand(tmp_path: Path) -> None:
    assert _oracle_failures(wires_bundle(tmp_path / "a", x_scene())) == set()
    sc = x_scene()
    sc.wires[1].beauty_rgb = (BG[0], BG[1], BG[2])  # in the mask and id buffer, invisible to people
    out = wires_bundle(tmp_path / "b", sc)
    ghost = load_bundle(out).kind("strand")[1]
    assert ghost in _oracle_failures(out)
