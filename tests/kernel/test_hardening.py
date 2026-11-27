"""v2 kernel fixes: scale wraparound, port search at the canvas edge, crossing majority vote,
occlusion diagnostics, claim accounting, trail preservation and strict single-answer grading.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fixtures import (
    CFG,
    canvas,
    int_answer,
    label_answer,
    tid,
    wires_bundle,
    write_claims,
    x_scene,
)

from knotwork.knotcore import engine, gauge
from knotwork.knotcore.contracts import (
    Claim,
    ErrorCode,
    Family,
    GaugeEntry,
    PixelBox,
    QuestionProgram,
    TerminalEntry,
    TickEntry,
)
from knotwork.knotcore.grading import committed_answers, grade_prediction, is_hedged
from knotwork.knotcore.io import KernelFault, load_bundle
from knotwork.knotcore.topo import crossings, ontop, overlaps, window
from knotwork.knotcore.witness import witness

BID = "kw-" + "0" * 20


# ------------------------------------------------------------------ _phi / _wrap
@pytest.mark.parametrize(
    ("ref", "theta", "want"),
    [
        (225.0, 225.0, 0.0),  # scale start
        (225.0, 270.0, -45.0),  # lower wrap edge is included
        (225.0, 270.0 + 1e-9, 315.0),  # just past it wraps to the top
        (225.0, 315.0, 270.0),  # scale end
        (0.0, 0.0, 0.0),
        (0.0, 360.0, 0.0),
        (360.0, 0.0, 0.0),
        (1.0, 359.0, 2.0),  # across the 0/360 seam, forwards
        (359.0, 1.0, -2.0),  # across the 0/360 seam, backwards
    ],
)
def test_phi_wraparound(ref: float, theta: float, want: float) -> None:
    assert float(gauge._phi(ref, theta)) == pytest.approx(want, abs=1e-6)


def test_phi_range_never_reaches_upper_bound() -> None:
    theta = np.array([-720.0, -1e-17, 0.0, 1e-17, 45.0, 359.9999999999999, 360.0, 1080.0])
    for ref in (0.0, -1e-17, 45.0, 315.0, 360.0):
        phi = gauge._phi(ref, theta + 45.0)
        assert float(phi.min()) >= -45.0 and float(phi.max()) < 315.0, (ref, phi)
    assert float(gauge._wrap(-1e-17)) == 0.0
    assert float(gauge._wrap(360.0)) == 0.0


# ------------------------------------------------------------------ gauge_of at the canvas edge
def _edge_bundle(out: Path, cx: float, cy: float, box: PixelBox) -> Path:
    c = canvas()
    dial = c.new_object("fx/dial:g1", "dial", (255, 255, 255), parent="fx/gauge:g1")
    c.disk(dial, (cx, cy), 40.0)
    needle = c.new_object("fx/needle:g1", "needle", (200, 30, 30), parent="fx/gauge:g1")
    c.polyline(needle, [(cx, cy), (cx + 20, cy)], 6)
    ticks = []
    for i in range(2):
        tick = c.new_object(f"fx/tick:g1_{i}", "tick", (0, 0, 0), parent="fx/gauge:g1")
        c.disk(tick, (cx, cy - 20 + 40 * i), 3.0)
        lab = c.new_object(f"fx/label:g1_t{i}", "label", (0, 0, 0), label_text=str(i))
        c.text(lab, (500.0 + 60 * i, 500.0), str(i), 26)
        ticks.append(TickEntry(tick_obj=tick, label_obj=lab))
    entry = GaugeEntry(
        gauge_id="fx/gauge:g1", dial_obj=dial, needle_obj=needle, ticks=ticks,
        claimed_center=(cx, cy), claimed_radius=40.0,
    )  # fmt: skip
    port = TerminalEntry(terminal_id="fx/terminal:p1", role="port", box=box)
    c.finish(out, "fx", terminals=[port], gauges=[entry])
    write_claims(out, "fx", [])
    return out


@pytest.mark.parametrize(
    ("center", "box"),
    [
        ((50.0, 200.0), (0, 190, 8, 210)),  # < 9 px from the left edge
        ((200.0, 50.0), (190, 3, 210, 8)),  # < 9 px from the top edge
        ((45.0, 45.0), (0, 40, 4, 50)),  # corner
        ((975.0, 975.0), (1016, 970, 1024, 980)),  # bottom-right corner
        ((50.0, 200.0), (-6, 190, 8, 210)),  # box partly off canvas is clipped, not wrapped
    ],
)
def test_gauge_of_near_canvas_edge(
    tmp_path: Path, center: tuple[float, float], box: tuple[int, int, int, int]
) -> None:
    px = PixelBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3])
    b = load_bundle(_edge_bundle(tmp_path, *center, px))
    assert gauge.gauge_of(b, "fx/terminal:p1", CFG)[0] == "fx/gauge:g1"


@pytest.mark.parametrize("box", [(-30, -30, -10, -10), (0, 600, 6, 620), (1040, 10, 1060, 30)])
def test_gauge_of_far_or_off_canvas_port_has_no_gauge(
    tmp_path: Path, box: tuple[int, int, int, int]
) -> None:
    px = PixelBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3])
    b = load_bundle(_edge_bundle(tmp_path, 60.0, 60.0, px))
    with pytest.raises(KernelFault) as e:
        gauge.gauge_of(b, "fx/terminal:p1", CFG)
    assert e.value.code is ErrorCode.E_TRACE_NO_TERMINAL


# ------------------------------------------------------------------ crossing vote
def test_neighbour_halo_over_centre_pixel_is_not_occlusion(tmp_path: Path) -> None:
    """A halo slit through the centre pixel (as a neighbouring crossing draws) keeps the vote."""

    def extra(c, ids):  # type: ignore[no-untyped-def]
        c._idraw.rectangle([507, 497, 513, 501], fill=0)

    b = load_bundle(wires_bundle(tmp_path, x_scene(top="A", extra=extra)))
    centre = crossings(b, tid("s", "A"), tid("s", "B"))[0]["center"]
    assert b.ids[round(centre[1]), round(centre[0])] == 0
    got, ev = ontop(b, tid("s", "A"), tid("s", "B"), 0, CFG)
    assert got == "A" and ev["share"] > 0.9


def test_third_object_on_the_core_is_still_occlusion(tmp_path: Path) -> None:
    def extra(c, ids):  # type: ignore[no-untyped-def]
        o = c.new_object("fx/occluder:o", "occluder", (120, 120, 120))
        c.disk(o, (510.0, 500.0), 7)

    b = load_bundle(wires_bundle(tmp_path, x_scene(extra=extra)))
    with pytest.raises(KernelFault) as e:
        ontop(b, tid("s", "A"), tid("s", "B"), 0, CFG)
    assert e.value.code is ErrorCode.E_OCCLUDED_CROSSING


def test_overlaps_are_computed_once_per_unordered_pair(x_dir: Path) -> None:
    b = load_bundle(x_dir)
    a, o = b.kind("strand")
    assert overlaps(b, a, o) is overlaps(b, o, a)
    full = np.argwhere(b.masks[a])
    sy, sx = window(b, a)
    assert sy.start <= full[:, 0].min() and sy.stop > full[:, 0].max()
    assert sx.start <= full[:, 1].min() and sx.stop > full[:, 1].max()


# ------------------------------------------------------------------ witness diagnostics
def test_decor_over_crossing_is_reported_as_occlusion(tmp_path: Path) -> None:
    def extra(c, ids):  # type: ignore[no-untyped-def]
        o = c.new_object("fx/decor:cover", "decor", (120, 120, 120))
        c.disk(o, (510.0, 500.0), 18)

    checks = witness(load_bundle(wires_bundle(tmp_path, x_scene(extra=extra))), CFG)
    bad = [c for c in checks if not c.passed and c.name == "crossing_color"]
    assert [c.code for c in bad] == [ErrorCode.E_WITNESS_OCCLUDED]
    assert bad[0].obj_id is not None and bad[0].metric == bad[0].obj_id


# ------------------------------------------------------------------ engine accounting
def _trace_q() -> QuestionProgram:
    return QuestionProgram(
        program_id="q", family=Family.T1_TRACE, text="Follow the cable that starts at A.",
        steps=[{"step_id": "s0", "op": "TRACE", "args": [tid("s", "A")]}], answer_type="label",
    )  # fmt: skip


def test_repeated_failing_claim_counts_once(tmp_path: Path) -> None:
    sc = x_scene()
    wrong = Claim(claim_id="c", op="CROSS", args=[tid("s", "A"), "*"], expected=int_answer(4))
    other = Claim(claim_id="d", op="CROSS", args=[tid("s", "B"), "*"], expected=int_answer(9))
    sc.claims = [wrong, wrong.model_copy(update={"claim_id": "c2"}), other]
    b = load_bundle(wires_bundle(tmp_path, sc))
    n_failed, codes = engine.check_claims(b, b.claims, CFG)
    assert n_failed == 2 and codes == [ErrorCode.E_CLAIM_MISMATCH] * 2


def test_step_fault_keeps_d1_d3_mismatch_in_trail(x_dir: Path) -> None:
    q = QuestionProgram(
        program_id="q", family=Family.T3_ONTOP, text="Which cable is on top at crossing 5?",
        steps=[{"step_id": "s0", "op": "ONTOP", "args": [tid("s", "A"), tid("s", "B"), 4]}],
        answer_type="label",
    )  # fmt: skip
    cert = engine.certify(x_dir, q, label_answer("A"), None, CFG, BID)
    assert {ErrorCode.E_CLAIM_MISMATCH, ErrorCode.E_D1_D3_MISMATCH} <= set(cert.codes)
    assert not cert.agree


def test_clean_program_has_no_mismatch(x_dir: Path) -> None:
    cert = engine.certify(x_dir, _trace_q(), label_answer("1"), None, CFG, BID)
    assert ErrorCode.E_D1_D3_MISMATCH not in cert.codes and cert.agree


# ------------------------------------------------------------------ strict single answer
@pytest.mark.parametrize(
    ("pred", "hedged"),
    [
        ("ANSWER: 6\nANSWER: 7", True),
        ("ANSWER: 7 ANSWER: 6", True),
        ("answer: red\nANSWER: amber\nANSWER: green", True),
        ("ANSWER: 7", False),
        ("ANSWER: 7\nANSWER: 7.", False),
        ("ANSWER: Red\n  answer:   red", False),
        ("The format is 'ANSWER: <number>' so:\nANSWER: 7", False),
        ("no marker 6 or 7", False),
    ],
)
def test_hedging_detection(pred: str, hedged: bool) -> None:
    assert is_hedged(pred) is hedged


def test_strict_flag_controls_hedged_grading() -> None:
    answer = {"type": "label", "value": "7", "tolerance": 0.0}
    spam = "ANSWER: 6\nANSWER: 7"
    assert grade_prediction(spam, answer) == 0.0
    assert grade_prediction(spam, answer, strict_single_answer=False) == 1.0
    assert committed_answers(spam) == ["6", "7"]
    assert not is_hedged(7) and grade_prediction(7, answer) == 1.0
