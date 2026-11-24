"""Task 1.3: TRACE (SPEC 7.1-7.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fixtures import CFG, Scene, Wire, polyline, tid, wires_bundle, x_scene

from knotwork.knotcore.contracts import ErrorCode
from knotwork.knotcore.io import Bundle, KernelFault, load_bundle
from knotwork.knotcore.topo import trace


def test_trace_end_labels(x_bundle: Bundle) -> None:
    lab, ev = trace(x_bundle, tid("s", "A"), CFG)
    assert lab == "1" and ev["end_terminal"] == tid("e", "A") and ev["ocr_text"] == "1"
    assert ev["path_px"] > 700
    assert trace(x_bundle, tid("s", "B"), CFG)[0] == "2"


def test_broken_mask_has_no_terminal(tmp_path: Path) -> None:
    sc = x_scene()
    sc.wires[0].gap = (600.0, 660.0)
    b = load_bundle(wires_bundle(tmp_path, sc))
    with pytest.raises(KernelFault) as e:
        trace(b, tid("s", "A"), CFG)
    assert e.value.code is ErrorCode.E_TRACE_NO_TERMINAL


def test_mask_touching_two_end_boxes_is_ambiguous(tmp_path: Path) -> None:
    sc = x_scene()

    def extra(c, ids):  # type: ignore[no-untyped-def]
        # a spur on A from (700, ~605) up to B's end box at (870, 300)
        c.polyline(
            ids["A"], np.array([[700.0, 300 + 550 * 400 / 720], [800.0, 230.0], [870.0, 300.0]]), 9
        )

    sc.extra = extra
    b = load_bundle(wires_bundle(tmp_path, sc))
    with pytest.raises(KernelFault) as e:
        trace(b, tid("s", "A"), CFG)
    assert e.value.code is ErrorCode.E_TRACE_AMBIGUOUS


def test_start_box_on_two_strands_is_ambiguous(tmp_path: Path) -> None:
    sc = Scene(
        wires=[
            Wire("A", "1", polyline(150, 870, lambda x: 0 * x + 300)),
            Wire("B", "2", polyline(150, 870, lambda x: 0 * x + 306)),
        ]
    )
    b = load_bundle(wires_bundle(tmp_path, sc))
    with pytest.raises(KernelFault) as e:
        trace(b, tid("s", "A"), CFG)
    assert e.value.code is ErrorCode.E_TRACE_AMBIGUOUS
