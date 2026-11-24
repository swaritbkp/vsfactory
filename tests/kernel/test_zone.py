"""Task 1.7: ZONE, GAUGE_OF, READ_ALL (SPEC 7.6-7.7)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fixtures import CFG, GID, Dial, gauge_bundle, panel_bundle

from knotwork.knotcore.contracts import ErrorCode
from knotwork.knotcore.gauge import gauge_of, read_all, zone
from knotwork.knotcore.io import KernelFault, load_bundle


@pytest.mark.parametrize(("value", "want"), [(25.0, "green"), (47.0, "amber"), (71.0, "red")])
def test_zone_from_arc_masks(tmp_path: Path, value: float, want: str) -> None:
    b = load_bundle(gauge_bundle(tmp_path, Dial(value=value)))
    assert zone(b, GID, CFG)[0] == want


def test_needle_near_boundary_is_unresolved(tmp_path: Path) -> None:
    b = load_bundle(gauge_bundle(tmp_path, Dial(value=45.1, zones=(45.0, 60.0))))
    with pytest.raises(KernelFault) as e:
        zone(b, GID, CFG)
    assert e.value.code is ErrorCode.E_ZONE_UNRESOLVED


def test_gauge_of_and_read_all(tmp_path: Path) -> None:
    b = load_bundle(panel_bundle(tmp_path))
    assert gauge_of(b, "fx/terminal:p1", CFG)[0] == "fx/gauge:g1"
    assert gauge_of(b, "G2", CFG)[0] == "fx/gauge:g2"
    rows = read_all(b, ["fx/terminal:s_S", "fx/terminal:s_T"], CFG)[0]
    assert [r[0] for r in rows] == ["S", "T"]
    assert abs(rows[0][1] - 62.0) <= rows[0][2] and abs(rows[1][1] - 30.0) <= rows[1][2]


def test_gauge_of_needs_a_touching_dial(tmp_path: Path) -> None:
    b = load_bundle(panel_bundle(tmp_path, far=True))
    with pytest.raises(KernelFault) as e:
        gauge_of(b, "fx/terminal:p1", CFG)
    assert e.value.code is ErrorCode.E_TRACE_NO_TERMINAL
