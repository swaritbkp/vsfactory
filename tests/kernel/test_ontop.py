"""Task 1.5: ONTOP (SPEC 7.4)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fixtures import CFG, doubleback_scene, tid, wires_bundle, x_scene

from knotwork.knotcore.contracts import ErrorCode
from knotwork.knotcore.io import KernelFault, load_bundle
from knotwork.knotcore.topo import crossings, ontop


@pytest.mark.parametrize("top", ["A", "B"])
def test_reads_id_buffer_at_centre(tmp_path: Path, top: str) -> None:
    b = load_bundle(wires_bundle(tmp_path, x_scene(top=top)))
    got, ev = ontop(b, tid("s", "A"), tid("s", "B"), 0, CFG)
    assert got == top and ev["visible_frac"] >= 0.6


def test_geodesic_index(tmp_path: Path) -> None:
    b = load_bundle(wires_bundle(tmp_path, doubleback_scene()))
    assert ontop(b, tid("s", "A"), tid("s", "B"), 0, CFG)[0] == "B"
    assert ontop(b, tid("s", "A"), tid("s", "B"), 1, CFG)[0] == "A"


def test_occluder_over_crossing(tmp_path: Path) -> None:
    def extra(c, ids):  # type: ignore[no-untyped-def]
        o = c.new_object("fx/occluder:o", "occluder", (120, 120, 120))
        c.disk(o, (510.0, 500.0), 18)

    b = load_bundle(wires_bundle(tmp_path, x_scene(extra=extra)))
    assert len(crossings(b, tid("s", "A"), tid("s", "B"))) == 1
    with pytest.raises(KernelFault) as e:
        ontop(b, tid("s", "A"), tid("s", "B"), 0, CFG)
    assert e.value.code is ErrorCode.E_OCCLUDED_CROSSING


def test_out_of_range_index(tmp_path: Path) -> None:
    b = load_bundle(wires_bundle(tmp_path, x_scene()))
    with pytest.raises(KernelFault) as e:
        ontop(b, tid("s", "A"), tid("s", "B"), 1, CFG)
    assert e.value.code is ErrorCode.E_CLAIM_MISMATCH
