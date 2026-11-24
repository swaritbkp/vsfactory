"""Task 1.4: CROSS with geodesic ordering (SPEC 7.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fixtures import doubleback_scene, tid, wave_scene, wires_bundle

from knotwork.knotcore.io import Bundle, load_bundle
from knotwork.knotcore.topo import cross


@pytest.mark.parametrize("n", [0, 3])
def test_wave_counts(tmp_path: Path, n: int) -> None:
    b = load_bundle(wires_bundle(tmp_path, wave_scene(n)))
    count, ev = cross(b, tid("s", "A"), tid("s", "B"))
    assert count == n
    if n:
        xs = [c["center"][0] for c in ev["crossings"]]
        assert xs == sorted(xs)  # along A from its left start
        assert [round(x / 10) for x in xs] == [27, 51, 75]


def test_one_crossing_and_star(x_bundle: Bundle) -> None:
    assert cross(x_bundle, tid("s", "A"), tid("s", "B"))[0] == 1
    assert cross(x_bundle, tid("s", "A"), "*")[0] == 1
    assert cross(x_bundle, tid("s", "B"), "*")[0] == 1


def test_star_sums_over_strands(tmp_path: Path) -> None:
    from fixtures import ORANGE, Scene, Wire, polyline

    sc = wave_scene(3)
    sc.wires.append(Wire("C", "3", polyline(150, 870, lambda x: 0 * x + 800), ORANGE))
    ys = np.linspace(120, 760, 161)
    sc.wires.append(Wire("D", "4", np.stack([np.full_like(ys, 600.0), ys], axis=1), ORANGE))
    b = load_bundle(wires_bundle(tmp_path, Scene(wires=sc.wires, crossings=sc.crossings)))
    assert cross(b, tid("s", "A"), tid("s", "B"))[0] == 3
    assert cross(b, tid("s", "A"), "*")[0] == 4  # 3 with B + 1 with D
    assert cross(b, tid("s", "C"), "*")[0] == 0


def test_geodesic_order_differs_from_x_order(tmp_path: Path) -> None:
    b = load_bundle(wires_bundle(tmp_path, doubleback_scene()))
    count, ev = cross(b, tid("s", "A"), tid("s", "B"))
    assert count == 2
    xs = [c["center"][0] for c in ev["crossings"]]
    geo = [c["geo"] for c in ev["crossings"]]
    assert geo == sorted(geo)
    assert xs[0] > xs[1]  # first crossing along A is the right-hand one
