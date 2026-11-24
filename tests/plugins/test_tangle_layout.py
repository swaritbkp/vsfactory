"""Task 2.2: tangle layouts + shapely validation (SPEC 8.1)."""

from __future__ import annotations

from typing import Any

from shapely.geometry import LineString, Point

from knotwork.knotcore.contracts import Family
from knotwork.knotcore.seeds import SeedTree
from knotwork.plugins.tangle import PLUGIN, braid, geom
from knotwork.plugins.tangle.plugin import lay_of, word_of


def test_200_scenes_fold_free_with_exact_intersections(profile: dict[str, Any]) -> None:
    kinds = set()
    for i in range(200):
        scene = PLUGIN.sample_scene(SeedTree(11).child("layout", i), profile)
        pay = scene.payload
        lay = lay_of(pay)
        kinds.add(pay["layout"]["kind"])
        lines = [LineString(p) for p in lay.paths]
        assert all(line.is_simple for line in lines)
        n, word = pay["n"], word_of(pay)
        for a in range(n):
            for b in range(a + 1, n):
                got = len(geom._points(lines[a].intersection(lines[b])))
                assert got == len(braid.pair_crossings(n, word, a, b))
        for c in lay.crossings:
            assert geom.crossing_angle(lines[c["a"]], lines[c["b"]], Point(c["center"])) >= 35.0
    assert kinds == {"strip", "horseshoe"}


def test_d2_agrees_with_d1(profile: dict[str, Any]) -> None:
    fams = [Family.T1_TRACE, Family.T2_CROSS, Family.T3_ONTOP]
    for i in range(45):
        fam = fams[i % 3]
        at = SeedTree(12).child("d2", i)
        scene = PLUGIN.sample_scene(at.child("scene"), dict(profile, _family=fam.value))
        q = PLUGIN.questions(scene, at.child("question"), fam)
        d2 = PLUGIN.geometric_answer(scene, q)
        assert d2 is not None and d2.canonical == PLUGIN.symbolic_answer(scene, q).canonical
