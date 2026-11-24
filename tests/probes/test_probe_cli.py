"""Task 3.4: `knot probe` writes a ProbeReport (SPEC 11.2) with GroupKFold-by-scene evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from knotwork import cli as K
from knotwork.probes.items import load_items
from knotwork.probes.run import fold_ids, run_probes

REQUIRED = {
    "tangle": {"straight_across", "nearest", "color_match", "gbdt_image_stats", "text_only"},
    "gauge": {"gbdt_image_stats", "text_only"},
    "panel": {"nearest", "color_match", "gbdt_image_stats", "text_only"},
}


def test_probe_report_schema(small_split: Path, tmp_path: Path) -> None:
    out = tmp_path / "probe.json"
    assert run_probes(small_split, out, seed=0, folds=5) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert set(rep) >= {"plugins", "balance", "dedup"}
    for plugin, names in REQUIRED.items():
        row = rep["plugins"][plugin]
        assert row["n"] == {"tangle": 12, "gauge": 10, "panel": 14}[plugin]
        assert 0.0 < row["pooled"]["chance"] < 1.0
        assert names <= set(row["pooled"]["probes"])
        for pr in row["pooled"]["probes"].values():
            assert {"acc", "n", "wilson_upper", "chance"} <= set(pr)
            assert 0.0 <= pr["acc"] <= pr["wilson_upper"] <= 1.0 and pr["n"] > 0
        for fam, frow in row["families"].items():
            assert fam[:1] in "TGP" and {"chance", "probes", "n"} <= set(frow)
        assert set(rep["dedup"][plugin]) == {"pairs_within_4", "dup_rate"}
    for v in rep["balance"].values():
        assert {"chi2", "dof", "p"} <= set(v)
    again = tmp_path / "probe2.json"
    assert run_probes(small_split, again, seed=0, folds=5) == 0
    assert again.read_bytes() == out.read_bytes()


def test_group_kfold_by_scene(small_split: Path) -> None:
    items = load_items(small_split / "tangle")
    f = fold_ids(items, 5)
    by_scene: dict[str, set[int]] = {}
    for it, k in zip(items, f, strict=True):
        by_scene.setdefault(it.scene_id, set()).add(int(k))
    assert all(len(v) == 1 for v in by_scene.values())
    assert len(set(int(x) for x in f)) == 5


def test_cli_probe(small_split: Path, tmp_path: Path) -> None:
    res = CliRunner().invoke(
        K.cli, ["probe", "--data", str(small_split / "gauge"), "--out", str(tmp_path / "p.json")]
    )
    assert res.exit_code == 0, res.output
    assert json.loads((tmp_path / "p.json").read_text())["plugins"]["gauge"]["n"] == 10
