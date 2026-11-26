"""Task 2.12: contact sheet with Pixel-Witness Proof strips (SPEC 9.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from knotwork.pipeline.contact_sheet import build
from knotwork.pipeline.io import read_json


def test_contact_sheet(
    datasets: Path, mutate_run: dict[str, Any], profile_path: Path, tmp_path: Path
) -> None:
    out = tmp_path / "sheet.png"
    build(datasets, out, profile_path, fault_report=mutate_run["path"])
    w, h = Image.open(out).size
    assert w >= 2048 and h >= 1536
    meta = read_json(out.with_suffix(".json"))
    rows = {r["name"]: r for r in meta["rows"]}
    for plugin in ("tangle", "gauge", "panel"):
        assert [p["kind"] for p in rows[plugin]["panels"]][:3] == ["base", "causal", "nuisance"]
    for name in ("pixel_witness:tangle", "pixel_witness:panel"):
        assert [p["kind"] for p in rows[name]["panels"]] == [
            "beauty",
            "id_buffer",
            "kernel_overlay",
        ]
        assert all(p["bundle_id"] and p["caption"] for p in rows[name]["panels"])
    faults = {f["fault"]: f["code"] for f in meta["faults"]}
    assert set(faults) == {f"M{i}" for i in range(1, 11)}
    assert faults["M1"] == "E_OOB" and faults["M9"] == "E_NEEDLE_OFFCENTER"
