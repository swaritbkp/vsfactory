"""Cross-OS determinism pin: one fixed seed per plugin, committed pixel digests.

Raster bits are exact per platform but not across platforms (SPEC D-35), so the committed pin holds:

* ``portable`` -- manifest object names, claims digest, canonical answer and a coarse luminance grid
  of the consumption image. Checked on every platform (the grid within `GRID_TOLERANCE` levels).
* ``platforms[<tag>]`` -- exact decoded-pixel digests of beauty, id buffer, masks and the
  consumption image, recorded per platform. Checked bit-exactly wherever that platform has a record;
  a platform without one still runs the portable checks and a render-twice determinism check.

The consumption side length defaults to the kernel's 768 px and is configurable with
``KNOTWORK_CONSUMPTION_PX``. Record the current platform with ``KNOTWORK_UPDATE_PINS=1``.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image

from knotwork.knotcore.contracts import CONSUMPTION_PX
from knotwork.knotcore.seeds import SeedTree, array_digest, json_digest
from knotwork.pipeline.io import file_digests, write_claims
from knotwork.plugins.registry import NAMES, get_plugin

REPO = Path(__file__).resolve().parents[2]
PINS = REPO / "tests" / "golden" / "cross_platform.json"
PROFILE: dict[str, Any] = yaml.safe_load(
    (REPO / "configs" / "default_profile.yaml").read_text(encoding="utf-8")
)
PIN_SEED = 20261126
GRID = 16
GRID_TOLERANCE = 3
LABEL_PX = int(os.environ.get("KNOTWORK_CONSUMPTION_PX", str(CONSUMPTION_PX)))


def platform_tag() -> str:
    return f"{sys.platform}-{platform.machine().lower()}"


def render_pin(name: str, out: Path) -> dict[str, Any]:
    """Render the pinned scene of plugin `name` into `out` and describe it."""
    plugin = get_plugin(name)
    family = plugin.families[0]
    at = SeedTree(PIN_SEED).child(name, "pin")
    scene = plugin.sample_scene(at.child("scene"), dict(PROFILE, _family=family.value))
    question = plugin.questions(scene, at.child("question"), family)
    manifest = plugin.render(scene, out)
    write_claims(out, plugin.claims(scene))
    beauty = Image.open(out / "beauty.png").convert("RGB")
    small = beauty.resize((LABEL_PX, LABEL_PX), Image.Resampling.LANCZOS)
    grid = small.convert("L").resize((GRID, GRID), Image.Resampling.BOX)
    files = file_digests(out)
    return {
        "portable": {
            "objects": json_digest([o.name for o in manifest.objects]),
            "claims": files["claims.json"],
            "answer": plugin.symbolic_answer(scene, question).canonical,
            "grid": np.asarray(grid, dtype=np.int64).ravel().tolist(),
        },
        "exact": {
            "beauty": files["beauty.png"],
            "id_buffer": files["id_buffer.png"],
            "masks": json_digest({k: v for k, v in files.items() if k.startswith("masks/")}),
            f"consumption_{LABEL_PX}": array_digest(np.asarray(small, dtype=np.uint8)),
        },
    }


def _load() -> dict[str, Any]:
    return json.loads(PINS.read_text(encoding="utf-8")) if PINS.is_file() else {}


@pytest.mark.parametrize("name", NAMES)
def test_pinned_render(name: str, tmp_path: Path) -> None:
    got = render_pin(name, tmp_path / "a")
    pins = _load()
    if os.environ.get("KNOTWORK_UPDATE_PINS") == "1":
        row = pins.setdefault(name, {})
        row["portable"] = got["portable"]
        row.setdefault("platforms", {})[platform_tag()] = got["exact"]
        PINS.write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return
    assert name in pins, f"no pin for {name}; record with KNOTWORK_UPDATE_PINS=1"
    want = pins[name]
    for key in ("objects", "claims", "answer"):
        assert got["portable"][key] == want["portable"][key], key
    drift = np.abs(np.subtract(got["portable"]["grid"], want["portable"]["grid"]))
    assert int(drift.max()) <= GRID_TOLERANCE, f"coarse pixels moved by {int(drift.max())} levels"
    exact = want["platforms"].get(platform_tag())
    if exact is None:
        assert render_pin(name, tmp_path / "b") == got, "render is not deterministic"
        return
    for key, digest in got["exact"].items():
        if key in exact:
            assert digest == exact[key], f"{name} {key} differs on {platform_tag()}"


def test_pin_file_covers_every_plugin_and_this_repository_platform() -> None:
    pins = _load()
    assert set(pins) == set(NAMES)
    for name, row in pins.items():
        assert len(row["portable"]["grid"]) == GRID * GRID, name
        assert row["platforms"], name
