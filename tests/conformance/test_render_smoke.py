"""Task 0.3: the deterministic layered renderer (SPEC.md section 6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from knotwork.knotcore.contracts import PixelBox, RenderManifest, TerminalEntry
from knotwork.knotcore.seeds import array_digest
from knotwork.render.canvas import LayeredCanvas, RenderError

REPO = Path(__file__).resolve().parents[2]
FONT = REPO / "assets" / "fonts" / "DejaVuSans.ttf"
PROFILE = yaml.safe_load((REPO / "configs" / "default_profile.yaml").read_text(encoding="utf-8"))
SHA = PROFILE["render"]["font_sha256"]
BG = (246, 244, 238)


def _scene(out: Path) -> tuple[LayeredCanvas, RenderManifest, dict[str, int]]:
    c = LayeredCanvas(1024, 2, BG, FONT, SHA)
    a = c.new_object("smoke/strand:a", "strand", (0, 114, 178))
    b = c.new_object("smoke/strand:b", "strand", (213, 94, 0))
    lab = c.new_object("smoke/label:a", "label", (20, 20, 20), label_text="A")
    xs = np.linspace(100, 900, 81)
    pa = np.stack([xs, 300 + (xs - 100) * 0.5], axis=1)
    pb = np.stack([xs, 700 - (xs - 100) * 0.5], axis=1)
    c.polyline(a, pa, 9, shade="harness")
    c.polyline(b, pb, 9)
    c.text(lab, (70, 300), "A", 26)
    c.halo_patch(a, pa, 9, 5, 22, (500.0, 500.0))
    term = TerminalEntry(
        terminal_id="smoke/terminal:a",
        role="start",
        box=PixelBox(x0=92, y0=290, x1=112, y1=310),
        label_obj=lab,
    )
    man = c.finish(out, "smoke", terminals=[term])
    return c, man, {"a": a, "b": b, "label": lab}


@pytest.fixture(scope="module")
def rendered(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, RenderManifest, dict[str, int]]:
    out = tmp_path_factory.mktemp("smoke")
    _, man, ids = _scene(out)
    return out, man, ids


def test_beauty_rgb_1024_supersampled(
    rendered: tuple[Path, RenderManifest, dict[str, int]],
) -> None:
    out, man, _ = rendered
    img = Image.open(out / "beauty.png")
    assert img.mode == "RGB" and img.size == (1024, 1024)
    arr = np.asarray(img)
    assert arr.dtype == np.uint8
    assert man.supersample == 2
    # Anti-aliasing from the 2x Lanczos downsample produces intermediate colours.
    assert len(np.unique(arr.reshape(-1, 3), axis=0)) > 20


def test_id_buffer_uint16_values(rendered: tuple[Path, RenderManifest, dict[str, int]]) -> None:
    out, man, _ = rendered
    arr = np.asarray(Image.open(out / "id_buffer.png"), dtype=np.uint16)
    assert arr.dtype == np.uint16 and arr.shape == (1024, 1024)
    assert set(np.unique(arr).tolist()) <= {0} | {o.obj_id for o in man.objects}


def test_masks_binary_and_text_non_aa(
    rendered: tuple[Path, RenderManifest, dict[str, int]],
) -> None:
    out, man, ids = rendered
    for o in man.objects:
        m = np.asarray(Image.open(out / o.mask))
        assert m.dtype == np.uint8 and set(np.unique(m).tolist()) <= {0, 255}
    text = np.asarray(Image.open(out / man.obj(ids["label"]).mask))
    assert (text == 255).sum() > 20 and set(np.unique(text).tolist()) == {0, 255}
    assert man.obj(ids["label"]).box is not None


def test_render_is_deterministic(tmp_path: Path) -> None:
    c1, _, _ = _scene(tmp_path / "x")
    c2, _, _ = _scene(tmp_path / "y")
    assert array_digest(c1.beauty_array()) == array_digest(c2.beauty_array())
    assert array_digest(c1.id_array()) == array_digest(c2.id_array())
    for a, b in zip(
        sorted((tmp_path / "x").rglob("*.png")),
        sorted((tmp_path / "y").rglob("*.png")),
        strict=True,
    ):
        assert array_digest(np.asarray(Image.open(a))) == array_digest(np.asarray(Image.open(b)))


def test_halo_patch_layering(rendered: tuple[Path, RenderManifest, dict[str, int]]) -> None:
    out, man, ids = rendered
    idb = np.asarray(Image.open(out / "id_buffer.png"), dtype=np.uint16)
    assert idb[500, 500] == ids["a"]
    mb = np.asarray(Image.open(out / man.obj(ids["b"]).mask)) > 0
    # Under-wire b near the crossing but outside a's stroke: halo pixels are background (0).
    ys, xs = np.nonzero(mb)
    d = np.hypot(xs - 500, ys - 500)
    # perpendicular distance to wire a (through (500, 500), direction (1, 0.5))
    perp = np.abs((xs - 500) * 0.5 - (ys - 500)) / np.hypot(1.0, 0.5)
    near = (d < 20) & (perp > 5.5) & (perp < 8.5)
    assert near.sum() > 10
    assert np.all(idb[ys[near], xs[near]] != ids["b"])
    # Amodal mask of b stays one connected piece.
    from scipy import ndimage

    _, n = ndimage.label(mb, structure=np.ones((3, 3)))
    assert n == 1
    beauty = np.asarray(Image.open(out / "beauty.png")).astype(int)
    halo_px = beauty[ys[near], xs[near]]
    assert np.abs(halo_px - np.array(BG)).max(axis=1).mean() < 40


def test_font_mismatch_raises_e_font() -> None:
    with pytest.raises(RenderError, match="E_FONT"):
        LayeredCanvas(1024, 2, BG, FONT, "0" * 64)


def test_finish_returns_valid_manifest(
    rendered: tuple[Path, RenderManifest, dict[str, int]],
) -> None:
    out, man, _ = rendered
    assert isinstance(man, RenderManifest)
    reloaded = RenderManifest.model_validate_json(
        (out / "manifest.json").read_text(encoding="utf-8")
    )
    assert reloaded == man
    assert man.font_sha256 == SHA
