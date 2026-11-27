"""Rack patch-bay TaskPlugin: a dense loom of patch cables between a 2U switch (top bank) and a
patch panel (bottom bank). Untrusted plugin code, like every plugin.

The scene is a tangle braid scene (same braid words, layouts, claims and twins) whose strip layout
is drawn rotated by 90 degrees: strand positions become port columns and cables run from the top
bank down to the bottom bank. A transpose is an isometry, so the layout validation and the
plugin's geometric answer, computed in strip coordinates, describe exactly what is drawn. The
kernel never sees any of this: it re-derives the answer from the rendered masks and OCR'd port
numbers as for any tangle.

Chassis furniture (anodised bank bands, rack ears, screws, link LEDs, bank titles) is registered
as `decor` objects so its ink is explained to the witness; label strips, jack recesses and tray
rungs are beauty-only relief kept inside the surface-finish background tolerance (SPEC D-24 (a)).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from knotwork.knotcore.contracts import (
    DifficultySpec,
    Family,
    FaultCode,
    PixelBox,
    QuestionProgram,
    RenderManifest,
    SceneIR,
)
from knotwork.knotcore.seeds import SeedTree
from knotwork.plugins.difficulty import adjust_profile, finalize_scene
from knotwork.plugins.tangle import geom
from knotwork.plugins.tangle.plugin import TanglePlugin, lay_of
from knotwork.plugins.tangle.render import halo_mask, hex_rgb, render_tangle
from knotwork.render import realism
from knotwork.render.canvas import LayeredCanvas, make_canvas

RGB = tuple[int, int, int]

#: overrides of the tangle profile section: a denser, strip-only loom of 8-14 cables
RACK_DEFAULTS: dict[str, Any] = {
    "strands": [8, 14],
    "braid_length": [6, 16],
    "identity_slots": [2, 3],
    "r2_distractor_pairs": [0, 2],
    "layouts": {"strip": 1.0},
    "styles": {"harness": 0.6, "schematic": 0.4},
    "color_regimes": {"mono": 0.5, "partial": 0.3, "unique": 0.2},
    "benchmark_color_regimes": ["mono"],
    "stroke_px": [7, 9],
    "jitter_amp_px": [0, 8],
    "families": {"T1_TRACE": 0.6, "T2_CROSS": 0.4},
    "t2_target_count": [0, 6],
}
#: the strip is compressed along the cable direction to make room for the two port banks
SPAN_SCALE = 0.84
CENTRE = 512.0
RESAMPLES = 24
#: data-centre jacket colours; every set is CVD-separable and keeps luminance contrast on the tray
MONO_JACKETS = ("#2F2F2F", "#3A4350")
DUAL_JACKETS = ("#0072B2", "#1A1A1A")
MIXED_JACKETS = ("#0072B2", "#D55E00", "#1A1A1A", "#009E73", "#CC79A7")
TRAYS: tuple[RGB, ...] = ((214, 218, 222), (220, 222, 224), (210, 214, 218))
ANODISED: RGB = (38, 41, 46)
EAR: RGB = (54, 57, 63)
SILKSCREEN: RGB = (206, 210, 214)
LED_RGB: tuple[RGB, ...] = ((64, 206, 96), (242, 172, 38), (72, 76, 82))  # link, activity, off
LABEL_GAP = 41.0  # label centre distance from the terminal (tangle's gap for two glyphs)
BANKS = {
    "top": ("SWITCH-A", "48P GbE CORE  ·  U23-24"),
    "bottom": ("PATCH-B", "CAT6 KEYSTONE  ·  U21-22"),
}
PROMPTS = {
    Family.T1_TRACE: "The image shows a server-rack patch bay: switch ports SW-01, SW-02, ... on the "
    "top bank and patch-panel ports PP-01, PP-02, ... on the bottom bank. Trace the patch cable "
    "plugged into top switch port SW-{S}. Which bottom patch-bay port does it terminate in? "
    "Answer with the two-digit port number printed under it.",
    Family.T2_CROSS: "The image shows a server-rack patch bay. Follow the patch cable plugged into "
    "top switch port SW-{S} along its whole run down to the patch panel. How many times does it "
    "cross another patch cable?",
}
FORMAT = {
    Family.T1_TRACE: "ANSWER: <two-digit port number>",
    Family.T2_CROSS: "ANSWER: <integer>",
}


def rack_section(profile: dict[str, Any]) -> dict[str, Any]:
    """The tangle profile section with the rack overrides (the profile file is left untouched)."""
    own: dict[str, Any] = copy.deepcopy(profile["tangle"])
    own.update(copy.deepcopy(profile.get("rack_patch", RACK_DEFAULTS)))
    return own


def port_label(index: int) -> str:
    return f"{index + 1:02d}"


def _jackets(regime: str, n: int, rng: np.random.Generator) -> tuple[str, list[str]]:
    """(jacket style, one hex colour per cable) for the sampled colour regime."""
    if regime == "mono":
        return "mono", [MONO_JACKETS[int(rng.integers(len(MONO_JACKETS)))]] * n
    if regime == "partial":
        return "dual", [DUAL_JACKETS[int(rng.integers(2))] for _ in range(n)]
    order = [MIXED_JACKETS[int(i)] for i in rng.permutation(len(MIXED_JACKETS))]
    return "mixed", [order[k % len(order)] for k in range(n)]


def _transpose(lay: Any) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    paths = [np.ascontiguousarray(p[:, ::-1]) for p in lay.paths]
    crossings = [dict(x, center=[x["center"][1], x["center"][0]]) for x in lay.crossings]
    return paths, crossings


class RackPatchPlugin(TanglePlugin):
    name = "rack_patch"
    version = "1.0.0"
    families = (Family.T1_TRACE, Family.T2_CROSS)  # type: ignore[assignment]

    # ------------------------------------------------------------------ scene
    def sample_scene(
        self, seeds: SeedTree, profile: dict[str, Any], difficulty: DifficultySpec | None = None
    ) -> SceneIR:
        prof = dict(profile, tangle=rack_section(profile))
        if difficulty is not None:
            prof = adjust_profile("tangle", prof, difficulty)  # the tangle dials on rack ranges
        for k in range(RESAMPLES):
            at = seeds if k == 0 else seeds.child("rack_resample", k)
            scene = self._sample(at, prof)
            if self._rackify(scene.payload, at.child("rack").rng()):
                scene.params.update(jacket=scene.payload["jacket"], ports=scene.payload["n"])
                return finalize_scene(scene, difficulty)
        raise RuntimeError(f"no legible rack loom in {RESAMPLES} resamples ({seeds.path_str})")

    @staticmethod
    def _rackify(pay: dict[str, Any], rng: np.random.Generator) -> bool:
        """Turn a sampled tangle payload into a rack loom in place; False if it is not legible."""
        lay = pay["layout"]
        lay["x0"] = CENTRE - (CENTRE - float(lay["x0"])) * SPAN_SCALE
        lay["x1"] = CENTRE + (float(lay["x1"]) - CENTRE) * SPAN_SCALE
        n = int(pay["n"])
        if geom.validate(lay_of(pay), n, float(pay["stroke"])) is not None:
            return False
        pay["start_labels"] = [port_label(j) for j in range(n)]
        pay["end_labels"] = [port_label(j) for j in range(n)]
        jacket, colors = _jackets(str(pay["color_regime"]), n, rng)
        pay["jacket"] = jacket
        pay["colors"] = [list(hex_rgb(h)) for h in colors]
        if jacket == "mixed" and n > len(MIXED_JACKETS):
            pay["color_regime"] = "partial"  # colours repeat across the loom
        pay["render"]["background_rgb"] = list(TRAYS[int(rng.integers(len(TRAYS)))])
        pay["relief"] = []
        pay["leds"] = [
            [int(v) for v in rng.choice(3, size=n, p=[0.62, 0.28, 0.10])] for _ in range(2)
        ]
        return True

    def render(
        self, scene: SceneIR, out_dir: Path, fault: FaultCode | None = None
    ) -> RenderManifest:
        pay = scene.payload
        c = make_canvas(pay["render"])
        paths, crossings = _transpose(lay_of(pay))
        if not pay.get("plain"):
            draw_chassis(c, pay, paths)
        terms = render_tangle(c, pay, paths, crossings, fault)
        realism.add_surface(c, pay["surface"], ("label",), lambda cv: halo_mask(crossings, cv.size))
        return c.finish(out_dir, self.name, terminals=terms)

    # ------------------------------------------------------------------ questions
    def _question(self, scene: SceneIR, seeds: SeedTree, family: Family) -> QuestionProgram:
        if family not in PROMPTS:
            raise RuntimeError(f"rack_patch has no {family.value} questions")
        program = super()._question(scene, seeds, family)
        port = str(program.steps[0].args[0]).rsplit("_", 1)[1]
        text = PROMPTS[family].format(S=port)
        text += f"\nEnd your reply with a final line of the form '{FORMAT[family]}'."
        pid = program.program_id.replace("tangle/", f"{self.name}/", 1)
        return program.model_copy(update={"text": text, "program_id": pid})


# ---------------------------------------------------------------------- chassis furniture
def _shift(rgb: RGB, delta: int) -> RGB:
    r, g, b = (min(255, max(0, v + delta)) for v in rgb)
    return r, g, b


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _bank(
    c: LayeredCanvas, side: str, xs: list[float], term_y: float, leds: list[int], size: float
) -> None:
    """One port bank: anodised band with ears, screws, LEDs and silkscreen titles (decor objects),
    then the white label strip and the jack recesses (beauty-only relief)."""
    up = side == "top"
    sign = -1.0 if up else 1.0
    label_y = term_y + sign * LABEL_GAP
    edge = 18.0 if up else size - 18.0
    inner = label_y + sign * 33.0  # band edge facing the cables, clear of the label's OCR crop
    y0, y1 = sorted((edge, inner))
    left, right = 10.0, size - 10.0
    bg = c.background
    strip = _rect(72.0, label_y - 27.0, size - 72.0, label_y + 27.0)
    c.raw_beauty_polygon(strip, _shift(bg, 26))
    for x in xs:
        a, b = sorted((term_y - sign * 12.0, term_y + sign * 22.0))
        c.raw_beauty_polygon(_rect(x - 15.0, a, x + 15.0, b), _shift(bg, -28))
    band = c.new_object(f"rack/decor:band_{side}", "decor", ANODISED)
    c.set_box(band, PixelBox(x0=int(left), y0=int(y0) - 8, x1=int(right), y1=int(y1) + 8))
    c.polygon(band, _rect(left, y0, right, y1))
    for ex in (left, right - 56.0):
        c.polygon(band, _rect(ex, y0 - 8.0, ex + 56.0, y1 + 8.0), beauty_rgb=EAR)
    title, sub = BANKS[side]
    mid = (y0 + y1) / 2.0
    c.text(band, (86.0, mid - 9.0), title, 22, anchor="lm", beauty_rgb=SILKSCREEN)
    c.text(band, (86.0, mid + 15.0), sub, 13, anchor="lm", beauty_rgb=_shift(SILKSCREEN, -50))
    led_y = inner + sign * 10.0
    for x, state in zip(xs, leds, strict=True):
        c.disk(band, (x, led_y), 4.5, beauty_rgb=LED_RGB[state])
    for k, ex in enumerate((left + 28.0, right - 28.0)):
        for j, ey in enumerate((y0 + 12.0, y1 - 12.0)):
            realism.screw(c, "rack", f"{side}_ear{k}{j}", (ex, ey), 8.0)


def draw_chassis(c: LayeredCanvas, pay: dict[str, Any], paths: list[np.ndarray]) -> None:
    """Both banks and the cable-tray rungs between them, drawn before the cables."""
    n, size = int(pay["n"]), float(c.size)
    top_y = min(float(p[0, 1]) for p in paths)
    bottom_y = max(float(p[-1, 1]) for p in paths)
    bg = c.background
    for y in np.linspace(top_y + 90.0, bottom_y - 90.0, 5):
        c.raw_beauty_polygon(_rect(84.0, y - 2.0, size - 84.0, y + 2.0), _shift(bg, -20))
    tops = [float(paths[s][0, 0]) for s in range(n)]
    perm = pay["perm"]
    bottoms = [float(paths[perm[j]][-1, 0]) for j in range(n)]
    leds = pay.get("leds") or [[0] * n, [0] * n]
    _bank(c, "top", tops, top_y, list(leds[0]), size)
    _bank(c, "bottom", bottoms, bottom_y, list(leds[1]), size)


def fault_effective(scene: SceneIR, fault: FaultCode) -> bool:
    """The fault matrix runs on tangle, gauge and panel; the rack loom declares no fault hooks."""
    del scene, fault
    return False


PLUGIN = RackPatchPlugin()
