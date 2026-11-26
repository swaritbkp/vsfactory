"""Gauge TaskPlugin: one analog dial per scene (SPEC 8.2). Untrusted."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np

from knotwork.knotcore.contracts import (
    Answer,
    AnswerType,
    Claim,
    ClaimSet,
    DifficultySpec,
    Family,
    FaultCode,
    Op,
    QuestionProgram,
    RenderManifest,
    SceneIR,
)
from knotwork.knotcore.seeds import SeedTree
from knotwork.plugins.difficulty import adjust_profile, finalize_scene, restress, with_difficulty
from knotwork.plugins.gauge import geom
from knotwork.plugins.gauge.render import M8_SWEEP, M9_SHIFT, bezel_width, draw_dial
from knotwork.plugins.gauge.scale import ZONES, admissible, dial_zone, sample_zones
from knotwork.plugins.gauge.stem import draw_plate, draw_stem, sample_plate, sample_stem
from knotwork.render import realism
from knotwork.render.canvas import make_canvas, render_cfg

#: metal bezel finishes: chrome, black anodized, brass, gunmetal
BEZELS = [(178, 182, 188), (62, 64, 70), (182, 156, 100), (114, 118, 126)]
#: aged-enamel / off-white dial faces (light faces keep dark numerals OCR- and template-readable)
FACES = [(250, 249, 244), (246, 241, 226), (241, 238, 229), (252, 252, 250)]
#: instrument housing / wall tones, light to mid-dark (the dial face carries every read-out)
TINTS = [(214, 216, 218), (206, 210, 214), (220, 216, 206), (210, 214, 208),
         (156, 160, 164), (128, 134, 126), (104, 108, 116), (180, 172, 152)]  # fmt: skip
UNITS = ["bar", "psi", "kPa", "MPa", "°C"]
GID = "gauge/gauge:g1"
CANVAS = 1024.0
DIAL_MARGIN_PX = 40.0  # bezel clearance from the canvas edge (outside the 24-px margin band)
PROMPTS = {
    Family.G1_READ: "What value does the gauge needle point to?",
    Family.G2_ZONE: "Is the gauge needle in the green, amber, or red zone?",
}
FORMAT = {AnswerType.FLOAT: "ANSWER: <number>", AnswerType.ZONE: "ANSWER: <green|amber|red>"}


class TwinUnavailable(RuntimeError):
    """No admissible twin edit (the pipeline retries another attempt)."""


def float_answer(v: float) -> Answer:
    return Answer(
        type=AnswerType.FLOAT, value=float(v), tolerance=0.0, canonical=format(float(v), "g")
    )


def zone_answer(z: str) -> Answer:
    return Answer(type=AnswerType.ZONE, value=z, canonical=z)


def values_for(d: dict[str, Any], tol_units: float) -> list[int]:
    return admissible(float(d["vmin"]), float(d["vmax"]), tuple(d["zones"]), 3.0 * tol_units,
                      0.0)  # fmt: skip


def pick_value(rng: np.random.Generator, d: dict[str, Any], tol_units: float, zone: str | None = None,
               away_from: float | None = None, min_move: float = 0.0) -> int | None:  # fmt: skip
    """Admissible integer value (optionally in ``zone``, and >= min_move from ``away_from``)."""
    vals = [v for v in values_for(d, tol_units) if zone is None or dial_zone(d, v) == zone]
    if away_from is not None:
        vals = [v for v in vals if abs(v - away_from) >= min_move]
    return int(vals[int(rng.integers(len(vals)))]) if vals else None


def ordinal(profile: dict[str, Any], k: int = 1) -> int | None:
    """This slot's index among same-family slots of the run (generator hint, D-34), or None."""
    o = profile.get("_ordinal")
    return None if o is None else int(o) // k


def pick_value_binned(
    rng: np.random.Generator,
    d: dict[str, Any],
    tol_units: float,
    bins: int = 10,
    decile: int | None = None,
) -> int:
    """Admissible value whose scale decile is uniform (D-28): a decile is drawn first, then a value
    inside it, so READ answers are balanced over equal-width bins of the printed scale even though
    the needle keeps clear of zone boundaries and scale ends."""
    lo, hi = float(d["vmin"]), float(d["vmax"])
    groups: list[list[int]] = [[] for _ in range(bins)]
    for v in values_for(d, tol_units):
        groups[min(bins - 1, int(bins * (v - lo) / (hi - lo)))].append(v)
    nonempty = [g for g in groups if g]
    if decile is not None and groups[decile % bins]:
        nonempty = [groups[decile % bins]]  # exact decile balance from the slot ordinal (D-34)
    g = nonempty[int(rng.integers(len(nonempty)))]
    return int(g[int(rng.integers(len(g)))])


def sample_dial(rng: np.random.Generator, prof: dict[str, Any], name: str, center: tuple[float, float],
                radius: float, zone: str | None, label_frac: float = 0.62) -> dict[str, Any]:  # fmt: skip
    """A dial dict with a random scale, zones, bezel and admissible needle value."""
    scales = prof["scales"]
    keys = sorted(prof["scale_mix"])
    w = np.array([prof["scale_mix"][k] for k in keys], dtype=np.float64)
    scale = keys[int(rng.choice(len(keys), p=w / w.sum()))]
    s = scales[scale]
    vmin, vmax = float(s["vmin"]), float(s["vmax"])
    for _ in range(50):
        d = {"name": name, "center": [float(center[0]), float(center[1])], "radius": float(radius), "scale": scale,
             "labels": list(s["labels"]), "vmin": vmin, "vmax": vmax, "zones": list(sample_zones(rng, vmin, vmax)),
             "bezel_rgb": list(BEZELS[int(rng.integers(len(BEZELS)))]), "band_inner": 0.92,
             "face_rgb": list(FACES[int(rng.integers(len(FACES)))]), "unit": UNITS[int(rng.integers(len(UNITS)))],
             "serial": f"No. {int(rng.integers(10000, 99999))}   CL 1.6",
             "minor_per_major": int(prof.get("minor_ticks_per_major", 9)),
             "surface": realism.sample_surface(rng, "powder"),
             "label_frac": label_frac}  # fmt: skip
        v = pick_value(rng, d, tol_units(prof, d), zone)
        if v is not None:
            d["value"] = float(v)
            return d
    raise RuntimeError("no admissible needle value")


def restyle_dial(d: dict[str, Any], rng: np.random.Generator) -> None:
    """New bezel finish, face ageing and dial surface (lighting / grime); geometry and value kept."""
    d["bezel_rgb"] = list(
        [b for b in BEZELS if list(b) != d["bezel_rgb"]][int(rng.integers(len(BEZELS) - 1))]
    )
    d["face_rgb"] = list(
        [f for f in FACES if list(f) != d.get("face_rgb")][int(rng.integers(len(FACES) - 1))]
    )
    d["surface"] = realism.sample_surface(rng, "powder")


def tol_units(prof: dict[str, Any], d: dict[str, Any]) -> float:
    return float(prof["grade_tolerance_frac"]) * (float(d["vmax"]) - float(d["vmin"]))


def kernel_tol_estimate(d: dict[str, Any]) -> float:
    """Upper estimate of the kernel's resolution tolerance (SPEC 7.5 step 6) for this dial."""
    r = float(d["radius"])
    slope = (float(d["vmax"]) - float(d["vmin"])) / 4 / 67.5
    return 2.0 * slope * (57.3 / (0.95 * r) + 2 * 57.3 / (0.82 * r)) * 1.25


def _scene(scene: SceneIR, pay: dict[str, Any]) -> SceneIR:
    return SceneIR(plugin=scene.plugin, plugin_version=scene.plugin_version, seed_path=scene.seed_path,
                   params=dict(scene.params), payload=pay)  # fmt: skip


class GaugePlugin:
    name = "gauge"
    version = "1.0.0"
    families = (Family.G1_READ, Family.G2_ZONE)

    def sample_scene(
        self, seeds: SeedTree, profile: dict[str, Any], difficulty: DifficultySpec | None = None
    ) -> SceneIR:
        if difficulty is not None:
            profile = adjust_profile(self.name, profile, difficulty)
        return finalize_scene(self._sample(seeds, profile), difficulty)

    def _sample(self, seeds: SeedTree, profile: dict[str, Any]) -> SceneIR:
        """``profile["_family"] == "G2_ZONE"`` samples the needle zone uniformly (balanced answers)."""
        prof, rng = profile["gauge"], seeds.rng()
        draw = int(rng.integers(3))
        o = ordinal(profile)
        zone = (
            str(ZONES[draw if o is None else o % 3])
            if profile.get("_family") == Family.G2_ZONE.value
            else None
        )
        radius = float(rng.uniform(prof["radius_px"][0], prof["radius_px"][1]))
        # the dial floats on the housing (D-27): a fixed centre made every scene a pHash near-duplicate
        reach = radius + bezel_width(radius) + DIAL_MARGIN_PX
        cx, cy = (float(rng.uniform(reach, CANVAS - reach)) for _ in range(2))
        # numerals move toward the hub on small dials (as on panel dials) so their plates stay
        # clear of the needle tip and the tick band (D-27)
        frac = float(np.clip(0.47 + (radius - 120.0) * 0.15 / 140.0, 0.47, 0.62))
        d = sample_dial(rng, prof, "g1", (cx, cy), radius, zone, frac)
        if profile.get("_family") == Family.G1_READ.value:
            d["value"] = float(
                pick_value_binned(rng, d, tol_units(prof, d), decile=ordinal(profile))
            )
        pay: dict[str, Any] = {
            "dial": d,
            "render": render_cfg(profile),
            "grade_tol": tol_units(prof, d),
        }
        pay["render"]["background_rgb"] = list(TINTS[int(rng.integers(len(TINTS)))])
        pay["surface"] = realism.sample_surface(rng, "brushed")
        pay["stem"] = sample_stem(rng, d, radius + bezel_width(radius), CANVAS)
        pay["plate"] = sample_plate(rng, d, radius + bezel_width(radius), CANVAS)
        pay["relief"] = realism.relief_panels(rng)
        return SceneIR(plugin=self.name, plugin_version=self.version, seed_path=seeds.path_str,
                       params={"scale": d["scale"], "radius": radius}, payload=pay)  # fmt: skip

    def render(
        self, scene: SceneIR, out_dir: Path, fault: FaultCode | None = None
    ) -> RenderManifest:
        c = make_canvas(scene.payload["render"])
        realism.draw_relief(c, scene.payload.get("relief", []))
        draw_plate(c, "gauge", scene.payload.get("plate"))
        realism.corner_screws(c, "gauge")
        if "stem" in scene.payload:
            draw_stem(c, "gauge", scene.payload["stem"])
        g = draw_dial(c, "gauge", scene.payload["dial"], fault)
        realism.add_surface(c, scene.payload["surface"])
        return c.finish(out_dir, self.name, gauges=[g])

    def claims(self, scene: SceneIR) -> ClaimSet:
        d = scene.payload["dial"]
        return ClaimSet(plugin=self.name, claims=[
            Claim(claim_id="read_g1", op=Op.READ, args=[GID], expected=float_answer(d["value"])),
            Claim(claim_id="zone_g1", op=Op.ZONE, args=[GID], expected=zone_answer(dial_zone(d))),
        ])  # fmt: skip

    def questions(self, scene: SceneIR, seeds: SeedTree, family: Family) -> QuestionProgram:
        d = scene.payload["dial"]
        at = AnswerType.FLOAT if family is Family.G1_READ else AnswerType.ZONE
        op = "READ" if family is Family.G1_READ else "ZONE"
        text = f"{PROMPTS[family]}\nEnd your reply with a final line of the form '{FORMAT[at]}'."
        span = float(d["vmax"]) - float(d["vmin"])
        question = QuestionProgram(program_id=f"gauge/{family.value}/{seeds.digest()[:16]}", family=family,
                               steps=[{"step_id": "s0", "op": op, "args": [GID]}], text=text, answer_type=at,  # type: ignore[list-item]
                               options=list(ZONES) if at is AnswerType.ZONE else None,
                               difficulty={"span": span, "radius": float(d["radius"])})  # fmt: skip
        return with_difficulty(question, scene.payload)

    def symbolic_answer(self, scene: SceneIR, program: QuestionProgram) -> Answer:
        d = scene.payload["dial"]
        if program.family is Family.G1_READ:
            return float_answer(d["value"])
        return zone_answer(dial_zone(d))

    def geometric_answer(self, scene: SceneIR, program: QuestionProgram) -> Answer | None:
        d = scene.payload["dial"]
        if program.family is Family.G1_READ:
            return float_answer(geom.geometric_reading(d))
        return zone_answer(geom.geometric_zone(d))

    def causal_twin(
        self, scene: SceneIR, program: QuestionProgram, seeds: SeedTree
    ) -> tuple[SceneIR, str]:
        pay, rng = copy.deepcopy(scene.payload), seeds.rng()
        d, tol = pay["dial"], float(pay["grade_tol"])
        old = float(d["value"])
        if program.family is Family.G1_READ:
            v = pick_value(rng, d, tol, away_from=old, min_move=5 * tol)
        else:
            here = dial_zone(d, old)
            other = [z for z in ZONES if z != here][int(rng.integers(2))]
            v = pick_value(rng, d, tol, zone=other)
        if v is None:
            raise TwinUnavailable("no admissible causal needle value")
        d["value"] = float(v)
        return _scene(scene, pay), f"needle {old:g} -> {v}"

    def nuisance_twin(self, scene: SceneIR, program: QuestionProgram, seeds: SeedTree) -> SceneIR:
        """Domain shift with the answer kept: bezel finish, dial ageing, housing tone, zone band width,
        lighting direction, grime pattern and surface finish all resampled."""
        pay, rng = copy.deepcopy(scene.payload), seeds.rng()
        restyle_dial(pay["dial"], rng)
        tints = [t for t in TINTS if list(t) != pay["render"]["background_rgb"]]
        pay["render"]["background_rgb"] = list(tints[int(rng.integers(len(tints)))])
        pay["surface"] = realism.sample_surface(rng, "brushed")
        pay["dial"]["band_inner"] = [0.90, 0.94][int(rng.integers(2))]
        restress(self.name, pay)
        return _scene(scene, pay)


def fault_effective(scene: SceneIR, fault: FaultCode) -> bool:
    d = scene.payload["dial"]
    vmin, vmax, v = float(d["vmin"]), float(d["vmax"]), float(d["value"])
    if fault is FaultCode.M8:
        drawn = vmin + (v - vmin) * M8_SWEEP
        return drawn <= vmax - 1.0 and abs(drawn - v) > 3.0 * kernel_tol_estimate(d)
    if fault is FaultCode.M9:
        deg = 225.0 - (v - vmin) / (vmax - vmin) * 270.0
        ux, uy = math.cos(math.radians(deg)), -math.sin(math.radians(deg))
        perp = abs(M9_SHIFT[0] * uy - M9_SHIFT[1] * ux)
        return perp > max(3.0, 0.03 * float(d["radius"])) + 1.5
    return False


PLUGIN = GaugePlugin()
