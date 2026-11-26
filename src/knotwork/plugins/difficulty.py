"""Orthogonal difficulty control: reasoning load, visual stress and occlusion are separate dials.

Each dial is optional and in [0, 1]; a dial left at None keeps the profile's own sampling ranges, so
a call without difficulty samples exactly as before.

| dial            | tangle                          | gauge                       | panel          |
|-----------------|---------------------------------|-----------------------------|----------------|
| reasoning_load  | strands, braid length, targets  | offset scale, fewer ticks   | number of      |
|                 |                                 |                             | gauges         |
| visual_stress   | stroke, jitter, colour regime,  | dial radius, surface, noise | colour regime, |
|                 | surface, noise, resolution      | and resolution              | surface, noise |
| occlusion_level | distractor crossing pairs       | smudges and dust            | distractor     |
|                 |                                 |                             | cables         |

`stroke_px`, `jitter_px`, `realism`, `noise`, `dpi_scale` and `color_regime` override one knob of the
visual dial. The panel's stroke is fixed by its port geometry and ignores `stroke_px`.
"""

from __future__ import annotations

import copy
from typing import Any

from knotwork.knotcore.contracts import DifficultySpec, QuestionProgram, SceneIR
from knotwork.plugins.payloads import validate_payload


def _pin(bounds: list[float], t: float) -> list[int]:
    """The single integer a dial position `t` selects inside [lo, hi], as a degenerate range."""
    lo, hi = float(bounds[0]), float(bounds[1])
    value = int(round(lo + t * (hi - lo)))
    return [value, value]


def _regime(spec: DifficultySpec) -> str | None:
    if spec.color_regime is not None:
        return spec.color_regime
    t = spec.visual_stress
    if t is None:
        return None
    return "unique" if t < 0.34 else ("partial" if t < 0.67 else "mono")


def adjust_profile(plugin: str, profile: dict[str, Any], spec: DifficultySpec) -> dict[str, Any]:
    """A copy of the generation profile with the ranges the dials select."""
    out = dict(profile)
    own = out[plugin] = copy.deepcopy(profile[plugin])
    load, stress, hidden = spec.reasoning_load, spec.visual_stress, spec.occlusion_level
    regime = _regime(spec)
    if regime is not None and plugin in ("tangle", "panel"):
        tangle = out["tangle"] = copy.deepcopy(out["tangle"])
        tangle["color_regimes"] = {regime: 1.0}
        tangle["benchmark_color_regimes"] = [regime]
        own = out[plugin]
    if plugin == "tangle":
        if load is not None:
            own["strands"] = _pin(own["strands"], load)
            centre = _pin(own["braid_length"], load)[0]
            lo, hi = profile["tangle"]["braid_length"]
            own["braid_length"] = [max(lo, centre - 1), min(hi, centre + 1)]
            top = int(profile["tangle"]["t2_target_count"][1])
            own["t2_target_count"] = [0, min(top, 2 + int(round(load * (top - 2))))]
        if spec.stroke_px is not None:
            own["stroke_px"] = [int(round(spec.stroke_px))] * 2
        elif stress is not None:
            own["stroke_px"] = _pin(own["stroke_px"], 1.0 - stress)
        if spec.jitter_px is not None:
            own["jitter_amp_px"] = [spec.jitter_px, spec.jitter_px]
        elif stress is not None:
            own["jitter_amp_px"] = _pin(profile["tangle"]["jitter_amp_px"], stress)
        if hidden is not None:
            own["r2_distractor_pairs"] = _pin(own["r2_distractor_pairs"], hidden)
    elif plugin == "gauge":
        if load is not None:
            scales = sorted(own["scales"])
            want = "offset" if load >= 0.5 and "offset" in scales else scales[0]
            own["scale_mix"] = {want: 1.0}
            own["minor_ticks_per_major"] = 9 if load < 0.34 else (4 if load < 0.67 else 1)
        if stress is not None:
            own["radius_px"] = _pin(own["radius_px"], 1.0 - stress)
    elif plugin == "panel":
        if load is not None:
            own["gauges"] = _pin(own["gauges"], load)
        if hidden is not None:
            own["extra_strands"] = _pin(own["extra_strands"], hidden)
    return out


def _stress_surface(surface: dict[str, Any], plugin: str, spec: DifficultySpec) -> None:
    stress = spec.visual_stress
    realism = spec.realism if spec.realism is not None else stress
    if realism is not None:
        k = 2.0 * realism  # 0.5 keeps the sampled finish, 0 removes it, 1 doubles it
        for key in ("light", "grain", "brush"):
            surface[key] = float(surface[key]) * k
        for key in ("smudges", "dust"):
            surface[key] = int(round(surface[key] * k))
    if plugin == "gauge" and spec.occlusion_level is not None:
        surface["smudges"] = int(round(8 * spec.occlusion_level))
        surface["dust"] = int(round(120 * spec.occlusion_level))
    noise = spec.noise
    if noise is None and stress is not None:
        noise = max(0.0, stress - 0.5)
    if noise is not None:
        surface["noise"] = float(noise)
    dpi = spec.dpi_scale
    if dpi is None and stress is not None:
        dpi = 1.0 - max(0.0, stress - 0.7)
    if dpi is not None:
        surface["dpi_scale"] = float(dpi)


def restress(plugin: str, payload: dict[str, Any]) -> None:
    """Re-apply the recorded visual dials after a twin resampled the surfaces."""
    if not payload.get("difficulty_spec"):
        return
    spec = DifficultySpec.model_validate(payload["difficulty_spec"])
    surfaces = [payload["surface"]]
    surfaces += [d["surface"] for d in payload.get("dials", [])]
    if "dial" in payload:
        surfaces.append(payload["dial"]["surface"])
    for s in surfaces:
        _stress_surface(s, plugin, spec)


def finalize_scene(scene: SceneIR, spec: DifficultySpec | None) -> SceneIR:
    """Apply the surface dials, record the spec, and validate the typed payload."""
    if spec is not None:
        scene.payload["difficulty_spec"] = spec.model_dump(exclude_none=True)
        restress(scene.plugin, scene.payload)
        scene.params["difficulty"] = scene.payload["difficulty_spec"]
    validate_payload(scene.plugin, scene.payload)
    return scene


def with_difficulty(program: QuestionProgram, payload: dict[str, Any]) -> QuestionProgram:
    """The program annotated with the dials its scene was sampled under (if any)."""
    if not payload.get("difficulty_spec"):
        return program
    spec = DifficultySpec.model_validate(payload["difficulty_spec"])
    return program.model_copy(update={"difficulty_spec": spec})
