"""Twin builder (SPEC 8.1-8.4): causal twins flip the answer with a minimal pixel edit, nuisance twins
change style but keep the answer. Each twin is certified like any other candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from knotwork.knotcore.contracts import QuarantineRecord, QuestionProgram, SceneIR, TaskPlugin
from knotwork.knotcore.seeds import SeedTree
from knotwork.pipeline.certify import Candidate, quarantine_of, run_candidate, twin_quarantine

TWIN_RETRIES = 3


def pixel_diff_frac(a: Path, b: Path) -> float:
    """mean(any_c |beauty_a - beauty_b| > 8) over the 1024x1024 canvas."""
    x = np.asarray(Image.open(a / "beauty.png").convert("RGB"), dtype=np.int16)
    y = np.asarray(Image.open(b / "beauty.png").convert("RGB"), dtype=np.int16)
    return float((np.abs(x - y) > 8).any(axis=2).mean())


@dataclass
class TwinResult:
    candidate: Candidate | None
    edit: str | None = None
    diff: float | None = None
    quarantine: list[QuarantineRecord] = field(default_factory=list)
    n_candidates: int = 0


def twin_question(plugin: TaskPlugin, scene: SceneIR, base: QuestionProgram) -> QuestionProgram:
    """The twin's question: the base program unless the plugin re-derives it (premise truth)."""
    hook = getattr(plugin, "twin_question", None)
    return base if hook is None else hook(scene, base)


def build_twin(
    kind: str,
    plugin: TaskPlugin,
    base: Candidate,
    attempt: SeedTree,
    work: Path,
    profile: dict[str, Any],
    family_value: str,
) -> TwinResult:
    """Up to TWIN_RETRIES tries; causal: answer differs and diff < causal_max; nuisance: answer equal
    and diff >= nuisance_min. Unavailable edits and invalid twins become `twin` quarantine records."""
    from knotwork.plugins.registry import twin_unavailable

    unavailable = twin_unavailable(plugin.name)
    lim = profile["twins"]
    out = TwinResult(None)
    for t in range(TWIN_RETRIES):
        seeds = attempt.child(f"twin_{kind}", t)
        try:
            if kind == "causal":
                scene, edit = plugin.causal_twin(base.scene, base.question, seeds)
            else:
                scene, edit = (
                    plugin.nuisance_twin(base.scene, base.question, seeds),
                    "nuisance restyle",
                )
        except unavailable:  # no admissible edit: not a candidate, nothing to quarantine
            continue
        q = twin_question(plugin, scene, base.question)
        cand = run_candidate(plugin, scene, q, work / f"{kind}{t}", profile)
        out.n_candidates += 1
        if not cand.certified:
            out.quarantine.append(
                quarantine_of(
                    f"{seeds.path_str}", plugin.name, q.family, seeds.path_str, cand.certificate
                )
            )
            continue
        diff = pixel_diff_frac(base.dir, cand.dir)
        same = cand.answer.canonical == base.answer.canonical
        ok = (not same and diff < lim["causal_max_pixel_frac"]) if kind == "causal" else (
            same and diff >= lim["nuisance_min_pixel_frac"])  # fmt: skip
        if ok:
            return TwinResult(cand, edit, diff, out.quarantine, out.n_candidates)
        out.quarantine.append(twin_quarantine(seeds.path_str, plugin.name, q.family, seeds.path_str,
                                              f"{kind}: answer_same={same} pixel_diff_frac={diff:.4f}"))  # fmt: skip
    del family_value
    return out
