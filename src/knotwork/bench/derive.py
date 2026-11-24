"""Derived benchmark items (SPEC 12.3) built from pool bundles and re-certified by the kernel:

* colour ablation -- the same T1 scene (same braid word, geometry, labels) re-rendered with unique
  cable colours instead of mono;
* twins -- causal and nuisance twins of a T1 base item (pipeline.twins.build_twin).

Scenes are re-created exactly from the dataset receipt root and the bundle seed path (checked against
the stored scene_id), so derived items are deterministic and replay keys reproduce.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from knotwork.diversity import parse_path, rebuild_scene
from knotwork.knotcore.contracts import Answer, Certificate, QuestionProgram
from knotwork.knotcore.seeds import SeedTree


def _root(ds: Path) -> int:
    return int(json.loads((ds / "freshness_receipt.json").read_text(encoding="utf-8"))["root_seed"])


def unique_variant(job: dict[str, Any]) -> dict[str, Any]:
    """Re-render a mono T1 bundle with unique colours; {"ok", "dir"}."""
    from knotwork.pipeline.certify import run_candidate
    from knotwork.plugins.registry import get_plugin
    from knotwork.plugins.tangle.plugin import PALETTE
    from knotwork.plugins.tangle.render import hex_rgb

    ds, bdir, profile = Path(job["ds"]), Path(job["dir"]), job["profile"]
    task = json.loads((bdir / "task.json").read_text(encoding="utf-8"))
    scene = rebuild_scene(ds, task, profile)
    pay = copy.deepcopy(scene.payload)
    rng = SeedTree(_root(ds)).child("bench", "ablation", task["bundle_id"]).rng()
    order = [PALETTE[int(i)] for i in rng.permutation(len(PALETTE))]
    pay["colors"] = [list(hex_rgb(order[i % len(order)])) for i in range(len(pay["colors"]))]
    pay["color_regime"] = "unique"
    variant = scene.model_copy(update={"payload": pay})
    q = QuestionProgram.model_validate(task["question"])
    cand = run_candidate(get_plugin(task["plugin"]), variant, q, Path(job["out"]), profile)
    ok = cand.certified and cand.answer.canonical == task["answer"]["canonical"]
    return {"ok": bool(ok), "dir": str(cand.dir), "base_id": task["bundle_id"]}


def twin_set(job: dict[str, Any]) -> dict[str, Any]:
    """Causal + nuisance twins of a T1 base bundle; {"ok", "causal", "nuisance"} (dirs)."""
    from knotwork.pipeline.certify import Candidate
    from knotwork.pipeline.twins import build_twin
    from knotwork.plugins.registry import get_plugin

    ds, bdir, profile = Path(job["ds"]), Path(job["dir"]), job["profile"]
    task = json.loads((bdir / "task.json").read_text(encoding="utf-8"))
    plugin = get_plugin(task["plugin"])
    scene = rebuild_scene(ds, task, profile)
    cert = Certificate.model_validate(task["certificate"])
    base = Candidate(dir=bdir, scene=scene, question=QuestionProgram.model_validate(task["question"]),
                     d1=cert.d1, d2=cert.d2, answer=Answer.model_validate(task["answer"]),
                     bundle_id=task["bundle_id"], certificate=cert, files=dict(task["files"]))  # fmt: skip
    at = SeedTree(_root(ds), parse_path(task["seed_path"]))
    out = Path(job["out"])
    res: dict[str, Any] = {"ok": False, "base_id": task["bundle_id"], "base": str(bdir)}
    causal = build_twin("causal", plugin, base, at, out, profile, task["family"])
    nuis = build_twin("nuisance", plugin, base, at, out, profile, task["family"])
    if causal.candidate is not None and nuis.candidate is not None:
        for name, c in (("causal", causal.candidate), ("nuisance", nuis.candidate)):
            res[name] = {"dir": str(c.dir), "bundle_id": c.bundle_id,
                         "answer": c.answer.model_dump(mode="json"),
                         "question": c.question.model_dump(mode="json")}  # fmt: skip
        res["ok"] = True
    return res


__all__ = ["twin_set", "unique_variant"]
