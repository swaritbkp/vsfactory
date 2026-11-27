"""Closed loop from the arena to Forensics: a failed trial becomes a Diagnostic Twin Suite.

The suite has the shape `knot counterfactual` writes (`suites.json`), so the Forensics page shows it
like any other: the failed item and the model's reply beside a perceptual-clean, palette-swap and
step-down twin (each certified by the kernel) and a text-only twin. Arena items keep their sampled
scene (`scene.json`), so the twins are built from it directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from knotwork.arena.schemas import ItemRecord
from knotwork.knotcore.contracts import QuestionProgram, SceneIR
from knotwork.pipeline import counterfactual, text_ir
from knotwork.pipeline.io import read_json, write_json
from knotwork.studio import factory


def build_suite(item: ItemRecord, replies: dict[str, str], data_dir: Path, out: Path) -> Path:
    """Write `<out>/suites.json` for one failed item and return `out`."""
    bundle = data_dir / item.dir
    task = read_json(bundle / "task.json")
    scene_path = bundle / "scene.json"
    scene = SceneIR.model_validate(read_json(scene_path)) if scene_path.is_file() else None
    question = QuestionProgram.model_validate(task["question"])
    prof = factory.profile()
    suite: dict[str, Any] = {
        "bundle_id": item.bundle_id,
        "plugin": item.domain,
        "family": item.family,
        "question": question.text,
        "answer": task["answer"],
        "replies": dict(replies),
        "image": str(bundle / "beauty.png"),
        "twins": {},
        "source": "model-arena",
    }
    why = "this item was generated without a stored scene"
    for kind in counterfactual.VISUAL:
        if scene is None:
            suite["twins"][kind] = {"status": "unavailable", "reason": why}
            continue
        try:
            suite["twins"][kind] = counterfactual.build_visual_twin(
                kind, scene, question, out / "twins" / kind, prof
            )
        except (counterfactual.Unavailable, ValueError, RuntimeError, KeyError) as exc:
            suite["twins"][kind] = {"status": "unavailable", "reason": str(exc)[:200]}
    if scene is None:
        suite["twins"]["text_ir_twin"] = {"status": "unavailable", "reason": why}
    else:
        text = text_ir.from_scene(scene, question.text)
        suite["twins"]["text_ir_twin"] = {
            "status": "text", "source": "scene", "prompt": text, "answer": task["answer"],
        }  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "suites.json", {"suites": [suite], "missing_bundles": []})
    return out
