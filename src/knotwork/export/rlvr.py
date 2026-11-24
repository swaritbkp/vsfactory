"""RLVR exporter (SPEC 13.2): `OUT/rlvr/tasks.jsonl` + `OUT/rlvr/images/<id>.png` (768 px).

Each line: {id, prompt, image_path, answer (or answer_commitment when sealed), steps (the kernel's
process trace), reward_fn, canary}."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knotwork.export.harbor import consumption_png
from knotwork.export.records import ExportItem

REWARD_FN = "knotwork.knotcore.grading:grade_prediction"


def line(item: ExportItem) -> dict[str, Any]:
    t = item.task
    rec: dict[str, Any] = {
        "id": t["bundle_id"],
        "prompt": t["question"]["text"],
        "image_path": f"images/{t['bundle_id']}.png",
        "steps": [
            {"op": s["op"], "inputs": s["inputs"], "output": s["output"]}
            for s in t["certificate"]["steps"]
        ],
        "reward_fn": REWARD_FN,
        "canary": t["canary"],
        "family": t["family"],
        "split": t["split"],
    }
    if item.sealed:
        rec["answer_commitment"] = item.commitment
    else:
        rec["answer"] = t["answer"]
    return rec


def write_rlvr(items: list[ExportItem], out: Path) -> int:
    root = out / "rlvr"
    (root / "images").mkdir(parents=True, exist_ok=True)
    with (root / "tasks.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for it in items:
            consumption_png(it.dir / "beauty.png", root / "images" / f"{it.task['bundle_id']}.png")
            f.write(json.dumps(line(it), sort_keys=True) + "\n")
    return len(items)


__all__ = ["REWARD_FN", "line", "write_rlvr"]
