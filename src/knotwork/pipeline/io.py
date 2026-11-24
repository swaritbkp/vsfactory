"""Bundle I/O helpers (SPEC 3): decoded-array file digests, bundle ids, task/index/quarantine files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel

from knotwork.knotcore import seeds
from knotwork.knotcore.contracts import Answer, ClaimSet, QuestionProgram, TaskBundle


def write_json(path: Path, obj: Any) -> None:
    data = obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for r in rows:
            data = r.model_dump(mode="json") if isinstance(r, BaseModel) else r
            f.write(json.dumps(data, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_claims(d: Path, claims: ClaimSet) -> None:
    write_json(d / "claims.json", claims)


def file_digests(d: Path) -> dict[str, str]:
    """SPEC 3 `TaskBundle.files`: decoded-array digests for PNGs, canonical JSON digests otherwise."""
    out = {
        "beauty.png": seeds.array_digest(
            np.asarray(Image.open(d / "beauty.png").convert("RGB"), dtype=np.uint8)
        ),
        "id_buffer.png": seeds.array_digest(
            np.asarray(Image.open(d / "id_buffer.png"), dtype=np.uint16)
        ),
        "manifest.json": seeds.json_digest(read_json(d / "manifest.json")),
        "claims.json": seeds.json_digest(read_json(d / "claims.json")),
    }
    for m in sorted((d / "masks").glob("obj_*.png")):
        out[f"masks/{m.name}"] = seeds.array_digest(np.asarray(Image.open(m)) > 0)
    return out


def bundle_id_for(files: dict[str, str], question: QuestionProgram, answer: Answer) -> str:
    return seeds.bundle_id(files, question.model_dump(mode="json"), answer.model_dump(mode="json"))


def write_task(d: Path, task: TaskBundle) -> None:
    write_json(d / "task.json", task)


def read_task(d: Path) -> TaskBundle:
    return TaskBundle.model_validate(read_json(d / "task.json"))
