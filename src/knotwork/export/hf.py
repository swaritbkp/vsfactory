"""HuggingFace parquet exporter (SPEC 13.2): `OUT/hf/data/<split>-00000-of-00001.parquet` (zstd) plus a
dataset card `OUT/hf/README.md`. Sealed answers (SPEC 11.1) are replaced by `answer_commitment`."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from knotwork.export.harbor import difficulty_score
from knotwork.export.records import ExportItem

COLUMNS = (
    "bundle_id", "plugin", "family", "split", "image", "question", "options", "answer_type", "answer",
    "answer_json", "answer_commitment", "tolerance", "program_json", "certificate_json", "steps_json",
    "canary", "twin_role", "base_id", "difficulty_json",
)  # fmt: skip


def _png_bytes(path: Path) -> bytes:
    from PIL import Image

    from knotwork.knotcore.contracts import CONSUMPTION_PX

    with Image.open(path) as im:
        rgb = im.convert("RGB")
        s = CONSUMPTION_PX / max(rgb.size)
        buf = io.BytesIO()
        rgb.resize((round(rgb.size[0] * s), round(rgb.size[1] * s)), Image.Resampling.LANCZOS).save(
            buf, format="PNG"
        )
        return buf.getvalue()


def row(item: ExportItem) -> dict[str, Any]:
    t, q, a = item.task, item.task["question"], item.task["answer"]
    steps = [
        {"op": s["op"], "inputs": s["inputs"], "output": s["output"]}
        for s in t["certificate"]["steps"]
    ]
    return {
        "bundle_id": t["bundle_id"], "plugin": t["plugin"], "family": t["family"], "split": t["split"],
        "image": {"bytes": _png_bytes(item.dir / "beauty.png"), "path": f"images/{t['bundle_id']}.png"},
        "question": q["text"], "options": list(q.get("options") or []), "answer_type": a["type"],
        "answer": None if item.sealed else a["canonical"],
        "answer_json": None if item.sealed else json.dumps(a, sort_keys=True),
        "answer_commitment": item.commitment, "tolerance": float(a.get("tolerance") or 0.0),
        "program_json": json.dumps(q, sort_keys=True), "certificate_json": json.dumps(t["certificate"], sort_keys=True),
        "steps_json": json.dumps(steps, sort_keys=True), "canary": t["canary"], "twin_role": t["twins"]["role"],
        "base_id": t["twins"].get("base_id"),
        "difficulty_json": json.dumps({**q.get("difficulty", {}), "score": difficulty_score(t)}, sort_keys=True),
    }  # fmt: skip


def schema() -> Any:
    import pyarrow as pa

    s = pa.string()
    return pa.schema([
        ("bundle_id", s), ("plugin", s), ("family", s), ("split", s),
        ("image", pa.struct([("bytes", pa.binary()), ("path", s)])), ("question", s),
        ("options", pa.list_(s)), ("answer_type", s), ("answer", s), ("answer_json", s),
        ("answer_commitment", s), ("tolerance", pa.float64()), ("program_json", s),
        ("certificate_json", s), ("steps_json", s), ("canary", s), ("twin_role", s), ("base_id", s),
        ("difficulty_json", s),
    ])  # fmt: skip


def card(splits: dict[str, int], canaries: list[str]) -> str:
    configs = "\n".join(f"  - split: {s}\n    path: data/{s}-*.parquet" for s in sorted(splits))
    counts = "\n".join(f"| {s} | {n} |" for s, n in sorted(splits.items()))
    canary_lines = "\n".join(f"- `{c}`" for c in sorted(set(canaries)))
    return f"""---
license: apache-2.0
task_categories:
- visual-question-answering
tags:
- knotwork
- visual-reasoning
- synthetic
- proof-carrying
configs:
- config_name: default
  data_files:
{configs}
---

# Knotwork: proof-carrying synthetic visual reasoning tasks

Every row is an (image, question, answer, evidence) bundle whose answer was re-derived **from pixels**
by the Knotwork trusted kernel (`certificate_json`, `steps_json`). Images are at the 768-px
consumption resolution.

| split | rows |
|---|---|
{counts}

## Intended use
Evaluation and RLVR training of vision-language models on cable tracing, crossing counting, z-order,
analog gauge reading and compositional / premise-checking questions. Grade with
`knotwork.knotcore.grading.grade_prediction` (stdlib only).

## Sealed answers
Rows of the `test`/`sealed` splits selected by the emission mask carry `answer_commitment =
SHA256(salt || canonical)` instead of `answer` / `answer_json`.

## Canary
Do not train on this data if your corpus contains any of these strings (contamination canary):
{canary_lines}

## Limitations
Synthetic 2-D scenes (photographic finish, no 3-D perspective); English prompts only.
"""


def write_hf(items: list[ExportItem], out: Path) -> dict[str, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = out / "hf" / "data"
    data.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        by_split.setdefault(it.task["split"], []).append(row(it))
    for split, rows in by_split.items():
        table = pa.Table.from_pylist(rows, schema=schema())
        pq.write_table(table, data / f"{split}-00000-of-00001.parquet", compression="zstd")
    counts = {s: len(r) for s, r in by_split.items()}
    (out / "hf" / "README.md").write_text(
        card(counts, [it.task["canary"] for it in items]), encoding="utf-8", newline="\n"
    )
    return counts


__all__ = ["COLUMNS", "card", "row", "schema", "write_hf"]
