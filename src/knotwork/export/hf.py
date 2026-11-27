"""HuggingFace parquet exporter (SPEC 13.2): `OUT/hf/data/<split>-NNNNN-of-MMMMM.parquet` (zstd) with
the PNG bytes embedded in the `image` column, plus a dataset card `OUT/hf/README.md`. Shards are
closed when they reach `max_shard_mb` of row data, so memory stays bounded by one shard. Sealed
answers (SPEC 11.1) are replaced by `answer_commitment`."""

from __future__ import annotations

import io
import json
from collections import Counter
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


def hf_metadata() -> dict[bytes, bytes]:
    """Parquet schema metadata that makes `datasets` decode `image` as a picture."""
    value = {"dtype": "string", "_type": "Value"}
    features: dict[str, Any] = {name: dict(value) for name in COLUMNS}
    features["image"] = {"_type": "Image"}
    features["options"] = {"feature": dict(value), "_type": "Sequence"}
    features["tolerance"] = {"dtype": "float64", "_type": "Value"}
    return {b"huggingface": json.dumps({"info": {"features": features}}, sort_keys=True).encode()}


def row_bytes(r: dict[str, Any]) -> int:
    """Approximate stored size of a row: the PNG plus its text columns."""
    text = sum(len(v) for v in r.values() if isinstance(v, str))
    return len(r["image"]["bytes"]) + text


def card(
    splits: dict[str, int],
    canaries: list[str],
    shards: dict[str, int] | None = None,
    families: dict[str, int] | None = None,
) -> str:
    configs = "\n".join(f"  - split: {s}\n    path: data/{s}-*.parquet" for s in sorted(splits))
    counts = "\n".join(f"| {s} | {n} |" for s, n in sorted(splits.items()))
    canary_lines = "\n".join(f"- `{c}`" for c in sorted(set(canaries)))
    total = sum(splits.values())
    size = "n<1K" if total < 1000 else ("1K<n<10K" if total < 10000 else "10K<n<100K")
    family_lines = "\n".join(f"| {f} | {n} |" for f, n in sorted((families or {}).items()))
    shard_note = ", ".join(f"{s}: {n}" for s, n in sorted((shards or {}).items())) or "1 per split"
    return f"""---
license: apache-2.0
task_categories:
- visual-question-answering
tags:
- knotwork
- visual-reasoning
- synthetic
- proof-carrying
language:
- en
pretty_name: Knotwork visual reasoning tasks
size_categories:
- {size}
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

| family | rows |
|---|---|
{family_lines}

## Loading
```python
from datasets import load_dataset

data = load_dataset("path/to/this/folder")
```
`image` holds the PNG bytes inside the parquet shards ({shard_note}). `../dataset_manifest.json`
records the git commit, kernel hash, font hash and the SHA-256 of every exported file.

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


def _write_part(
    data: Path, split: str, index: int, rows: list[dict[str, Any]], target: Any
) -> Path:
    """One shard under a temporary name (its final name needs the shard count)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    part = data / f"{split}-{index:05d}.partial"
    pq.write_table(pa.Table.from_pylist(rows, schema=target), part, compression="zstd")
    return part


def write_hf(
    items: list[ExportItem], out: Path, max_shard_mb: float | None = None
) -> dict[str, int]:
    """Write the shards and the card; returns rows per split."""
    data = out / "hf" / "data"
    data.mkdir(parents=True, exist_ok=True)
    limit = None if max_shard_mb is None else max_shard_mb * 1024 * 1024
    target = schema().with_metadata(hf_metadata())
    counts: Counter[str] = Counter()
    shards: dict[str, int] = {}
    for split in sorted({it.task["split"] for it in items}):
        rows: list[dict[str, Any]] = []
        size = 0
        parts: list[Path] = []

        for it in (i for i in items if i.task["split"] == split):
            r = row(it)
            n = row_bytes(r)
            if rows and limit is not None and size + n > limit:
                parts.append(_write_part(data, split, len(parts), rows, target))
                rows, size = [], 0
            rows.append(r)
            size += n
            counts[split] += 1
        parts.append(_write_part(data, split, len(parts), rows, target))
        for i, part in enumerate(parts):
            part.replace(data / f"{split}-{i:05d}-of-{len(parts):05d}.parquet")
        shards[split] = len(parts)
    families = Counter(it.task["family"] for it in items)
    text = card(dict(counts), [it.task["canary"] for it in items], shards, dict(families))
    (out / "hf" / "README.md").write_text(text, encoding="utf-8", newline="\n")
    return dict(counts)


__all__ = ["COLUMNS", "card", "hf_metadata", "row", "schema", "write_hf"]
