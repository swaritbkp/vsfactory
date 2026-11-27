"""Sharded parquet with embedded images and the dataset manifest, on tiny hand-made bundles."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from PIL import Image

from knotwork.export.hf import write_hf
from knotwork.export.manifest import FILENAME, check_manifest, write_manifest
from knotwork.export.records import ExportItem

REPO = Path(__file__).resolve().parents[2]


def _item(root: Path, i: int) -> ExportItem:
    d = root / f"b{i}"
    d.mkdir(parents=True)
    Image.effect_noise((1024, 1024), 40 + i).convert("RGB").save(d / "beauty.png")
    answer = {"type": "int", "value": i, "canonical": str(i), "tolerance": 0.0}
    task = {
        "bundle_id": f"kw-{i:020d}", "plugin": "tangle", "family": "T2_CROSS", "split": "dev",
        "schema_version": "1.1.0", "question": {"text": "How many?", "difficulty": {}, "steps": []},
        "answer": answer, "certificate": {"steps": [], "kernel_sha256": "0" * 64},
        "canary": "KNOTWORK-CANARY-" + "1" * 32, "twins": {"role": "base"},
        "freshness_receipt_id": "2" * 64,
    }  # fmt: skip
    return ExportItem(d, task, False, None)


def test_shards_embed_images_and_manifest_detects_changes(tmp_path: Path) -> None:
    items = [_item(tmp_path / "src", i) for i in range(3)]
    out = tmp_path / "out"
    assert write_hf(items, out, max_shard_mb=0.001) == {"dev": 3}
    shards = sorted((out / "hf" / "data").glob("*.parquet"))
    assert [s.name for s in shards] == [f"dev-{i:05d}-of-00003.parquet" for i in range(3)]
    first = pq.read_table(shards[0])
    assert b"huggingface" in first.schema.metadata
    assert Image.open(io.BytesIO(first.to_pylist()[0]["image"]["bytes"])).size == (768, 768)
    profile = yaml.safe_load((REPO / "configs/default_profile.yaml").read_text(encoding="utf-8"))
    write_manifest(items, out, profile)
    manifest = json.loads((out / FILENAME).read_text(encoding="utf-8"))
    assert manifest["cardinality"]["per_family_split"] == {"T2_CROSS/dev": 3}
    assert manifest["font"]["pinned_sha256"] == manifest["font"]["file_sha256"]
    assert len(manifest["artifacts"]) == 4 and check_manifest(out) == []
    (out / "hf" / "README.md").write_text("edited", encoding="utf-8")
    assert check_manifest(out) == ["changed hf/README.md"]
