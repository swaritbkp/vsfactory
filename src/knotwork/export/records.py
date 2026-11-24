"""Bundles prepared for export: task record, directory and emission-mask sealing (SPEC 11.1)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knotwork.freshness.emission import answer_commitment, is_sealed, load_salt

SEALABLE_SPLITS = ("test", "sealed")


@dataclass
class ExportItem:
    dir: Path
    task: dict[str, Any]
    sealed: bool
    commitment: str | None


def _rows(data: Path) -> list[tuple[Path, dict[str, Any]]]:
    def read(ds: Path) -> list[tuple[Path, dict[str, Any]]]:
        lines = (ds / "index.jsonl").read_text(encoding="utf-8").splitlines()
        return [(ds, json.loads(x)) for x in lines if x.strip()]

    if (data / "index.jsonl").is_file():
        return read(data)
    out: list[tuple[Path, dict[str, Any]]] = []
    for ds in sorted(p for p in data.iterdir() if (p / "index.jsonl").is_file()):
        out += read(ds)
    return out


def load_items(data: Path, profile: dict[str, Any], limit: int | None = None) -> list[ExportItem]:
    """Every bundle of a dataset (or directory of datasets) in index order, sealed per the mask."""
    fr = profile["freshness"]
    salt: bytes | None = None
    items: list[ExportItem] = []
    for ds, r in _rows(Path(data)):
        d = ds / r["dir"]
        task = json.loads((d / "task.json").read_text(encoding="utf-8"))
        sealed, commit = False, None
        if task["split"] in SEALABLE_SPLITS:
            salt = salt if salt is not None else load_salt(fr["emission_salt_path"])
            if is_sealed(salt, task["bundle_id"], float(fr["sealed_fraction"])):
                sealed, commit = True, answer_commitment(salt, task["answer"]["canonical"])
        items.append(ExportItem(d, task, sealed, commit))
        if limit is not None and len(items) >= limit:
            break
    return items


__all__ = ["ExportItem", "load_items"]
