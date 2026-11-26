"""`knot export` (SPEC 13): Harbor task folders, HF parquet, RLVR jsonl and the dataset manifest."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from knotwork.export.harbor import difficulty_label, difficulty_score, export_task, tertiles
from knotwork.export.hf import write_hf
from knotwork.export.manifest import FILENAME as MANIFEST
from knotwork.export.manifest import write_manifest
from knotwork.export.records import load_items
from knotwork.export.rlvr import write_rlvr
from knotwork.render.canvas import REPO_ROOT

PROFILE = REPO_ROOT / "configs" / "default_profile.yaml"


def run_export(
    data: Path, fmt: str, out: Path, limit: int | None, max_shard_mb: float | None = None
) -> int:
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    items = load_items(Path(data), profile, limit)
    if not items:
        print(f"export: no bundles under {data}")
        return 1
    chosen = set(fmt.split(","))  # "all", one format, or a comma separated list
    if "all" in chosen:
        chosen = {"harbor", "hf", "rlvr"}
    out = Path(out)
    summary: dict[str, object] = {"n_bundles": len(items), "sealed": sum(i.sealed for i in items)}
    if "harbor" in chosen:
        harbor = out / "harbor"
        shutil.rmtree(harbor, ignore_errors=True)
        harbor.mkdir(parents=True)
        # sealed answers never leave the machine: Harbor tasks need the oracle, so they are skipped
        open_items = [i for i in items if not i.sealed]
        cuts: dict[str, tuple[float, float]] = {}
        for plugin in sorted({i.task["plugin"] for i in open_items}):
            cuts[plugin] = tertiles(
                [difficulty_score(i.task) for i in open_items if i.task["plugin"] == plugin]
            )
        records = [
            export_task(i.dir, i.task, harbor, difficulty_label(difficulty_score(i.task), cuts[i.task["plugin"]]))
            for i in open_items
        ]  # fmt: skip
        (harbor / "index.jsonl").write_text(
            "".join(json.dumps(r.model_dump(mode="json"), sort_keys=True) + "\n" for r in records),
            encoding="utf-8",
            newline="\n",
        )
        summary["harbor"] = len(records)
    if "hf" in chosen:
        shutil.rmtree(out / "hf", ignore_errors=True)
        summary["hf"] = write_hf(items, out, max_shard_mb)
    if "rlvr" in chosen:
        shutil.rmtree(out / "rlvr", ignore_errors=True)
        summary["rlvr"] = write_rlvr(items, out)
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(items, out, profile)  # last data file written: it hashes all the others
    summary["manifest"] = MANIFEST
    (out / "export_summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8"
    )
    print("export: " + json.dumps(summary, sort_keys=True))
    return 0


__all__ = ["run_export"]
