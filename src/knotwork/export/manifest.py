"""`dataset_manifest.json`: what a dataset is, what built it, and the hash of every exported file.

Contains no timestamps, so exporting the same bundles from the same checkout twice gives the same
manifest.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from knotwork.export.records import ExportItem
from knotwork.knotcore import engine
from knotwork.knotcore.contracts import KERNEL_VERSION, SCHEMA_VERSION
from knotwork.render.canvas import REPO_ROOT

FILENAME = "dataset_manifest.json"
MANIFEST_SCHEMA = "knotwork.dataset-manifest/1"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state(root: Path = REPO_ROOT) -> dict[str, Any]:
    """HEAD commit and whether tracked files differ from it; "unknown" outside a git checkout."""

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip()

    sha = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"sha": sha or "unknown", "dirty": None if status is None else bool(status)}


def artifact_hashes(out: Path) -> dict[str, dict[str, Any]]:
    """relative path -> sha256 and size of every file under the export directory but the manifest."""
    skip = {FILENAME, "export_summary.json"}
    files = (p for p in out.rglob("*") if p.is_file() and p.name not in skip)
    return {
        p.relative_to(out).as_posix(): {"sha256": file_sha256(p), "bytes": p.stat().st_size}
        for p in sorted(files, key=lambda p: p.relative_to(out).as_posix())
    }


def _counts(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def build_manifest(
    items: Sequence[ExportItem], out: Path, profile: dict[str, Any]
) -> dict[str, Any]:
    tasks = [i.task for i in items]
    sealed = sum(i.sealed for i in items)
    font = Path(profile["render"]["font_path"])
    font = font if font.is_absolute() else REPO_ROOT / font
    kernel = engine.kernel_sha256()
    certified_by = sorted({t["certificate"]["kernel_sha256"] for t in tasks})
    per_family_split = Counter(f"{t['family']}/{t['split']}" for t in tasks)
    return {
        "schema": MANIFEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "bundle_schema_versions": sorted({str(t.get("schema_version")) for t in tasks}),
        "kernel_version": KERNEL_VERSION,
        "git": git_state(),
        "kernel_sha256": kernel,
        "certificate_kernel_sha256": certified_by,
        "certified_by_this_kernel": certified_by == [kernel],
        "font": {
            "path": profile["render"]["font_path"],
            "pinned_sha256": profile["render"]["font_sha256"],
            "file_sha256": file_sha256(font),
        },
        "n_bundles": len(tasks),
        "cardinality": {
            "per_plugin": _counts([t["plugin"] for t in tasks]),
            "per_family": _counts([t["family"] for t in tasks]),
            "per_split": _counts([t["split"] for t in tasks]),
            "per_twin_role": _counts([t["twins"]["role"] for t in tasks]),
            "per_family_split": dict(sorted(per_family_split.items())),
        },
        "sealed": {"n": sealed, "fraction": round(sealed / len(tasks), 6) if tasks else 0.0},
        "canaries": sorted({t["canary"] for t in tasks}),
        "freshness_receipts": sorted({t["freshness_receipt_id"] for t in tasks}),
        "artifacts": artifact_hashes(out),
    }


def write_manifest(items: Sequence[ExportItem], out: Path, profile: dict[str, Any]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / FILENAME
    path.write_text(
        json.dumps(build_manifest(items, out, profile), indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def check_manifest(out: Path) -> list[str]:
    """Exported files that are missing, changed or unlisted compared with the manifest."""
    listed = json.loads((out / FILENAME).read_text(encoding="utf-8"))["artifacts"]
    actual = artifact_hashes(out)
    problems = [f"missing {p}" for p in sorted(set(listed) - set(actual))]
    problems += [f"unlisted {p}" for p in sorted(set(actual) - set(listed))]
    problems += [
        f"changed {p}" for p in sorted(set(listed) & set(actual)) if listed[p] != actual[p]
    ]
    return problems
