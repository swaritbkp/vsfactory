"""Burn ledger (SPEC 11.1): append-only record of every item sent to a live API. Scenes in the
ledger are never re-emitted for split test/sealed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knotwork.render.canvas import REPO_ROOT

FIELDS = ("bundle_id", "scene_id", "slot", "model", "request_sha256")


def ledger_path(profile: dict[str, Any]) -> Path:
    p = Path(profile["freshness"]["burn_ledger_path"])
    return p if p.is_absolute() else REPO_ROOT / p


def append(profile: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Append rows (exactly the FIELDS keys); existing lines are never rewritten."""
    path = ledger_path(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for r in rows:
            if set(r) != set(FIELDS):
                raise ValueError(f"ledger rows need exactly {FIELDS}")
            f.write(json.dumps({k: r[k] for k in FIELDS}, sort_keys=True) + "\n")


def read(profile: dict[str, Any]) -> list[dict[str, Any]]:
    path = ledger_path(profile)
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def burned_scene_ids(profile: dict[str, Any]) -> set[str]:
    return {str(r["scene_id"]) for r in read(profile)}
