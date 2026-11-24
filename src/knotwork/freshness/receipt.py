"""Freshness receipt (SPEC 11.1): binds a generation run to beacon randomness + the protocol files."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from knotwork.freshness.beacon import BeaconValue
from knotwork.freshness.emission import load_salt, mask_commit
from knotwork.knotcore.contracts import FreshnessReceipt
from knotwork.knotcore.seeds import canary_for, derive_root_seed, json_digest
from knotwork.render.canvas import REPO_ROOT


def protocol_commit(files: list[str], root: Path = REPO_ROOT) -> str:
    """SHA256 over sorted files: path_utf8 || 0x00 || bytes (CRLF -> LF) || 0x00 (SPEC 5)."""
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(f.encode("utf-8") + b"\0")
        h.update((root / f).read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()


def build_receipt(beacon: BeaconValue, profile: dict[str, Any]) -> FreshnessReceipt:
    fr = profile["freshness"]
    commit = protocol_commit(list(fr["protocol_files"]))
    root = derive_root_seed(beacon.randomness_hex, commit)
    fields: dict[str, Any] = {
        "protocol_commit": commit,
        "source": beacon.source,
        "round": beacon.round,
        "randomness_hex": beacon.randomness_hex,
        "root_seed": root,
        "beacon_url": beacon.url,
        "emission_mask_commit": mask_commit(load_salt(fr["emission_salt_path"])),
        "canary": canary_for(root),
    }
    rid = json_digest({**fields, "root_seed": str(root)})
    return FreshnessReceipt(receipt_id=rid, fetched_at_utc=beacon.fetched_at_utc, **fields)


def receipt_id_ok(r: FreshnessReceipt) -> bool:
    data = r.model_dump(mode="json")
    data.pop("receipt_id")
    data.pop("fetched_at_utc")
    data["root_seed"] = str(r.root_seed)
    return json_digest(data) == r.receipt_id
