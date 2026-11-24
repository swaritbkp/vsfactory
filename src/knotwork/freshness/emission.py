"""Emission mask (SPEC 11.1): a local secret salt decides which test/sealed answers are sealed.

The salt never affects generation (bundle ids stay deterministic); it only feeds the receipt's
`emission_mask_commit` and the exporters' answer sealing.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path

from knotwork.render.canvas import REPO_ROOT


def salt_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_salt(path: str | Path) -> bytes:
    """32-byte salt, created with os.urandom on first use (the .secrets/ directory is git-ignored)."""
    p = salt_path(path)
    if not p.is_file():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(os.urandom(32))
    data = p.read_bytes()
    if len(data) != 32:
        raise ValueError(f"emission salt at {p} must be 32 bytes")
    return data


def mask_commit(salt: bytes) -> str:
    return hashlib.sha256(salt).hexdigest()


def is_sealed(salt: bytes, bundle_id: str, sealed_fraction: float) -> bool:
    return hmac.new(salt, bundle_id.encode(), hashlib.sha256).digest()[0] / 256 < sealed_fraction


def answer_commitment(salt: bytes, canonical: str) -> str:
    return hashlib.sha256(salt + canonical.encode("utf-8")).hexdigest()
