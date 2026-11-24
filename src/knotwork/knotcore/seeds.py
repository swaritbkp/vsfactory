"""Hierarchical deterministic seed tree and SHA-256 content addressing.

Rules (SPEC.md section 5):
* Every random draw in Knotwork comes from `SeedTree(...).rng()`; Python's `random` module and
  `np.random.<legacy>` functions are forbidden.
* A node is identified by (root, path). Children are derived with SHA-256 of a typed path
  component, so `child("a", 1)` and `child("a", "1")` are distinct and order matters.
* Content digests hash *decoded* data (canonical JSON, raw pixel arrays), never encoder output
  such as PNG bytes, so digests are stable across zlib / Pillow builds.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

ROOT_BITS = 128
_ROOT_LIMIT = 1 << ROOT_BITS
OFFLINE_LABEL = "knotwork-offline-beacon-v1"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON bytes: sorted keys, no whitespace, UTF-8, NaN/Inf forbidden."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def json_digest(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))


def array_digest(arr: npt.NDArray[Any]) -> str:
    """SHA-256 over little-endian dtype string, shape and C-ordered bytes."""
    a = np.ascontiguousarray(arr)
    a = a.astype(a.dtype.newbyteorder("<"), copy=False)
    header = f"{a.dtype.str}|{','.join(map(str, a.shape))}|".encode()
    return sha256_hex(header + a.tobytes(order="C"))


def _component_key(part: str | int) -> int:
    if isinstance(part, bool) or not isinstance(part, (str, int)):
        raise TypeError(f"seed path components must be str or int, got {type(part).__name__}")
    tag = "i" if isinstance(part, int) else "s"
    digest = hashlib.sha256(f"{tag}:{part}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class SeedTree:
    """Immutable node of the hierarchical seed tree."""

    __slots__ = ("_path", "_root")

    def __init__(self, root: int, path: tuple[str | int, ...] = ()) -> None:
        if isinstance(root, bool) or not isinstance(root, int) or not 0 <= root < _ROOT_LIMIT:
            raise ValueError(f"root seed must be an int in [0, 2**{ROOT_BITS})")
        for part in path:
            _component_key(part)
        self._root = root
        self._path = tuple(path)

    @property
    def root(self) -> int:
        return self._root

    @property
    def path(self) -> tuple[str | int, ...]:
        return self._path

    @property
    def path_str(self) -> str:
        return "/".join(f"{'i' if isinstance(p, int) else 's'}:{p}" for p in self._path) or "<root>"

    def child(self, *parts: str | int) -> SeedTree:
        if not parts:
            raise ValueError("child() needs at least one path component")
        return SeedTree(self._root, self._path + tuple(parts))

    def seed_sequence(self) -> np.random.SeedSequence:
        spawn_key = tuple(_component_key(p) for p in self._path)
        return np.random.SeedSequence(entropy=self._root, spawn_key=spawn_key)

    def rng(self) -> np.random.Generator:
        return np.random.Generator(np.random.PCG64(self.seed_sequence()))

    def int_seed(self, bits: int = 64) -> int:
        if bits not in (32, 64, 128):
            raise ValueError("bits must be 32, 64 or 128")
        words = self.seed_sequence().generate_state(4, np.uint32)
        value = 0
        for w in words:
            value = (value << 32) | int(w)
        return value >> (128 - bits)

    def digest(self) -> str:
        return json_digest({"root": str(self._root), "path": self.path_str})

    def __repr__(self) -> str:
        return f"SeedTree(root=0x{self._root:032x}, path={self.path_str!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SeedTree) and (self._root, self._path) == (
            other._root,
            other._path,
        )

    def __hash__(self) -> int:
        return hash((self._root, self._path))


def derive_root_seed(randomness_hex: str, protocol_commit: str) -> int:
    """root = first 128 bits of SHA256(randomness_bytes || protocol_commit_utf8)."""
    if len(randomness_hex) % 2 or not randomness_hex:
        raise ValueError("randomness_hex must be a non-empty even-length hex string")
    data = bytes.fromhex(randomness_hex) + protocol_commit.encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:16], "big")


def offline_randomness(counter: int = 0) -> str:
    """Deterministic stand-in for beacon randomness when running --offline."""
    if counter < 0:
        raise ValueError("counter must be >= 0")
    return sha256_hex(f"{OFFLINE_LABEL}:{counter}".encode())


def content_id(entries: Mapping[str, str], prefix: str = "kw", n_hex: int = 20) -> str:
    """Content address over a mapping of logical-name -> digest (order independent)."""
    if not entries:
        raise ValueError("content_id needs at least one entry")
    return f"{prefix}-{json_digest(dict(sorted(entries.items())))[:n_hex]}"


def bundle_id(file_digests: Mapping[str, str], question: Any, answer: Any) -> str:
    """kw-<20 hex>: SHA-256 over decoded file digests + canonical question + canonical answer."""
    entries = dict(file_digests)
    entries["__question__"] = json_digest(question)
    entries["__answer__"] = json_digest(answer)
    return content_id(entries)


def canary_for(root_seed: int) -> str:
    """GUID-style contamination canary embedded in every exported artifact."""
    return f"KNOTWORK-CANARY-{sha256_hex(f'canary:{root_seed}'.encode())[:32]}"


def selftest_vector() -> dict[str, str]:
    """Known-answer vector proving cross-machine determinism (compared with SELFTEST_EXPECTED)."""
    root = derive_root_seed(offline_randomness(0), "0" * 64)
    tree = SeedTree(root)
    rng = tree.child("tangle", 7).rng()
    return {
        "root_seed": f"{root:032x}",
        "child_int": f"{tree.child('tangle', 7, 'jitter').int_seed():016x}",
        "rng_draws": array_digest(rng.integers(0, 2**31, size=16, dtype=np.int64)),
        "array_u16": array_digest(np.arange(12, dtype=np.uint16).reshape(3, 4)),
        "bundle_id": bundle_id({"beauty": "0" * 64}, {"q": "x"}, {"a": 1}),
        "canary": canary_for(root),
    }


SELFTEST_EXPECTED: dict[str, str] = {
    "root_seed": "6f90f59ab594e6c94856b16cd7563439",
    "child_int": "dc14429ef3b92971",
    "rng_draws": "08ed185129a1c833612abfcd64009664f22f7c06df27c11f7d271d3639aaaf2f",
    "array_u16": "4920aaab215d4d6c539b2f372b7068d52a85cd709277f2bd4edea5361a53d28f",
    "bundle_id": "kw-3097ec38135e5cb9475b",
    "canary": "KNOTWORK-CANARY-aa13dca9f5db55b1d849e538bcf08605",
}
