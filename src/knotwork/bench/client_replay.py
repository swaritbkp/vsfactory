"""Replay cache (SPEC 12.4): every live response is stored before it is returned; replays never touch
the network. Key = SHA256(canonical_json({slot, slug, prompt, image_sha256, seed})); file
``<replay_dir>/<slot>/<key>.json`` = {request (no auth, image elided to its sha256), response, text,
usage, cost_usd}."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knotwork.bench.client_base import ReplayMiss, Reply, VLMClient
from knotwork.bench.prompts import image_sha256
from knotwork.knotcore.seeds import canonical_json, sha256_hex


def replay_key(slot: str, slug: str, prompt: str, image_sha: str, seed: int) -> str:
    return sha256_hex(
        canonical_json(
            {
                "slot": slot,
                "slug": slug,
                "prompt": prompt,
                "image_sha256": image_sha,
                "seed": int(seed),
            }
        )  # fmt: skip
    )


def elide_images(body: dict[str, Any], image_sha: str) -> dict[str, Any]:
    """Request body for the cache: the base64 image is replaced by its sha256."""
    out: dict[str, Any] = json.loads(json.dumps(body))
    for msg in out.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    part["image_url"]["url"] = f"sha256:{image_sha}"
    return out


def replay_path(replay_dir: Path, slot: str, key: str) -> Path:
    return Path(replay_dir) / slot / f"{key}.json"


def write_entry(replay_dir: Path, slot: str, key: str, entry: dict[str, Any]) -> Path:
    path = replay_path(replay_dir, slot, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entry, indent=1, sort_keys=True), encoding="utf-8", newline="\n")
    tmp.replace(path)
    return path


class ReplayClient:
    mode = "replay"

    def __init__(self, replay_dir: Path, slugs: dict[str, str]) -> None:
        self.replay_dir = Path(replay_dir)
        self.slugs = slugs
        self.network_calls = 0

    def ask(
        self,
        slot: str,
        image_png: bytes,
        prompt: str,
        seed: int,
        meta: dict[str, Any] | None = None,
    ) -> Reply:
        key = replay_key(slot, self.slugs[slot], prompt, image_sha256(image_png), seed)
        path = replay_path(self.replay_dir, slot, key)
        if not path.is_file():
            raise ReplayMiss(f"{slot}/{key}")
        e = json.loads(path.read_text(encoding="utf-8"))
        u = e.get("usage") or {}
        return Reply(text=str(e["text"]), usage_in=int(u.get("prompt_tokens", 0)),
                     usage_out=int(u.get("completion_tokens", 0)), cost_usd=0.0, cached=True)  # fmt: skip


class ReplayRecorder:
    """Wraps another client and writes each reply in replay format (used to build offline
    fixtures from the mock client; the live client writes its own entries before returning)."""

    def __init__(self, inner: VLMClient, replay_dir: Path, slugs: dict[str, str]) -> None:
        self.inner, self.replay_dir, self.slugs = inner, Path(replay_dir), slugs
        self.mode = inner.mode
        self.network_calls = 0

    def ask(
        self,
        slot: str,
        image_png: bytes,
        prompt: str,
        seed: int,
        meta: dict[str, Any] | None = None,
    ) -> Reply:
        reply = self.inner.ask(slot, image_png, prompt, seed, meta)
        sha = image_sha256(image_png)
        key = replay_key(slot, self.slugs[slot], prompt, sha, seed)
        write_entry(self.replay_dir, slot, key, {
            "request": {"slot": slot, "model": self.slugs[slot], "prompt": prompt, "image_sha256": sha,
                        "seed": int(seed)},
            "response": {"source": f"recorded from {self.inner.mode} client"},
            "text": reply.text, "usage": {"prompt_tokens": reply.usage_in, "completion_tokens": reply.usage_out},
            "cost_usd": reply.cost_usd,
        })  # fmt: skip
        return reply


__all__ = [
    "ReplayClient",
    "ReplayRecorder",
    "elide_images",
    "replay_key",
    "replay_path",
    "write_entry",
]
