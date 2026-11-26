"""Helpers for the Model Arena tests: a tiny frozen dataset written without rendering anything,
and a workspace root under the test's temporary folder (never the repository's artifacts/)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from knotwork.arena import storage
from knotwork.arena.schemas import BlockSpec, ItemRecord, SessionConfig

SECRET = "sk-or-v1-0123456789abcdef-test-secret"


def make_dataset(data_dir: Path, blocks: dict[str, tuple[str, str, int]]) -> list[ItemRecord]:
    """A frozen dataset: per block (domain, preset, n) items with a beauty.png and a label answer."""
    items = []
    for block_id, (domain, preset, n) in blocks.items():
        for k in range(n):
            rel = f"{block_id}/{k:05d}/a0"
            (data_dir / rel).mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 64), (200, 200, 200)).save(data_dir / rel / "beauty.png")
            truth = f"{k % 5 + 1:02d}"
            items.append(
                ItemRecord(
                    block_id=block_id, item_index=k, seed=k, domain=domain, preset=preset,
                    family="T1_TRACE", certified=True, bundle_id=f"kw-{block_id}{k:018d}"[:23],
                    dir=rel, question="Which port? End with 'ANSWER: <label>'.",
                    options=[f"{j:02d}" for j in range(1, 6)],
                    answer={"type": "label", "value": truth, "canonical": truth, "tolerance": 0.0},
                    truth=truth, gen_ms=100.0 + k, verify_ms=200.0 + 2 * k,
                )
            )  # fmt: skip
    storage.write_manifest(data_dir, items, {"session": "fixture"})
    return items


def config(
    root: Path,
    name: str = "Fixture session",
    data_mode: str = "reuse",
    mode: str = "mock",
    blocks: list[BlockSpec] | None = None,
    data_dir: Path | None = None,
    **extra: object,
) -> SessionConfig:
    slug = storage.slugify(name, "20261127")
    return SessionConfig.model_validate({
        "session_name": name, "slug": slug, "mode": mode, "data_mode": data_mode,
        "data_dir": str(data_dir or root / "data"), "results_dir": str(root / slug / "results"),
        "default_models": ["openai/gpt-5.4", "google/gemini-2.5-flash"],
        "blocks": [b.model_dump() for b in (blocks or [
            BlockSpec(block_id="b1", domain="rack_patch", preset="High stress", n=6, seed=0),
            BlockSpec(block_id="b2", domain="panel", preset="Hard logic", n=4, seed=0,
                      models=["qwen/qwen3-vl-235b-a22b-instruct"]),
        ])], **extra,
    })  # fmt: skip
