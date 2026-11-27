"""Post-training test helpers: real certified bundles (committed under tests/posttrain/fixtures,
without their 1 MB beauty.png) and an Arena session built on them without rendering anything.

The fixtures were rendered once with `knotwork.studio.factory.generate_one(<domain>, <family>, 1, ...)`
and are certified, so vCoT, grading and the pack exporter see exactly what the factory writes."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

from arena_fixtures import config
from knotwork.arena import storage
from knotwork.arena.schemas import BlockSpec, ItemRecord, SessionConfig, TrialRecord

FIXTURES = Path(__file__).resolve().parent / "posttrain" / "fixtures"
DOMAINS = ("rack_patch", "tangle", "gauge", "panel")


def task(domain: str) -> dict:  # type: ignore[type-arg]
    return json.loads((FIXTURES / domain / "task.json").read_text(encoding="utf-8"))


def bundle_copy(domain: str, dest: Path) -> Path:
    """A full bundle folder: the committed fixture plus a flat grey beauty.png."""
    shutil.copytree(FIXTURES / domain, dest)
    Image.new("RGB", (1024, 1024), (196, 200, 206)).save(dest / "beauty.png")
    return dest


#: block -> (domain, preset, per-model correctness pattern over the block's items)
BLOCKS = {
    "b1": ("rack_patch", "Clean", {"m/easy": [1, 1, 1, 1], "m/mid": [1, 0, 1, 0]}),
    "b2": ("panel", "Hard logic", {"m/easy": [0, 0, 0, 0], "m/mid": [0, 0, 0, 1]}),
}


def session(root: Path) -> SessionConfig:
    """A finished mock session: two blocks of 4 real items, two models, fixed outcomes.

    b1 / m/easy = 100 % (saturated), b1 / m/mid = 50 % (frontier),
    b2 / m/easy = 0 % (cold start), b2 / m/mid = 25 % (frontier)."""
    data = root / "data"
    items, trials = [], []
    for block_id, (domain, preset, outcomes) in BLOCKS.items():
        t = task(domain)
        for k in range(4):
            rel = f"{block_id}/{k:05d}/a0"
            bundle_copy(domain, data / rel)
            items.append(ItemRecord(
                block_id=block_id, item_index=k, seed=100 + k, domain=domain, preset=preset,
                family=t["family"], certified=True, bundle_id=f"{t['bundle_id']}-{block_id}{k}",
                dir=rel, question=t["question"]["text"], options=t["question"].get("options"),
                answer=t["answer"], truth=t["answer"]["canonical"], gen_ms=120.0 + k,
                verify_ms=300.0 + k,
            ))  # fmt: skip
            for model, pattern in outcomes.items():
                ok = bool(pattern[k])
                reply = f"ANSWER: {t['answer']['canonical'] if ok else 'zz'}"
                trials.append(TrialRecord(
                    trial_id=f"{block_id}:{k}:{model}", block_id=block_id, item_index=k,
                    bundle_id=items[-1].bundle_id, model=model, status="ok", correct=ok,
                    truth=t["answer"]["canonical"], reply=reply,
                    error_category=None if ok else "wrong_port_or_strand",
                ))  # fmt: skip
    storage.write_manifest(data, items, {"session": "posttrain-fixture"})
    blocks = [BlockSpec(block_id=b, domain=d, preset=p, n=4, seed=100)  # type: ignore[arg-type]
              for b, (d, p, _) in BLOCKS.items()]  # fmt: skip
    cfg = config(root, "Posttrain fixture", blocks=blocks, data_dir=data)
    cfg = cfg.model_copy(update={"default_models": ["m/easy", "m/mid"]})
    storage.write_session(cfg)
    results = storage.resolve(cfg.results_dir)
    for t in trials:
        storage.append_trial(results, t)
    return cfg
