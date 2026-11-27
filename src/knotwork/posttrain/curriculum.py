"""Adaptive GRPO curriculum from an Arena session, and the one-click post-training pack.

Frontier analysis. With a binary verifiable reward, a GRPO group of rollouts at success rate p has
reward variance p(1 - p): the group-relative advantage, and with it the policy gradient, vanishes
as p -> 0 or p -> 1. Each (block, model) cell is therefore classed by its observed accuracy:

* `saturated`  (p > 0.75)          -- little advantage left: down-weight;
* `frontier`   (0.20 <= p <= 0.75) -- peak gradient signal: prioritise for GRPO;
* `cold_start` (p < 0.20)          -- vanishing advantage: grounded-vCoT SFT first.

Sampling weights are proportional to max(p(1 - p), WEIGHT_FLOOR) and sum to 1.

Seed disjointness. Training seeds start at the first multiple of SEED_STRIDE strictly above the
largest seed the session evaluated, and the fresh OOD evaluation range starts strictly above the
last training seed, so no training item can share a seed (hence a scene) with an evaluation item.
`provenance_attestation.json` re-checks this and records hashes of everything it covers.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from knotwork.arena import storage
from knotwork.arena.schemas import ItemRecord, SessionConfig, TrialRecord
from knotwork.posttrain import vcot
from knotwork.posttrain.reward import score_rollout

FRONTIER_LO, FRONTIER_HI = 0.20, 0.75
WEIGHT_FLOOR = 0.02
SEED_STRIDE = 10_000
#: training items planned per evaluated item (spread over blocks by sampling weight)
TRAIN_PER_EVAL = 10
PACK_DIR = "posttrain_pack"
REGIMES = {
    "saturated": ("Saturated", "GRPO", "green"),
    "frontier": ("GRPO Frontier", "GRPO", "blue"),
    "cold_start": ("SFT Cold-Start", "SFT_COLD_START", "orange"),
}
GROUNDED_INSTRUCTION = (
    "Think step by step inside <think></think>. Ground every step with the region you look at as "
    "<box>[x1, y1, x2, y2]</box> on a 0-1000 grid of the image, then give the final answer as "
    "\\boxed{answer}."
)
ZIP_DATE = (2026, 1, 1, 0, 0, 0)


# ------------------------------------------------------------------ frontier analysis
def regime(p: float | None) -> str | None:
    if p is None:
        return None
    if p > FRONTIER_HI:
        return "saturated"
    return "frontier" if p >= FRONTIER_LO else "cold_start"


def advantage_proxy(p: float) -> float:
    return max(p * (1.0 - p), WEIGHT_FLOOR)


def _cells(cfg: SessionConfig, trials: dict[str, TrialRecord]) -> list[dict[str, Any]]:
    by: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for t in trials.values():
        if t.status == "ok" and t.correct is not None:
            by[(t.block_id, t.model)].append(bool(t.correct))
    cells = []
    for block in cfg.blocks:
        for model in cfg.models_for(block):
            got = by.get((block.block_id, model), [])
            p = sum(got) / len(got) if got else None
            cells.append({"block_id": block.block_id, "block": block.label, "domain": block.domain,
                          "model": model, "n": len(got), "correct": sum(got), "accuracy": p,
                          "regime": regime(p)})  # fmt: skip
    return cells


def _normalise(cells: list[dict[str, Any]], key: str = "accuracy") -> None:
    raw = [advantage_proxy(c[key]) if c[key] is not None else 0.0 for c in cells]
    total = sum(raw)
    for c, r in zip(cells, raw, strict=True):
        c["weight"] = r / total if total else 0.0
        info = REGIMES.get(c["regime"] or "")
        c["badge"], c["mode"] = (info[0], info[1]) if info else ("No data", None)


def eval_seeds(cfg: SessionConfig, items: list[ItemRecord]) -> list[int]:
    seeds = {i.seed for i in items}
    for b in cfg.blocks:
        lo, hi = storage.seed_range(b)
        seeds.update(range(lo, hi + 1))
    return sorted(seeds)


def _next_stride(x: int) -> int:
    """The first multiple of SEED_STRIDE strictly above x."""
    return (x // SEED_STRIDE + 1) * SEED_STRIDE


def plan(
    cfg: SessionConfig,
    items: list[ItemRecord],
    trials: dict[str, TrialRecord],
    train_budget: int | None = None,
) -> dict[str, Any]:
    """The curriculum spec (JSON-ready) for one session's files."""
    cells = _cells(cfg, trials)
    _normalise(cells)
    blocks: list[dict[str, Any]] = []
    for block in cfg.blocks:
        mine = [c for c in cells if c["block_id"] == block.block_id and c["n"]]
        n, k = sum(c["n"] for c in mine), sum(c["correct"] for c in mine)
        p = k / n if n else None
        blocks.append({"block_id": block.block_id, "label": block.label, "domain": block.domain,
                       "preset": block.preset, "eval_n": block.n, "eval_accuracy": p,
                       "regime": regime(p), "eval_seed_range": storage.seed_range(block)})  # fmt: skip
    _normalise(blocks, "eval_accuracy")
    seeds = eval_seeds(cfg, items)
    budget = train_budget or TRAIN_PER_EVAL * max(1, sum(b.n for b in cfg.blocks))
    start = _next_stride(max(seeds) if seeds else 0)
    cursor = start
    for b in blocks:
        n_train = max(1, round(b["weight"] * budget)) if b["eval_accuracy"] is not None else 0
        b["n_train"] = n_train
        b["train_seed_range"] = [cursor, cursor + n_train - 1] if n_train else None
        cursor += n_train
    train_end = cursor - 1
    ood_start = _next_stride(max(train_end, start))
    for i, b in enumerate(blocks):
        b["ood_seed_range"] = [
            ood_start + i * SEED_STRIDE,
            ood_start + i * SEED_STRIDE + b["eval_n"] - 1,
        ]
    ood_end = max(r["ood_seed_range"][1] for r in blocks) if blocks else ood_start
    seed_info = {
        "max_eval_seed": max(seeds) if seeds else None,
        "eval_seeds_sha256": _sha(json.dumps(seeds)),
        "train_seed_start": start, "train_seed_end": train_end,
        "ood_eval_seed_start": ood_start, "ood_eval_seed_end": ood_end,
    }  # fmt: skip
    seed_info["disjoint"] = seeds_disjoint(seeds, blocks)
    results = str(storage.resolve(cfg.results_dir)).replace("\\", "/")
    spec_path = f"{results}/{PACK_DIR}/curriculum_spec.json"
    return {
        "schema": "knotwork.posttrain.curriculum/1",
        "session": cfg.session_name, "slug": cfg.slug,
        "frontier_band": {"lo": FRONTIER_LO, "hi": FRONTIER_HI},
        "weighting": f"w ∝ max(p(1-p), {WEIGHT_FLOOR:.2f}), normalised to 1",
        "cells": cells, "blocks": blocks, "seeds": seed_info, "train_budget": budget,
        "cli": {
            "train": f'uv run knot posttrain-generate --spec "{spec_path}" --split train '
                     f'--out "{results}/{PACK_DIR}/train_data"',
            "ood_eval": f'uv run knot posttrain-generate --spec "{spec_path}" --split ood '
                        f'--out "{results}/{PACK_DIR}/ood_eval_data"',
        },
    }  # fmt: skip


def seeds_disjoint(eval_seed_list: list[int], blocks: list[dict[str, Any]]) -> bool:
    """True when train, OOD and eval seed sets are pairwise disjoint (and train ranges too)."""
    ev = set(eval_seed_list)
    train: set[int] = set()
    for b in blocks:
        r = b.get("train_seed_range")
        if r:
            span = set(range(r[0], r[1] + 1))
            if span & train:
                return False
            train |= span
    ood: set[int] = set()
    for b in blocks:
        r = b.get("ood_seed_range")
        if r:
            ood |= set(range(r[0], r[1] + 1))
    return not (ev & train or ev & ood or train & ood)


def load_session(
    results_dir: Path,
) -> tuple[SessionConfig, list[ItemRecord], dict[str, TrialRecord]]:
    results = storage.resolve(results_dir)
    cfg = storage.load_config(results)
    if cfg is None:
        raise ValueError(f"no valid config.json in {results}")
    wanted = {b.block_id for b in cfg.blocks}
    items = [
        i for i in storage.dataset_items(storage.resolve(cfg.data_dir)) if i.block_id in wanted
    ]
    return cfg, items, storage.load_trials(results)


def analyze(results_dir: Path) -> dict[str, Any]:
    cfg, items, trials = load_session(results_dir)
    return plan(cfg, items, trials)


# ------------------------------------------------------------------ pack records
def _sha(text: str | bytes) -> str:
    raw = text.encode("utf-8") if isinstance(text, str) else text
    return hashlib.sha256(raw).hexdigest()


def prompt_text(question: str) -> str:
    return question.split("\nEnd your reply")[0].strip() + "\n" + GROUNDED_INSTRUCTION


def _image(data: Path, item: ItemRecord, out: Path) -> str:
    rel = f"images/{item.bundle_id}.png"
    dest = out / rel
    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        from knotwork.export.harbor import consumption_png

        consumption_png(data / item.dir / "beauty.png", dest)
    return rel


def _trials_of(item: ItemRecord, trials: dict[str, TrialRecord]) -> list[TrialRecord]:
    return sorted((t for t in trials.values() if t.block_id == item.block_id
                   and t.item_index == item.item_index and t.status == "ok"),
                  key=lambda t: t.trial_id)  # fmt: skip


def build_records(
    cfg: SessionConfig,
    items: list[ItemRecord],
    trials: dict[str, TrialRecord],
    spec: dict[str, Any],
    out: Path,
    kernel_sha: str,
) -> dict[str, list[dict[str, Any]]]:
    """SFT, DPO and GRPO records of every certified item that has a grounded trace."""
    data = storage.resolve(cfg.data_dir)
    block_info = {b["block_id"]: b for b in spec["blocks"]}
    sft, dpo, grpo, skipped = [], [], [], []
    for item in sorted((i for i in items if i.certified), key=lambda i: (i.block_id, i.item_index)):
        trace = vcot.try_extract(data / item.dir)
        if trace is None:
            skipped.append({"bundle_id": item.bundle_id, "reason": "no grounded trace"})
            continue
        image = _image(data, item, out)
        prompt = prompt_text(item.question)
        info = block_info.get(item.block_id, {})
        truth = json.dumps(item.answer, sort_keys=True)
        extra = {"bundle_id": item.bundle_id, "block_id": item.block_id, "domain": item.domain,
                 "family": item.family, "seed": item.seed, "kernel_sha256": kernel_sha,
                 "vcot_boxes_1000": trace.boxes_1000, "regime": info.get("regime") or "",
                 "sampling_weight": float(info.get("weight") or 0.0), "split": "eval_mined"}  # fmt: skip
        grpo.append({
            "data_source": f"knotwork/{item.domain}", "ability": "visual_reasoning",
            "prompt": [{"role": "user", "content": "<image>" + prompt}],
            "images": [{"image": image}], "reward_model": {"style": "rule", "ground_truth": truth},
            "extra_info": extra,
        })  # fmt: skip
        mine = _trials_of(item, trials)
        failed = [t for t in mine if not t.correct]
        if failed or info.get("regime") == "cold_start":
            sft.append({"id": item.bundle_id, "image_path": image, "domain": item.domain,
                        "family": item.family, "seed": item.seed,
                        "reason": "model_failure" if failed else "cold_start_block",
                        "messages": [{"role": "user", "content": [
                            {"type": "image", "image": image}, {"type": "text", "text": prompt}]},
                            {"role": "assistant", "content": trace.formatted_cot}],
                        "vcot_steps": [s.model_dump() for s in trace.steps]})  # fmt: skip
        winners = [t for t in mine if t.correct and t.reply.strip()]
        for t in failed:
            win = next((w for w in winners if w.model != t.model), None) or next(
                iter(winners), None
            )
            chosen = win.reply if win else trace.formatted_cot
            score = {"vcot_boxes_1000": trace.boxes_1000, "domain": item.domain}
            dpo.append({
                "id": f"{t.trial_id}", "prompt": prompt, "image_path": image,
                "chosen": chosen, "rejected": t.reply,
                "chosen_source": f"model:{win.model}" if win else "deterministic_vcot",
                "rejected_model": t.model, "error_category": t.error_category or "",
                "chosen_reward": score_rollout(chosen, truth, score)["score"],
                "rejected_reward": score_rollout(t.reply, truth, score)["score"],
                "bundle_id": item.bundle_id, "seed": item.seed,
            })  # fmt: skip
    return {"sft": sft, "dpo": dpo, "grpo": grpo, "skipped": skipped}


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")


def _parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if rows:
        table = pa.Table.from_pylist(rows)
    else:
        table = pa.table({k: pa.array([], pa.string()) for k in
                          ("data_source", "ability", "prompt", "images", "reward_model", "extra_info")})  # fmt: skip
    pq.write_table(table, path)


# ------------------------------------------------------------------ RLVR drop-in kit
def kit_reward_source() -> str:
    """`knotwork_verl_reward.py`: the kernel's stdlib grader and the reward module in one file."""
    from knotwork.knotcore import grading
    from knotwork.posttrain import reward

    g = Path(grading.__file__).read_text(encoding="utf-8")
    r = Path(reward.__file__).read_text(encoding="utf-8")
    drop = ("from __future__ import annotations",
            "from knotwork.knotcore.grading import ANSWER_RE, committed_answers, grade_prediction")  # fmt: skip
    body = [line for line in (g + "\n\n" + r).splitlines() if line.strip() not in drop]
    head = ('"""Knotwork RLVR reward for verl / TRL: self-contained (stdlib only), generated from '
            "knotwork.knotcore.grading and knotwork.posttrain.reward.\n\nverl: "
            "reward.custom_reward_function.path=<this file> reward.custom_reward_function.name="
            'compute_score\nTRL:  from knotwork_verl_reward import trl_reward\n"""\n\n'
            "from __future__ import annotations\n\n")  # fmt: skip
    return head + "\n".join(body).rstrip() + "\n"


VERL_SH = """#!/usr/bin/env bash
# GRPO with verl on the Knotwork pack; the reward is the self-contained Knotwork verifier.
# Requires a verl install (https://github.com/volcengine/verl); nothing here downloads data.
set -euo pipefail
PACK="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PACK"  # image paths in the parquet are relative to the pack
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
python3 -m verl.trainer.main_ppo \\
  algorithm.adv_estimator=grpo \\
  data.train_files="$PACK/grpo_rlvr_verl.parquet" \\
  data.val_files="$PACK/grpo_rlvr_verl.parquet" \\
  data.image_key=images \\
  data.max_prompt_length=2048 \\
  data.max_response_length=1024 \\
  actor_rollout_ref.model.path="$MODEL" \\
  actor_rollout_ref.rollout.n=8 \\
  reward.custom_reward_function.path="$PACK/rlvr_kit/knotwork_verl_reward.py" \\
  reward.custom_reward_function.name=compute_score \\
  trainer.total_epochs=1 \\
  "$@"
"""

TRL_PY = '''"""GRPO with Hugging Face TRL on the Knotwork pack (requires: pip install trl datasets).

The reward is the self-contained Knotwork verifier next to this file; nothing is downloaded except
the base model you choose.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
sys.path.insert(0, str(HERE))

from datasets import Dataset  # noqa: E402
from PIL import Image  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from knotwork_verl_reward import trl_reward  # noqa: E402


def rows():
    with (PACK / "grpo_rlvr_verl.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            yield {
                "prompt": [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": r["prompt"][0]["content"].replace("<image>", "")}]}],
                "image": Image.open(PACK / r["images"][0]["image"]).convert("RGB"),
                "ground_truth": r["reward_model"]["ground_truth"],
                "extra_info": json.dumps(r["extra_info"]),
                "data_source": r["data_source"],
            }


def main() -> None:
    model = os.environ.get("MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    args = GRPOConfig(output_dir=str(PACK / "trl_grpo_out"), num_generations=8,
                      max_completion_length=1024, per_device_train_batch_size=8)
    trainer = GRPOTrainer(model=model, reward_funcs=trl_reward, args=args,
                          train_dataset=Dataset.from_list(list(rows())))
    trainer.train()


if __name__ == "__main__":
    main()
'''


def export_kit(out: Path) -> dict[str, Path]:
    """Write the RLVR drop-in kit into `out` (a self-contained reward + verl / TRL launchers)."""
    out.mkdir(parents=True, exist_ok=True)
    files = {"knotwork_verl_reward.py": kit_reward_source(), "run_verl_grpo.sh": VERL_SH,
             "run_trl_grpo.py": TRL_PY}  # fmt: skip
    paths = {}
    for name, text in files.items():
        path = out / name
        path.write_text(text, encoding="utf-8", newline="\n")
        if name.endswith(".sh"):
            path.chmod(0o755)
        paths[name] = path
    return paths


# ------------------------------------------------------------------ provenance + zip
def _file_digests(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): _sha(p.read_bytes())
            for p in sorted(root.rglob("*")) if p.is_file()
            and p.name != "provenance_attestation.json"}  # fmt: skip


def attestation(
    cfg: SessionConfig, items: list[ItemRecord], spec: dict[str, Any], out: Path,
    prov: dict[str, str],
) -> dict[str, Any]:  # fmt: skip
    from knotwork.render.canvas import REPO_ROOT

    font = REPO_ROOT / "assets" / "fonts" / "DejaVuSans.ttf"
    seeds = eval_seeds(cfg, items)
    return {
        "schema": "knotwork.posttrain.provenance/1",
        "session": cfg.slug,
        "git_sha": prov.get("git_sha", "unknown"),
        "kernel_sha256": prov.get("kernel_sha256", "unknown"),
        "dataset_sha256": storage.dataset_fingerprint(items),
        "eval_seeds_hash": _sha(json.dumps(seeds)),
        "eval_seed_count": len(seeds),
        "train_seed_range": [spec["seeds"]["train_seed_start"], spec["seeds"]["train_seed_end"]],
        "ood_eval_seed_range": [spec["seeds"]["ood_eval_seed_start"],
                                spec["seeds"]["ood_eval_seed_end"]],
        "train_seed_disjoint": seeds_disjoint(seeds, spec["blocks"]),
        "clean_ip_procedural": True,
        "zero_web_scrape": True,
        "zero_pii": True,
        "external_assets": [{
            "path": "assets/fonts/DejaVuSans.ttf",
            "sha256": _sha(font.read_bytes()) if font.is_file() else "missing",
            "license": "Bitstream Vera / DejaVu (permissive, redistributable)",
            "role": "label glyphs rendered into images; no third-party images or text",
        }],
        "statement": "Every image is procedurally rendered from seeded scene programs and every "
                     "answer is re-derived by the Knotwork kernel; no web data, no people, no "
                     "third-party images.",
        "files_sha256": _file_digests(out),
    }  # fmt: skip


def zip_dir(src: Path, dest: Path) -> Path:
    """Deterministic ZIP (sorted names, fixed timestamps) of everything under `src`."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                info = zipfile.ZipInfo(f"{src.name}/{p.relative_to(src).as_posix()}", ZIP_DATE)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o755 if p.suffix == ".sh" else 0o644) << 16
                z.writestr(info, p.read_bytes())
    dest.write_bytes(buf.getvalue())
    return dest


def export_pack(
    results_dir: Path, out: Path | None = None, prov: dict[str, str] | None = None,
    train_budget: int | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Write `<results>/posttrain_pack/` and `<results>/posttrain_pack.zip`; returns a summary."""
    cfg, items, trials = load_session(results_dir)
    results = storage.resolve(results_dir)
    out = out or results / PACK_DIR
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if prov is None:
        from knotwork.arena.report import provenance

        prov = provenance()
    spec = plan(cfg, items, trials, train_budget)
    recs = build_records(cfg, items, trials, spec, out, prov.get("kernel_sha256", "unknown"))
    _jsonl(out / "sft_grounded_vcot.jsonl", recs["sft"])
    _jsonl(out / "dpo_preferences.jsonl", recs["dpo"])
    _jsonl(out / "grpo_rlvr_verl.jsonl", recs["grpo"])
    _parquet(out / "grpo_rlvr_verl.parquet", recs["grpo"])
    spec["skipped_items"] = recs["skipped"]
    (out / "curriculum_spec.json").write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n",
                                              encoding="utf-8")  # fmt: skip
    export_kit(out / "rlvr_kit")
    att = attestation(cfg, items, spec, out, prov)
    (out / "provenance_attestation.json").write_text(
        json.dumps(att, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    archive = zip_dir(out, out.parent / f"{out.name}.zip")
    return {"dir": out, "zip": archive, "sft": len(recs["sft"]), "dpo": len(recs["dpo"]),
            "grpo": len(recs["grpo"]), "skipped": len(recs["skipped"]),
            "train_seed_disjoint": att["train_seed_disjoint"]}  # fmt: skip


# ------------------------------------------------------------------ seed-disjoint generation
def generation_jobs(spec: dict[str, Any], out: Path, split: str = "train",
                    limit: int | None = None) -> list[dict[str, Any]]:  # fmt: skip
    """Arena item jobs for the spec's train (or OOD eval) seed ranges."""
    key = "train_seed_range" if split == "train" else "ood_seed_range"
    jobs = []
    for b in spec["blocks"]:
        r = b.get(key)
        if not r:
            continue
        for k, seed in enumerate(range(r[0], r[1] + 1)):
            if limit is not None and k >= limit:
                break
            jobs.append({"block_id": b["block_id"], "domain": b["domain"], "preset": b["preset"],
                         "index": k, "seed": seed, "data_dir": str(out)})  # fmt: skip
    return jobs


def generate_from_spec(spec_path: Path, out: Path, split: str = "train", workers: int = 1,
                       limit: int | None = None) -> int:  # fmt: skip
    """Render + certify the planned seed-disjoint items into `out` (an Arena-style dataset)."""
    from concurrent.futures import ProcessPoolExecutor

    from knotwork.arena.generation import make_item_job

    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if not spec.get("seeds", {}).get("disjoint"):
        print("refusing: the spec's seed ranges are not disjoint from its evaluation seeds")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    jobs = generation_jobs(spec, out, split, limit)
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            done = list(pool.map(make_item_job, jobs))
    else:
        done = [make_item_job(j) for j in jobs]
    items = [ItemRecord.model_validate(r) for r in done]
    storage.write_manifest(out, items, {"split": split, "curriculum": spec.get("slug", "")})
    certified = sum(i.certified for i in items)
    print(f"{split}: {certified}/{len(items)} certified items written to {out}")
    return 0 if certified else 1


__all__ = [
    "analyze",
    "export_kit",
    "export_pack",
    "generate_from_spec",
    "plan",
    "regime",
    "seeds_disjoint",
]
