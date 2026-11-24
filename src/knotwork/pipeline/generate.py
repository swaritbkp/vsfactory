"""`knot generate` (SPEC 8.4): exact family schedule, per-slot deterministic attempts, twins,
certification, quarantine, and the dataset files of SPEC 3.

Determinism: slot i depends only on (root, plugin, i, attempt); slots run in any order / any number of
worker processes and results are sorted by slot before anything is written.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from knotwork.freshness.beacon import fetch
from knotwork.freshness.receipt import build_receipt
from knotwork.knotcore.contracts import Family, FreshnessReceipt, Split, TaskBundle, TwinRefs
from knotwork.knotcore.seeds import SeedTree, json_digest
from knotwork.pipeline.certify import Candidate, quarantine_of, run_candidate
from knotwork.pipeline.io import append_jsonl, write_json, write_task
from knotwork.pipeline.twins import build_twin
from knotwork.plugins.panel.programs import family_counts
from knotwork.plugins.registry import get_plugin, twin_unavailable


def load_profile(path: Path | str) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"profile {path} must be a mapping")
    return data


def schedule(
    plugin: str, n: int, profile: dict[str, Any], families: list[str] | None, root: int
) -> list[str]:
    """Slot -> family: exact counts (largest remainder; premise families exact for panel), then a
    permutation drawn from the canonical seed path (plugin, "schedule")."""
    weights = {str(k): float(v) for k, v in profile[plugin]["families"].items()}
    if families:
        unknown = set(families) - set(weights)
        if unknown:
            raise ValueError(f"unknown families for {plugin}: {sorted(unknown)}")
        weights = {f: w for f, w in weights.items() if f in families}
    counts = family_counts(n, weights)
    flat = [f for f in sorted(counts) for _ in range(counts[f])]
    perm = SeedTree(root).child(plugin, "schedule").rng().permutation(len(flat))
    return [flat[int(i)] for i in perm]


def _task(
    c: Candidate,
    plugin: str,
    split: str,
    seed_path: str,
    receipt: FreshnessReceipt,
    twins: TwinRefs,
) -> TaskBundle:
    return TaskBundle(
        bundle_id=c.bundle_id, plugin=plugin, family=c.question.family, split=Split(split), seed_path=seed_path,
        scene_id=json_digest(c.scene.model_dump(mode="json")), question=c.question, answer=c.answer,
        certificate=c.certificate, freshness_receipt_id=receipt.receipt_id, twins=twins, canary=receipt.canary,
        files=c.files,
    )  # fmt: skip


def run_slot(job: dict[str, Any]) -> dict[str, Any]:
    """Generate one slot: up to max_attempts candidates (+ twins). Pure function of its job dict."""
    name, i, family, profile = job["plugin"], job["slot"], job["family"], job["profile"]
    plugin = get_plugin(name)
    unavailable = twin_unavailable(name)
    receipt = FreshnessReceipt.model_validate(job["receipt"])
    slot = SeedTree(receipt.root_seed).child(name, "slot", i)
    prof = dict(profile, _family=family, _split=job["split"], _ordinal=job.get("ordinal"))
    work = Path(job["work"]) / f"slot{i:05d}"
    quarantine: list[dict[str, Any]] = []
    n_cand = 0
    for a in range(int(profile["generation"]["max_attempts_per_item"])):
        at = slot.child("attempt", a)
        adir = work / f"a{a}"
        try:
            scene = plugin.sample_scene(at.child("scene"), prof)
            q = plugin.questions(scene, at.child("question"), Family(family))
        except (unavailable, RuntimeError) as exc:
            quarantine.append({"skip": str(exc)[:200]})
            continue
        if job.get("burned") and json_digest(scene.model_dump(mode="json")) in job["burned"]:
            continue
        base = run_candidate(plugin, scene, q, adir / "base", profile)
        n_cand += 1
        if not base.certified:
            quarantine.append(
                quarantine_of(
                    at.path_str, name, q.family, at.path_str, base.certificate
                ).model_dump(mode="json")
            )
            continue
        if not job["twins"]:
            task = _task(base, name, job["split"], at.path_str, receipt, TwinRefs(role="base"))
            write_task(base.dir, task)
            return {"slot": i, "ok": True, "bundles": [("base", str(base.dir), task.model_dump(mode="json"))],
                    "quarantine": quarantine, "n_candidates": n_cand}  # fmt: skip
        causal = build_twin("causal", plugin, base, at, adir, profile, family)
        nuis = build_twin("nuisance", plugin, base, at, adir, profile, family)
        n_cand += causal.n_candidates + nuis.n_candidates
        quarantine += [r.model_dump(mode="json") for r in causal.quarantine + nuis.quarantine]
        if causal.candidate is None or nuis.candidate is None:
            continue
        cc, nc = causal.candidate, nuis.candidate
        tasks = [
            ("base", base, at.path_str, TwinRefs(role="base", causal_id=cc.bundle_id, nuisance_id=nc.bundle_id)),
            ("causal", cc, at.child("twin_causal").path_str,
             TwinRefs(role="causal", base_id=base.bundle_id, edit=causal.edit, pixel_diff_frac=round(causal.diff or 0.0, 6))),
            ("nuisance", nc, at.child("twin_nuisance").path_str,
             TwinRefs(role="nuisance", base_id=base.bundle_id, edit=nuis.edit, pixel_diff_frac=round(nuis.diff or 0.0, 6))),
        ]  # fmt: skip
        bundles = []
        for role, cand, path, refs in tasks:
            task = _task(cand, name, job["split"], path, receipt, refs)
            write_task(cand.dir, task)
            bundles.append((role, str(cand.dir), task.model_dump(mode="json")))
        return {
            "slot": i,
            "ok": True,
            "bundles": bundles,
            "quarantine": quarantine,
            "n_candidates": n_cand,
        }
    return {"slot": i, "ok": False, "bundles": [], "quarantine": quarantine, "n_candidates": n_cand}


def run_jobs(jobs: list[dict[str, Any]], fn: Any, workers: int) -> list[dict[str, Any]]:
    workers = max(1, (os.cpu_count() or 2) - 1) if workers == 0 else workers
    workers = min(workers, max(1, len(jobs)))
    if workers == 1:
        return [fn(j) for j in jobs]
    # one BLAS/OpenMP thread per worker process: N workers x N threads thrashes the CPU
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, jobs))


def run_generate(
    plugin: str,
    n: int,
    out: Path,
    seed: int,
    beacon: str,
    profile: Path,
    split: str,
    twins: bool,
    families: list[str] | None,
    workers: int,
    plan: list[str] | None = None,
) -> int:
    """SPEC 8.4. ``plan`` (not exposed by the CLI) fixes the slot -> family list explicitly; the
    golden selftest uses it for exactly two items per family."""
    t0 = time.perf_counter()
    prof = load_profile(profile)
    receipt = build_receipt(fetch(beacon, seed, prof), prof)
    out = Path(out)
    if (out / "bundles").exists():
        shutil.rmtree(out / "bundles")
    for f in ("index.jsonl", "quarantine.jsonl"):
        (out / f).unlink(missing_ok=True)
    work = out / ".work"
    shutil.rmtree(work, ignore_errors=True)
    if plan is not None and len(plan) != n:
        raise ValueError(f"plan has {len(plan)} slots, expected {n}")
    plan = (
        list(plan) if plan is not None else schedule(plugin, n, prof, families, receipt.root_seed)
    )
    burned = sorted(_burned(prof)) if split in ("test", "sealed") else []
    ordinals = [plan[:i].count(plan[i]) for i in range(n)]  # k-th slot of its family (D-34)
    jobs = [{"plugin": plugin, "slot": i, "family": plan[i], "ordinal": ordinals[i], "profile": prof, "split": split, "twins": twins,
             "receipt": receipt.model_dump(mode="json"), "work": str(work), "burned": burned} for i in range(n)]  # fmt: skip
    results = sorted(run_jobs(jobs, run_slot, workers), key=lambda r: r["slot"])
    index: list[dict[str, Any]] = []
    by_family: Counter[str] = Counter()
    by_code: Counter[str] = Counter()
    quarantine = [q for r in results for q in r["quarantine"] if "codes" in q]
    for r in results:
        for role, d, task in r["bundles"]:
            dest = out / "bundles" / task["bundle_id"]
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(d, dest)
            if role == "base":
                by_family[task["family"]] += 1
            index.append({"bundle_id": task["bundle_id"], "plugin": plugin, "family": task["family"], "split": split,
                          "role": role, "base_id": task["twins"]["base_id"], "answer_canonical": task["answer"]["canonical"],
                          "dir": f"bundles/{task['bundle_id']}", "scene_id": task["scene_id"]})  # fmt: skip
    for q in quarantine:
        by_code.update(q["codes"])
    append_jsonl(out / "index.jsonl", index)
    append_jsonl(out / "quarantine.jsonl", quarantine)
    n_cand = sum(r["n_candidates"] for r in results)
    stats = {
        "plugin": plugin, "n_requested": n, "n_base_certified": sum(r["ok"] for r in results), "n_bundles": len(index),
        "n_candidates": n_cand, "n_quarantined": len(quarantine),
        "false_quarantine_rate": round(len(quarantine) / n_cand, 6) if n_cand else 0.0,
        "families_expected": sorted(set(plan)), "schedule": plan, "by_family": dict(sorted(by_family.items())),
        "by_code": dict(sorted(by_code.items())), "seconds": round(time.perf_counter() - t0, 1),
        "workers": workers,
    }  # fmt: skip
    write_json(out / "stats.json", stats)
    write_json(out / "freshness_receipt.json", receipt)
    shutil.rmtree(work, ignore_errors=True)
    failed = [r["slot"] for r in results if not r["ok"]]
    if failed:
        print(f"generate: slots {failed} produced no certified item", file=sys.stderr)
        return 1
    print(
        f"generate: {plugin} {n} base items, {len(index)} bundles, fq={stats['false_quarantine_rate']:.3f}"
    )
    return 0


def _burned(profile: dict[str, Any]) -> set[str]:
    from knotwork.freshness.ledger import burned_scene_ids

    return burned_scene_ids(profile)


__all__ = ["load_profile", "run_generate", "run_slot", "schedule"]
