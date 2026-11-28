"""`knot generate` (SPEC 8.4): exact family schedule, per-slot deterministic attempts, twins,
certification, quarantine, and the dataset files of SPEC 3, for any number of items.

Determinism: slot i depends only on (root, plugin, i, attempt); slots run in any number of worker
processes and are consumed in slot order. Results are streamed: bundles are moved into place and
index / quarantine / lineage rows appended as each slot finishes, with a bounded number of slots in
flight, optionally into `bundles/shard-NNNNN/` directories.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from knotwork.freshness import lineage
from knotwork.freshness.beacon import fetch
from knotwork.freshness.receipt import build_receipt
from knotwork.knotcore.contracts import (
    DifficultySpec,
    Family,
    FreshnessReceipt,
    Split,
    TaskBundle,
    TwinRefs,
)
from knotwork.knotcore.seeds import SeedTree, json_digest
from knotwork.pipeline import ood, telemetry
from knotwork.pipeline.certify import Candidate, quarantine_of, run_candidate
from knotwork.pipeline.io import append_jsonl, write_json, write_task
from knotwork.pipeline.twins import build_twin
from knotwork.plugins.panel.programs import family_counts
from knotwork.plugins.registry import get_plugin, twin_unavailable

#: slots submitted ahead of the one being consumed, per worker process
IN_FLIGHT_PER_WORKER = 4


def load_profile(path: Path | str) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"profile {path} must be a mapping")
    if "rack_patch" not in data and "tangle" in data:
        from knotwork.plugins.rack_patch.plugin import rack_section

        data["rack_patch"] = rack_section(data)
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
    prof = dict(profile, _family=family, _split=job["split"], _ordinal=job.get("ordinal"),
                _balance_seed=receipt.root_seed)  # fmt: skip
    work = Path(job["work"]) / f"slot{i:05d}"
    spec = DifficultySpec.model_validate(job["difficulty"]) if job.get("difficulty") else None
    quarantine: list[dict[str, Any]] = []
    n_cand = 0
    for a in range(int(profile["generation"]["max_attempts_per_item"])):
        at = slot.child("attempt", a)
        adir = work / f"a{a}"
        try:
            scene = plugin.sample_scene(at.child("scene"), prof, spec)
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


def auto_workers() -> int:
    """Pool size for `--workers 0`: cpu_count - 1, or KNOTWORK_AUTO_WORKERS when set (so that nested
    pools, e.g. under pytest-xdist, do not oversubscribe the machine)."""
    cap = int(os.environ.get("KNOTWORK_AUTO_WORKERS", "0") or 0)
    return cap if cap > 0 else max(1, (os.cpu_count() or 2) - 1)


def iter_jobs(jobs: Iterable[dict[str, Any]], fn: Any, workers: int) -> Iterator[dict[str, Any]]:
    """`fn` over `jobs`, yielding results in job order while keeping only a few jobs in flight."""
    if workers <= 1:
        for job in jobs:
            yield fn(job)
        return
    # one BLAS/OpenMP thread per worker process: N workers x N threads thrashes the CPU
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        pending: deque[Future[dict[str, Any]]] = deque()
        for job in jobs:
            pending.append(ex.submit(fn, job))
            if len(pending) >= IN_FLIGHT_PER_WORKER * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def pool_size(workers: int, n_jobs: int) -> int:
    workers = auto_workers() if workers == 0 else workers
    return min(workers, max(1, n_jobs))


def run_jobs(jobs: list[dict[str, Any]], fn: Any, workers: int) -> list[dict[str, Any]]:
    return list(iter_jobs(jobs, fn, pool_size(workers, len(jobs))))


def ordinals_of(plan: list[str]) -> list[int]:
    """For each slot, how many earlier slots share its family (D-34)."""
    seen: Counter[str] = Counter()
    out = []
    for family in plan:
        out.append(seen[family])
        seen[family] += 1
    return out


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
    holdout_families: list[str] | None = None,
    holdout_params: list[str] | None = None,
    shard_size: int = 0,
    difficulty: dict[str, Any] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """SPEC 8.4. ``plan`` (not exposed by the CLI) fixes the slot -> family list explicitly; the
    golden selftest uses it for exactly two items per family. Holdouts and `split="ood"` follow
    `pipeline.ood`; ``difficulty`` holds `DifficultySpec` fields; ``progress(done, n)`` is called
    after every slot."""
    t0 = time.perf_counter()
    base_prof = load_profile(profile)
    held = ood.resolve(plugin, split, base_prof, families, holdout_families, holdout_params)
    prof, families = held.profile, held.families
    spec = DifficultySpec.model_validate(difficulty) if difficulty else None
    receipt = build_receipt(fetch(beacon, seed, base_prof), base_prof)
    out = Path(out)
    if (out / "bundles").exists():
        shutil.rmtree(out / "bundles")
    out.mkdir(parents=True, exist_ok=True)
    for f in ("index.jsonl", "quarantine.jsonl", lineage.FILENAME, telemetry.FILENAME):
        (out / f).unlink(missing_ok=True)
    for f in ("index.jsonl", "quarantine.jsonl", lineage.FILENAME):
        (out / f).touch()
    work = out / ".work"
    shutil.rmtree(work, ignore_errors=True)
    if plan is not None and len(plan) != n:
        raise ValueError(f"plan has {len(plan)} slots, expected {n}")
    plan = (
        list(plan) if plan is not None else schedule(plugin, n, prof, families, receipt.root_seed)
    )
    burned = sorted(_burned(prof)) if split in ("test", "sealed") else []
    ordinals = ordinals_of(plan)
    spec_data = spec.model_dump(exclude_none=True) if spec else None
    receipt_data = receipt.model_dump(mode="json")
    jobs = ({"plugin": plugin, "slot": i, "family": plan[i], "ordinal": ordinals[i], "profile": prof,
             "split": split, "twins": twins, "receipt": receipt_data, "work": str(work), "burned": burned,
             "difficulty": spec_data} for i in range(n))  # fmt: skip
    size = pool_size(workers, n)
    by_family: Counter[str] = Counter()
    by_code: Counter[str] = Counter()
    tally = telemetry.Telemetry()
    certificates: dict[str, str] = {}
    n_bundles = n_certified = n_candidates = n_quarantined = 0
    failed: list[int] = []
    for done, r in enumerate(iter_jobs(jobs, run_slot, size), 1):
        i = r["slot"]
        shard = f"shard-{i // shard_size:05d}/" if shard_size > 0 else ""
        rows, nodes = [], []
        for role, d, task in r["bundles"]:
            rel = f"bundles/{shard}{task['bundle_id']}"
            dest = out / rel
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(d, dest)
            if role == "base":
                by_family[task["family"]] += 1
            rows.append({"bundle_id": task["bundle_id"], "plugin": plugin, "family": task["family"],
                         "split": split, "role": role, "base_id": task["twins"]["base_id"],
                         "answer_canonical": task["answer"]["canonical"], "dir": rel,
                         "scene_id": task["scene_id"]})  # fmt: skip
            nodes += lineage.task_nodes(task, certificates)
        quarantine = [q for q in r["quarantine"] if "codes" in q]
        for q in quarantine:
            by_code.update(q["codes"])
        append_jsonl(out / "index.jsonl", rows)
        append_jsonl(out / "quarantine.jsonl", quarantine)
        append_jsonl(out / lineage.FILENAME, nodes)
        tally.add(plan[i], r)
        n_bundles += len(rows)
        n_certified += int(r["ok"])
        n_candidates += r["n_candidates"]
        n_quarantined += len(quarantine)
        if not r["ok"]:
            failed.append(i)
        certificates.clear()  # twins only refer to certificates of their own slot
        shutil.rmtree(work / f"slot{i:05d}", ignore_errors=True)
        if progress is not None:
            progress(done, n)
    seconds = time.perf_counter() - t0
    rate = round(n_quarantined / n_candidates, 6) if n_candidates else 0.0
    stats = {
        "plugin": plugin, "n_requested": n, "n_base_certified": n_certified, "n_bundles": n_bundles,
        "n_candidates": n_candidates, "n_quarantined": n_quarantined, "false_quarantine_rate": rate,
        "families_expected": sorted(set(plan)), "schedule": plan, "by_family": dict(sorted(by_family.items())),
        "by_code": dict(sorted(by_code.items())), "seconds": round(seconds, 1), "workers": workers,
        "split": split, "shard_size": shard_size, "difficulty": spec_data, "holdout": held.record,
        "profile_overrides": {plugin: prof[plugin], "tangle": prof["tangle"]} if held.record else None,
    }  # fmt: skip
    write_json(out / "stats.json", stats)
    write_json(out / "freshness_receipt.json", receipt)
    telemetry.write(out, tally.report(seconds, plugin=plugin, split=split, n_requested=n,
                                      workers=size, seed=seed, difficulty=spec_data,
                                      holdout=held.record))  # fmt: skip
    shutil.rmtree(work, ignore_errors=True)
    if failed:
        print(f"generate: slots {failed[:20]} produced no certified item", file=sys.stderr)
        return 1
    print(f"generate: {plugin} {n} base items, {n_bundles} bundles, fq={rate:.3f}")
    return 0


def _burned(profile: dict[str, Any]) -> set[str]:
    from knotwork.freshness.ledger import burned_scene_ids

    return burned_scene_ids(profile)


__all__ = ["iter_jobs", "load_profile", "run_generate", "run_jobs", "run_slot", "schedule"]
