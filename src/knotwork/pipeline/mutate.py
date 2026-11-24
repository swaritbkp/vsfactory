"""`knot mutate` (SPEC 10): inject each render fault M1-M10 and require the kernel to quarantine every
injection with the expected code; the same scene rendered clean is the control.

Scenes come from the canonical path ("mutate", fault, i, "attempt", a); a fault is injected only into
scenes where it is effective (`fault_effective`, <= 20 attempts). Claims, question and D1 always
describe the clean scene, so only pixels can reveal the fault.
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from knotwork.freshness.beacon import offline
from knotwork.freshness.receipt import protocol_commit
from knotwork.knotcore.contracts import FAULT_EXPECTED, FAULT_PLUGIN, FaultCode
from knotwork.knotcore.seeds import SeedTree, derive_root_seed
from knotwork.pipeline.certify import run_candidate
from knotwork.pipeline.generate import load_profile, run_jobs
from knotwork.pipeline.io import write_json
from knotwork.plugins.registry import fault_effective, get_plugin

MAX_EFFECTIVE_ATTEMPTS = 20


def _root(seed: int, profile: dict[str, Any]) -> int:
    """Same root derivation as `generate --beacon offline --seed <seed>` (SPEC 5)."""
    commit = protocol_commit(list(profile["freshness"]["protocol_files"]))
    return derive_root_seed(offline(seed).randomness_hex, commit)


def mutate_one(job: dict[str, Any]) -> dict[str, Any]:
    """Inject one fault into one effective scene; certify faulty + clean renders."""
    fault, i, profile = FaultCode(job["fault"]), job["i"], job["profile"]
    name = FAULT_PLUGIN[fault]
    plugin = get_plugin(name)
    work = Path(job["work"]) / f"{fault.value}_{i}"
    for a in range(MAX_EFFECTIVE_ATTEMPTS):
        at = SeedTree(job["root"]).child("mutate", fault.value, i, "attempt", a)
        family = plugin.families[i % len(plugin.families)]
        try:
            scene = plugin.sample_scene(at.child("scene"), dict(profile, _family=family.value))
            q = plugin.questions(scene, at.child("question"), family)
        except RuntimeError:
            continue
        if not fault_effective(name, scene, fault):
            continue
        bad = run_candidate(plugin, scene, q, work / "faulty", profile, fault)
        clean = run_candidate(plugin, scene, q, work / "clean", profile)
        return {"fault": fault.value, "i": i, "plugin": name, "effective": True, "seed_path": at.path_str,
                "codes": [c.value for c in bad.certificate.codes], "faulty_dir": str(bad.dir),
                "clean_codes": [c.value for c in clean.certificate.codes], "bundle_id": bad.bundle_id}  # fmt: skip
    return {
        "fault": fault.value,
        "i": i,
        "plugin": name,
        "effective": False,
        "codes": [],
        "clean_codes": [],
    }


def run_mutate(
    fault: str,
    n: int,
    seed: int,
    out: Path,
    profile: Path,
    keep_dir: Path | None = None,
    workers: int = 0,
) -> int:
    prof = load_profile(profile)
    faults = list(FaultCode) if fault == "all" else [FaultCode(fault)]
    out = Path(out)
    work = keep_dir or out.parent / f".{out.stem}_work"
    shutil.rmtree(work, ignore_errors=True)
    jobs = [{"fault": f.value, "i": i, "profile": prof, "root": _root(seed, prof), "work": str(work)}
            for f in faults for i in range(n)]  # fmt: skip
    results = sorted(
        run_jobs(jobs, mutate_one, workers), key=lambda r: (int(r["fault"][1:]), r["i"])
    )
    rows: dict[str, Any] = {}
    for f in faults:
        rs = [r for r in results if r["fault"] == f.value and r["effective"]]
        exp = FAULT_EXPECTED[f].value
        rows[f.value] = {
            "expected": exp, "plugin": FAULT_PLUGIN[f], "injected": len(rs),
            "caught": sum(bool(r["codes"]) for r in rs), "caught_exact": sum(exp in r["codes"] for r in rs),
            "primary_match": sum(bool(r["codes"]) and r["codes"][0] == exp for r in rs),
            "codes_seen": dict(sorted(Counter(c for r in rs for c in r["codes"]).items())),
            "examples": [{"bundle_id": r["bundle_id"], "dir": r["faulty_dir"], "codes": r["codes"]} for r in rs[:1]],
        }  # fmt: skip
    injected = sum(r["injected"] for r in rows.values())
    clean = [r for r in results if r["effective"]]
    q_clean = sum(bool(r["clean_codes"]) for r in clean)
    report: dict[str, Any] = {
        "n_per_fault": n, "faults": rows,
        "catch_rate": round(sum(r["caught"] for r in rows.values()) / injected, 6) if injected else 0.0,
        "exact_catch_rate": round(sum(r["caught_exact"] for r in rows.values()) / injected, 6) if injected else 0.0,
        "clean": {"n": len(clean), "quarantined": q_clean,
                  "false_quarantine_rate": round(q_clean / len(clean), 6) if clean else 0.0,
                  "codes_seen": dict(sorted(Counter(c for r in clean for c in r["clean_codes"]).items()))},
    }  # fmt: skip
    write_json(out, report)
    if keep_dir is None:
        shutil.rmtree(work, ignore_errors=True)
    ok = report["exact_catch_rate"] == 1.0 and injected == n * len(faults)
    print(f"mutate: exact catch {report['exact_catch_rate']:.3f} over {injected} injections; "
          f"clean false-quarantine {report['clean']['false_quarantine_rate']:.3f}")  # fmt: skip
    return 0 if ok and report["clean"]["false_quarantine_rate"] <= 0.10 else 1
