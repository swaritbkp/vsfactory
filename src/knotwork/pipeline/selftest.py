"""`knot selftest` (SPEC 9.1): golden end-to-end self test.

Generates the golden dataset (2 base items per family of every plugin, seed 1234, offline beacon, with
twins) into ``artifacts/selftest/data/<plugin>/`` plus a combined ``data/index.jsonl``, re-verifies it
file-only, and compares {bundle_id, canonical answer, verdict, id-buffer digest, mask digests} per
slot with ``tests/golden/golden.json`` (exact). ``--update-golden`` rewrites the golden file.
The report holds no timings, so two runs produce byte-identical reports.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from knotwork.devtools.kernel_audit import kernel_sha256 as audit_kernel_sha256
from knotwork.knotcore import engine
from knotwork.knotcore.seeds import SELFTEST_EXPECTED, json_digest, selftest_vector
from knotwork.pipeline.generate import run_generate
from knotwork.pipeline.io import append_jsonl, read_json, read_jsonl, write_json
from knotwork.pipeline.verify import run_verify
from knotwork.plugins.registry import get_plugin
from knotwork.render.canvas import REPO_ROOT

SEED = 1234
PER_FAMILY = 2
PLUGINS = ("tangle", "gauge", "panel")
PROFILE = REPO_ROOT / "configs" / "default_profile.yaml"


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def snapshot(data: Path, plugin: str) -> list[dict[str, Any]]:
    """Golden rows of one plugin dataset, in index (slot) order."""
    rows = []
    for r in read_jsonl(data / plugin / "index.jsonl"):
        task = read_json(data / plugin / r["dir"] / "task.json")
        files = task["files"]
        masks = {k: v for k, v in sorted(files.items()) if k.startswith("masks/")}
        rows.append(
            {
                "family": r["family"],
                "role": r["role"],
                "bundle_id": r["bundle_id"],
                "answer": task["answer"]["canonical"],
                "verdict": task["certificate"]["verdict"],
                "id_buffer": files["id_buffer.png"],
                "masks": json_digest(masks),
            }
        )
    return rows


#: row fields that are raster bits (bundle ids digest beauty.png): bit-exact only on the platform that
#: recorded the golden (libm / SIMD float differences in Lanczos and the realism finish, D-35)
RASTER_KEYS = ("bundle_id", "id_buffer", "masks")


def platform_tag() -> str:
    return f"{sys.platform}-{platform.machine().lower()}"


def _strip(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if k not in RASTER_KEYS} for r in rows]


def _diff(want: list[dict[str, Any]], got: list[dict[str, Any]]) -> str:
    if len(want) != len(got):
        return f"{len(got)} bundles, golden has {len(want)}"
    for i, (a, b) in enumerate(zip(want, got, strict=True)):
        if a != b:
            keys = sorted(k for k in a if a.get(k) != b.get(k))
            return f"row {i} ({a.get('family')}/{a.get('role')}): {', '.join(keys)} differ"
    return ""


def run_selftest(
    golden: Path,
    update_golden: bool,
    out_dir: Path = Path("artifacts/selftest"),
    workers: int = 0,
) -> int:
    out_dir, golden = Path(out_dir), Path(golden)
    data = out_dir / "data"
    shutil.rmtree(data, ignore_errors=True)
    data.mkdir(parents=True)
    checks = [
        _check("seeds_kat", selftest_vector() == SELFTEST_EXPECTED, "seed known-answer vector"),
        _check(
            "kernel_sha_parity",
            engine.kernel_sha256() == audit_kernel_sha256(),
            "engine.kernel_sha256() == devtools kernel_sha256()",
        ),
    ]
    snap: dict[str, Any] = {
        "seed": SEED,
        "per_family": PER_FAMILY,
        "kernel_sha256": engine.kernel_sha256(),
        "platform": platform_tag(),
        "plugins": {},
    }
    index: list[dict[str, Any]] = []
    for name in PLUGINS:
        fams = [f.value for f in get_plugin(name).families]
        plan = [f for f in fams for _ in range(PER_FAMILY)]
        rc = run_generate(name, len(plan), data / name, SEED, "offline", PROFILE, "dev", True,
                          None, workers, plan=plan)  # fmt: skip
        checks.append(
            _check(f"generate:{name}", rc == 0, f"{len(plan)} base items ({PER_FAMILY} per family)")
        )
        if rc != 0:
            continue
        snap["plugins"][name] = snapshot(data, name)
        index += [
            dict(r, dir=f"{name}/{r['dir']}") for r in read_jsonl(data / name / "index.jsonl")
        ]
    append_jsonl(data / "index.jsonl", index)
    vrc = run_verify(data, out_dir / "verify_report.json", PROFILE)
    vrep = read_json(out_dir / "verify_report.json")
    checks.append(
        _check(
            "verify", vrc == 0, f"{vrep['n_ok']}/{vrep['n_bundles']} bundles re-verified file-only"
        )
    )
    if update_golden:
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        checks.append(_check("golden_updated", True, str(golden.as_posix())))
    elif not golden.is_file():
        checks.append(_check("golden", False, f"missing {golden.as_posix()} (run --update-golden)"))
    else:
        want = json.loads(golden.read_text(encoding="utf-8"))
        checks.append(
            _check(
                "golden:kernel_sha256",
                want.get("kernel_sha256") == snap["kernel_sha256"],
                "kernel unchanged",
            )
        )
        same = want.get("platform") == snap["platform"]
        for name in PLUGINS:
            got, ref = snap["plugins"].get(name, []), want.get("plugins", {}).get(name, [])
            if same:  # reference platform: everything bit-exact
                diff = _diff(ref, got)
                checks.append(
                    _check(f"golden:{name}", not diff, diff or f"{len(got)} bundles match exactly")
                )
                continue
            # other platforms: answers, verdicts, families and roles stay exact; raster bits reported
            diff = _diff(_strip(ref), _strip(got))
            checks.append(_check(f"golden:{name}", not diff, diff or f"{len(got)} bundles: answers, "
                                 f"verdicts and slot structure match the golden"))  # fmt: skip
            moved = sum(
                1
                for a, b in zip(ref, got, strict=False)
                if any(a.get(k) != b.get(k) for k in RASTER_KEYS)
            )
            checks.append(_check(f"golden:{name}:raster_portability", True,
                                 f"{moved}/{len(got)} rows differ in raster digests on {snap['platform']} "
                                 f"(golden recorded on {want.get('platform')}; D-35)"))  # fmt: skip
    report = {"ok": all(c["ok"] for c in checks), "checks": checks}
    write_json(out_dir / "report.json", report)
    print(json.dumps({"ok": report["ok"], "failed": [c["name"] for c in checks if not c["ok"]]}))
    return 0 if report["ok"] else 1


__all__ = ["run_selftest", "snapshot"]
