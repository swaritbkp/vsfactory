"""File-only `knot verify` (SPEC 9.1): re-derive every bundle's evidence from the files on disk.

For each bundle: validate `task.json` against the contracts (E_SCHEMA), recompute the decoded-array
file digests and the bundle id (E_DIGEST_MISMATCH), and re-run `knotcore.engine.certify` with the
stored D1/D2. No plugin code is imported or executed (a test asserts this module never imports
`knotwork.plugins`); the only trusted code involved is the kernel.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from knotwork.knotcore import engine
from knotwork.knotcore.contracts import ErrorCode, TaskBundle, Verdict
from knotwork.pipeline.io import bundle_id_for, file_digests, read_json, read_jsonl, write_json
from knotwork.render.canvas import REPO_ROOT

DEFAULT_PROFILE = REPO_ROOT / "configs" / "default_profile.yaml"


def bundle_dirs(path: Path) -> list[Path]:
    """A single bundle directory, a dataset (index.jsonl), or a directory of datasets."""
    path = Path(path)
    if (path / "task.json").is_file():
        return [path]
    if (path / "index.jsonl").is_file():
        return [path / r["dir"] for r in read_jsonl(path / "index.jsonl")]
    out: list[Path] = []
    for sub in sorted(p for p in path.iterdir() if (p / "index.jsonl").is_file()):
        out += [sub / r["dir"] for r in read_jsonl(sub / "index.jsonl")]
    return out


def verify_bundle(d: Path, kernel_cfg: dict[str, Any]) -> dict[str, Any] | None:
    """None when the bundle verifies, else a failure record {bundle_id, codes, detail}."""
    name = d.name
    try:
        task = TaskBundle.model_validate(read_json(d / "task.json"))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        return {"bundle_id": name, "codes": [ErrorCode.E_SCHEMA.value], "detail": str(exc)[:300]}
    try:
        files = file_digests(d)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "bundle_id": task.bundle_id,
            "codes": [ErrorCode.E_SCHEMA.value],
            "detail": str(exc)[:300],
        }
    codes: list[str] = []
    details: list[str] = []
    if files != task.files:
        bad = sorted(k for k in set(files) | set(task.files) if files.get(k) != task.files.get(k))
        codes.append(ErrorCode.E_DIGEST_MISMATCH.value)
        details.append("files: " + ", ".join(bad))
    bid = bundle_id_for(files, task.question, task.answer)
    if bid != task.bundle_id:
        if ErrorCode.E_DIGEST_MISMATCH.value not in codes:
            codes.append(ErrorCode.E_DIGEST_MISMATCH.value)
        details.append(f"bundle_id {task.bundle_id} != recomputed {bid}")
    cert = engine.certify(
        d, task.question, task.certificate.d1, task.certificate.d2, kernel_cfg, task.bundle_id
    )
    if cert.verdict is not Verdict.CERTIFIED:
        codes += [c.value for c in cert.codes if c.value not in codes] or [
            ErrorCode.E_D1_D3_MISMATCH.value
        ]
        details.append("re-certification: " + ",".join(c.value for c in cert.codes))
    elif cert.d3.canonical != task.certificate.d3.canonical:
        codes.append(ErrorCode.E_D1_D3_MISMATCH.value)
        details.append(f"d3 {cert.d3.canonical} != stored {task.certificate.d3.canonical}")
    if not codes:
        return None
    return {"bundle_id": task.bundle_id, "codes": codes, "detail": "; ".join(details)[:500]}


def run_verify(path: Path, out: Path, profile: Path = DEFAULT_PROFILE) -> int:
    prof = yaml.safe_load(Path(profile).read_text(encoding="utf-8"))
    dirs = bundle_dirs(Path(path))
    failures = [f for f in (verify_bundle(d, prof["kernel"]) for d in dirs) if f is not None]
    report = {
        "ok": not failures and bool(dirs),
        "n_bundles": len(dirs),
        "n_ok": len(dirs) - len(failures),
        "failures": failures,
    }
    write_json(Path(out), report)
    print(f"verify: {report['n_ok']}/{report['n_bundles']} bundles ok")
    return 0 if report["ok"] else 1


__all__ = ["bundle_dirs", "run_verify", "verify_bundle"]
