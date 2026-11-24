"""Task 3.8: file-only `knot verify` -- clean datasets pass, tampering is detected, no plugin code."""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from knotwork.pipeline.io import read_json, read_jsonl
from knotwork.pipeline.verify import bundle_dirs, run_verify

REPO = Path(__file__).resolve().parents[2]
VERIFY_SRC = REPO / "src" / "knotwork" / "pipeline" / "verify.py"


def _one(datasets: Path, tmp: Path, name: str) -> Path:
    """Copy of the first gauge bundle as a one-item dataset."""
    src = datasets / "data_gauge"
    row = read_jsonl(src / "index.jsonl")[0]
    ds = tmp / name
    shutil.copytree(src / row["dir"], ds / row["dir"])
    (ds / "index.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return ds


def _codes(report: Path) -> set[str]:
    rep = read_json(report)
    return {c for f in rep["failures"] for c in f["codes"]}


def test_clean_dataset_verifies(datasets: Path, tmp_path: Path) -> None:
    out = tmp_path / "v.json"
    assert run_verify(datasets / "data_gauge", out) == 0
    rep = read_json(out)
    assert rep["ok"] is True and rep["n_ok"] == rep["n_bundles"] == 6 and rep["failures"] == []
    assert len(bundle_dirs(datasets)) == sum(
        len(read_jsonl(d / "index.jsonl")) for d in datasets.iterdir()
    )


def test_single_bundle_dir(datasets: Path, tmp_path: Path) -> None:
    ds = _one(datasets, tmp_path, "single")
    bdir = next((ds / "bundles").iterdir())
    assert run_verify(bdir, tmp_path / "r.json") == 0


def test_pixel_tamper_detected(datasets: Path, tmp_path: Path) -> None:
    ds = _one(datasets, tmp_path, "pixel")
    bdir = next((ds / "bundles").iterdir())
    arr = np.asarray(Image.open(bdir / "id_buffer.png"), dtype=np.uint16).copy()
    arr[512, 512] ^= 1
    Image.fromarray(arr).save(bdir / "id_buffer.png")
    assert run_verify(ds, tmp_path / "r.json") == 1
    assert "E_DIGEST_MISMATCH" in _codes(tmp_path / "r.json")


def test_beauty_and_claims_tamper_detected(datasets: Path, tmp_path: Path) -> None:
    ds = _one(datasets, tmp_path, "beauty")
    bdir = next((ds / "bundles").iterdir())
    img = np.asarray(Image.open(bdir / "beauty.png").convert("RGB")).copy()
    img[100, 100] = 255 - img[100, 100]
    Image.fromarray(img).save(bdir / "beauty.png")
    claims = read_json(bdir / "claims.json")
    claims["claims"] = claims["claims"][:1]
    (bdir / "claims.json").write_text(json.dumps(claims), encoding="utf-8")
    assert run_verify(ds, tmp_path / "r.json") == 1
    rep = read_json(tmp_path / "r.json")
    assert (
        "beauty.png" in rep["failures"][0]["detail"]
        and "claims.json" in rep["failures"][0]["detail"]
    )


def test_answer_tamper_detected(datasets: Path, tmp_path: Path) -> None:
    ds = _one(datasets, tmp_path, "answer")
    bdir = next((ds / "bundles").iterdir())
    task = read_json(bdir / "task.json")
    task["answer"]["canonical"] += "X"
    (bdir / "task.json").write_text(json.dumps(task), encoding="utf-8")
    assert run_verify(ds, tmp_path / "r.json") == 1
    assert "E_DIGEST_MISMATCH" in _codes(tmp_path / "r.json")


def test_schema_violation_detected(datasets: Path, tmp_path: Path) -> None:
    ds = _one(datasets, tmp_path, "schema")
    bdir = next((ds / "bundles").iterdir())
    task = read_json(bdir / "task.json")
    task["canary"] = "not-a-canary"
    (bdir / "task.json").write_text(json.dumps(task), encoding="utf-8")
    assert run_verify(ds, tmp_path / "r.json") == 1
    assert _codes(tmp_path / "r.json") == {"E_SCHEMA"}


def test_verify_never_imports_plugins() -> None:
    tree = ast.parse(VERIFY_SRC.read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not [m for m in names if m.startswith("knotwork.plugins")]
    code = (
        "import sys, knotwork.pipeline.verify; "
        "print(sorted(m for m in sys.modules if m.startswith('knotwork.plugins')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO
    )
    assert out.stdout.strip() == "[]"
