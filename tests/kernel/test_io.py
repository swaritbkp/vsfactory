"""Task 1.1: bundle loader."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from knotwork.knotcore.contracts import ErrorCode
from knotwork.knotcore.io import KernelFault, load_bundle


def test_load_bundle_arrays(x_dir: Path) -> None:
    b = load_bundle(x_dir)
    assert b.beauty.shape == (1024, 1024, 3) and b.beauty.dtype == np.uint8
    assert b.ids.dtype == np.uint16 and b.lum.shape == (1024, 1024)
    assert b.lum.min() >= 0.0 and b.lum.max() <= 1.0
    assert b.beauty768.shape == (768, 768, 3)
    assert set(b.masks) == {o.obj_id for o in b.manifest.objects}
    assert all(m.dtype == np.bool_ for m in b.masks.values())
    assert len(b.claims.claims) == 5


def test_invalid_json_is_schema_error(x_dir: Path, tmp_path: Path) -> None:
    d = tmp_path / "b"
    shutil.copytree(x_dir, d)
    (d / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(KernelFault) as e:
        load_bundle(d)
    assert e.value.code is ErrorCode.E_SCHEMA


def test_invalid_schema_and_shape(x_dir: Path, tmp_path: Path) -> None:
    d = tmp_path / "c"
    shutil.copytree(x_dir, d)
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    man["objects"][0]["obj_id"] = 0
    (d / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(KernelFault) as e:
        load_bundle(d)
    assert e.value.code is ErrorCode.E_SCHEMA
    d2 = tmp_path / "d"
    shutil.copytree(x_dir, d2)
    Image.new("RGB", (512, 512)).save(d2 / "beauty.png")
    with pytest.raises(KernelFault) as e:
        load_bundle(d2)
    assert e.value.code is ErrorCode.E_SCHEMA


def test_no_plugin_imports_in_kernel_tests() -> None:
    import ast

    for p in Path(__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom | ast.Import):
                names = (
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else [a.name for a in node.names]
                )
                assert not any(n.startswith("knotwork.plugins") for n in names), p
