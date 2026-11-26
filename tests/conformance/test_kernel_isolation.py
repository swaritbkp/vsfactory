"""Kernel isolation conformance (layer 2 of 2; layer 1 is import-linter via `.importlinter`).

* the LOC budgets are reported as telemetry (warnings, never failures; D-38).
* the TCB contract holds: allow-listed imports only, no network / process / dynamic-code modules,
  no `SceneIR.payload` access, deterministic `kernel_sha256`.
* knotcore never imports plugins / higher layers / network / geometry-truth / UI libraries,
  directly, relatively, or dynamically.
* knotcore, and the whole of src/, contain no placeholder markers (the scanner must not flag itself).
"""

from __future__ import annotations

import configparser
import textwrap
import warnings
from pathlib import Path

import pytest

from knotwork.devtools import kernel_audit as ka


def test_loc_counter_definition() -> None:
    src = textwrap.dedent(
        '''
        """Module docstring
        spanning lines."""
        # a comment

        import os  # trailing comment


        def f(x):
            """Docstring."""
            s = """not a docstring
            but a string literal"""
            return x + 1
        '''
    )
    # counted: import os | def f | s = (2 lines) | return  -> 5
    assert ka.count_loc_source(src) == 5


def test_knotcore_loc_telemetry() -> None:
    rep = ka.loc_report()
    for message in rep.warnings():
        warnings.warn(message, stacklevel=1)
    assert rep.executable == rep.total - rep.per_file.get("contracts.py", 0)
    assert rep.executable > 0 and rep.summary().startswith("knotcore executable LOC")


def test_loc_over_budget_warns_without_failing() -> None:
    over = ka.LocReport(per_file={}, executable=9999, total=19999)
    assert len(over.warnings()) == 2 and not over.ok
    assert ka.LocReport(per_file={}, executable=1, total=2).warnings() == []


def test_tcb_contract_holds() -> None:
    rep = ka.tcb_report()
    assert rep.ok, rep.summary()
    assert rep.kernel_sha256 == ka.kernel_sha256()


@pytest.mark.parametrize(
    "line",
    [
        "import requests",
        "import torch",
        "import yaml",
        "import urllib.parse",
        "import subprocess",
        "from knotwork import render",
        "from knotwork.render import canvas",
        "eval('1')",
        "def f(scene):\n    return scene.payload",
        "def f(d):\n    return d['payload']",
        "def f(d):\n    return getattr(d, 'payload')",
        "from knotwork.knotcore.contracts import SceneIR",
    ],
)
def test_tcb_rejects_planted_violation(tmp_path: Path, line: str) -> None:
    kernel = tmp_path / "src" / "knotwork" / "knotcore"
    kernel.mkdir(parents=True)
    (kernel / "bad.py").write_text(line + "\n", encoding="utf-8")
    assert not ka.tcb_report(kernel, tmp_path / "src").ok, line


def test_tcb_accepts_allowed_imports(tmp_path: Path) -> None:
    kernel = tmp_path / "src" / "knotwork" / "knotcore"
    kernel.mkdir(parents=True)
    (kernel / "good.py").write_text(
        "import hashlib\nimport numpy as np\nfrom scipy import ndimage\nfrom PIL import Image\n"
        "from pydantic import BaseModel\nfrom skimage.graph import MCP_Geometric\n"
        "from rapidocr_onnxruntime import RapidOCR\nfrom knotwork.knotcore import contracts\n",
        encoding="utf-8",
    )
    assert ka.tcb_report(kernel, tmp_path / "src").ok


def test_loc_caps_are_the_specified_values() -> None:
    assert (ka.EXECUTABLE_LOC_CAP, ka.TOTAL_LOC_CAP) == (850, 1350)
    assert frozenset({"contracts.py"}) == ka.DECLARATIVE_MODULES


def test_knotcore_has_no_forbidden_imports() -> None:
    violations = ka.scan_forbidden_imports()
    assert not violations, "\n".join(
        f"{v.file}:{v.line} {v.module} ({v.reason})" for v in violations
    )


def test_scanner_catches_planted_violations(tmp_path: Path) -> None:
    src_root = tmp_path / "src"
    kernel = src_root / "knotwork" / "knotcore"
    kernel.mkdir(parents=True)
    (src_root / "knotwork" / "__init__.py").write_text("")
    (kernel / "__init__.py").write_text("")
    (kernel / "bad.py").write_text(
        textwrap.dedent(
            """
            import numpy as np
            import knotwork.plugins.tangle
            from ..plugins import gauge
            from knotwork.bench import run
            from . import contracts
            import shapely.geometry
            import importlib
            mod = importlib.import_module("knotwork.plugins")
            other = __import__("httpx")
            """
        )
    )
    found = {
        (v.line, v.reason.split(": ")[-1]) for v in ka.scan_forbidden_imports(kernel, src_root)
    }
    assert (3, "knotwork.plugins") in found  # absolute
    assert (4, "knotwork.plugins") in found  # relative ..plugins
    assert (5, "knotwork.bench") in found
    assert (7, "shapely") in found
    assert any(line == 9 for line, _ in found)  # importlib.import_module
    assert any(line == 10 for line, _ in found)  # __import__
    assert not any(line in (2, 6) for line, _ in found)  # numpy and sibling kernel import OK


def test_knotcore_has_no_placeholders() -> None:
    hits = ka.scan_placeholders(ka.KNOTCORE_DIR)
    assert not hits, "\n".join(f"{f}:{n}: {t}" for f, n, t in hits)


def test_src_tree_has_no_placeholders(repo_root: Path) -> None:
    hits = ka.scan_placeholders(repo_root / "src")
    assert hits == [], "\n".join(f"{f}:{n}: {t}" for f, n, t in hits)


def test_placeholder_scanner_detects_markers(tmp_path: Path) -> None:
    marker = "TO" + "DO"
    (tmp_path / "a.py").write_text(f"x = 1  # {marker} later\n")
    (tmp_path / "b.py").write_text("raise " + "NotImpl" + "ementedError\n")
    (tmp_path / "c.py").write_text("y = 2\n")
    assert [Path(f).name for f, _, _ in ka.scan_placeholders(tmp_path)] == ["a.py", "b.py"]


def test_importlinter_contract_declared(repo_root: Path) -> None:
    cfg = configparser.ConfigParser()
    cfg.read(repo_root / ".importlinter")
    contract = cfg["importlinter:contract:knotcore-isolation"]
    assert contract["type"] == "forbidden"
    assert contract["source_modules"].split() == ["knotwork.knotcore"]
    forbidden = contract["forbidden_modules"].split()
    assert "knotwork.plugins" in forbidden


def test_kernel_sha256_is_stable() -> None:
    assert ka.kernel_sha256() == ka.kernel_sha256()
    assert len(ka.kernel_sha256()) == 64


@pytest.mark.parametrize("name", ["contracts.py", "seeds.py"])
def test_core_kernel_modules_present(name: str) -> None:
    assert (ka.KNOTCORE_DIR / name).is_file()
