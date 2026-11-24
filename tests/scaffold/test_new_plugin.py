"""Task 4.4: `knot new-plugin` scaffold compiles, satisfies TaskPlugin, passes its own tests, and
leaves knotcore untouched."""

from __future__ import annotations

import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from knotwork import cli as K
from knotwork.devtools.kernel_audit import kernel_sha256
from knotwork.scaffold.new_plugin import scaffold

REPO = Path(__file__).resolve().parents[2]


def test_scaffold_passes_its_tests(tmp_path: Path) -> None:
    before = kernel_sha256()
    res = CliRunner().invoke(K.cli, ["new-plugin", "demo_dots", "--dest", str(tmp_path)])
    assert res.exit_code == 0, res.output
    pkg = tmp_path / "demo_dots"
    for rel in ("__init__.py", "plugin.py", "README.md", "tests/test_plugin.py"):
        assert (pkg / rel).is_file()
    for p in pkg.rglob("*.py"):
        py_compile.compile(str(p), doraise=True)
    assert "{{name}}" not in (pkg / "plugin.py").read_text(encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(REPO / "src")])}
    cp = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(tmp_path),
         str(pkg / "tests")],
        env=env, cwd=REPO, capture_output=True, text=True,
    )  # fmt: skip
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "3 passed" in cp.stdout
    assert kernel_sha256() == before


def test_bad_names_and_existing_dest(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        scaffold("Bad-Name", tmp_path)
    scaffold("dup_plugin", tmp_path)
    with pytest.raises(FileExistsError):
        scaffold("dup_plugin", tmp_path)
    res = CliRunner().invoke(K.cli, ["new-plugin", "tangle", "--dest", str(tmp_path)])
    assert res.exit_code == 2
