"""docs/schemas/*.schema.json must equal the live Pydantic models (no silent contract drift)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_published_schemas_match_models(repo_root: Path) -> None:
    cp = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "export_schemas.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0, cp.stdout + cp.stderr
