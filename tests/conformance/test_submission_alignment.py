"""The Phase-6 docs gate, the submission checklist and /ship-pack agree on the official slide order."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _gate(repo_root: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "verify_phase", repo_root / "scripts" / "verify_phase.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verify_phase"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_checklist_section_c_matches_gate_slide_order(repo_root: Path) -> None:
    vp = _gate(repo_root)
    text = (repo_root / "docs" / "SUBMISSION_CHECKLIST.md").read_text(encoding="utf-8")
    section = text[text.index("## C.") : text.index("## D.")]
    rows = [
        line
        for line in section.splitlines()
        if line.startswith("| ") and line[2:4].strip().isdigit()
    ]
    assert len(rows) == 11
    for i, (row, title) in enumerate(zip(rows, vp.OFFICIAL_SLIDES, strict=True), start=1):
        assert row.startswith(f"| {i} |") and title in row, (i, title, row)


def test_template_phrases_on_the_right_slides(repo_root: Path) -> None:
    vp = _gate(repo_root)
    text = (repo_root / "docs" / "SUBMISSION_CHECKLIST.md").read_text(encoding="utf-8")
    section = text[text.index("## C.") : text.index("## D.")]
    rows = {
        int(r.split("|")[1]): r
        for r in section.splitlines()
        if r.startswith("| ") and r[2:4].strip().isdigit()
    }
    for num, phrase in vp.TEMPLATE_PHRASES.items():
        assert phrase in rows[num], (num, phrase)


def test_ship_pack_uses_official_titles(repo_root: Path) -> None:
    vp = _gate(repo_root)
    text = (repo_root / ".claude" / "commands" / "ship-pack.md").read_text(encoding="utf-8")
    for title in vp.OFFICIAL_SLIDES:
        assert title in text, title
