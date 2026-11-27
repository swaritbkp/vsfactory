"""Rebuild the committed Starter Working Set in `samples/starter_working_set/`.

    uv run python scripts/build_starter_set.py

One certified item per built-in domain (tangle, gauge, panel, rack_patch), rendered on the Clean
preset so the PNGs stay small, each certified by the live kernel. The Studio copies this folder on
a cold start instead of rendering, which keeps the first page load fast; `tests/studio/
test_working_set.py` re-certifies every bundle so the committed copy cannot drift from the kernel.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from knotwork.knotcore import engine
from knotwork.render.canvas import REPO_ROOT
from knotwork.studio import factory, working_set

#: (domain, question family, seed): fast, legible, certified on the first attempt
ITEMS = (
    ("tangle", "T1_TRACE", 11),
    ("gauge", "G1_READ", 11),
    ("panel", "P1_READ_TRACE", 11),
    ("rack_patch", "T1_TRACE", 11),
)
CLEAN = {"reasoning_load": 0.2, "visual_stress": 0.0, "occlusion_level": 0.0, "realism": 0.0,
         "color_regime": "unique"}  # fmt: skip


def build(out: Path = working_set.STARTER) -> Path:
    stage = Path(tempfile.mkdtemp(prefix="knotwork-starter-"))
    rows, attempts = [], 0
    for plugin, family, seed in ITEMS:
        made = factory.generate_one(plugin, family, seed, stage / plugin,
                                    factory.difficulty_from(CLEAN), certify_attempts=3)  # fmt: skip
        if not made.candidate.certified or made.task is None:
            raise SystemExit(f"{plugin} seed {seed} did not certify: {made.candidate.certificate}")
        attempts += made.attempts
        bid = made.task["bundle_id"]
        shutil.copytree(made.candidate.dir, stage / "set" / working_set.BUNDLES / bid)
        row = working_set.index_row(made.task, "starter", seed)
        row["added_utc"] = ""
        rows.append(row)
    meta = {"label": "Starter Working Set", "candidates": attempts, "certified": len(rows),
            "kernel_sha256": engine.kernel_sha256()}  # fmt: skip
    target = stage / "set"
    (target / working_set.INDEX).write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), "utf-8", newline="\n"
    )
    (target / working_set.META).write_text(json.dumps(meta, indent=1) + "\n", "utf-8", newline="\n")
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(target, out)
    shutil.rmtree(stage, ignore_errors=True)
    return out


if __name__ == "__main__":
    path = build()
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    print(f"starter working set: {path.relative_to(REPO_ROOT).as_posix()} ({size / 1e6:.2f} MB)")
    sys.exit(0)
