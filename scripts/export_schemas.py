#!/usr/bin/env python3
"""Write JSON Schemas for the public contracts to docs/schemas/ (tests assert they stay in sync).

uv run python scripts/export_schemas.py          # rewrite
uv run python scripts/export_schemas.py --check  # exit 1 if any file is stale
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from knotwork.knotcore import contracts as C  # noqa: E402

MODELS = [
    C.TaskBundle,
    C.Certificate,
    C.QuarantineRecord,
    C.FreshnessReceipt,
    C.HarborTask,
    C.RenderManifest,
    C.ClaimSet,
    C.QuestionProgram,
    C.SceneIR,
]
OUT = REPO / "docs" / "schemas"


def render(model: type) -> str:
    return json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"  # type: ignore[attr-defined]


def main() -> int:
    check = "--check" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    stale = []
    for m in MODELS:
        path = OUT / f"{m.__name__}.schema.json"
        text = render(m)
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(path.name)
        else:
            path.write_text(text, encoding="utf-8")
    if stale:
        print("stale schemas (run scripts/export_schemas.py):", ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
