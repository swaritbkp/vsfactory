"""Harbor task exporter (SPEC 13.1): one self-contained task folder per certified bundle.

```
OUT/harbor/<bundle_id>/
  instruction.md  task.toml  environment/{Dockerfile,image.png}  solution/{answer.json,solve.sh}
  tests/{test.sh,verify.py,expected.json}
```
`tests/verify.py` is the verbatim source of `knotcore/grading.py` plus a `main()`; it is stdlib-only
and writes `reward.txt`, `reward.json` (numeric only, D-3) and `verifier_meta.json` to
`${KNOT_LOG_DIR:-/logs/verifier}`. `KNOT_*_DIR` overrides let the oracle tests run locally.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from knotwork.knotcore import grading
from knotwork.knotcore.contracts import CONSUMPTION_PX, HarborTask

DOCKERFILE = 'FROM python:3.11-slim\nWORKDIR /app\nCOPY image.png /app/image.png\nCMD ["bash"]\n'
SOLVE_SH = """#!/usr/bin/env bash
# Oracle: copy the certified answer into the agent's answer slot.
set -euo pipefail
mkdir -p "${KNOT_APP_DIR:-/app}"
cp "${KNOT_SOLUTION_DIR:-/solution}/answer.json" "${KNOT_APP_DIR:-/app}/answer.json"
"""
TEST_SH = """#!/usr/bin/env bash
# Grade /app/answer.json against tests/expected.json; rewards land in ${KNOT_LOG_DIR:-/logs/verifier}.
TESTS="${KNOT_TESTS_DIR:-/tests}"
PY="$(command -v python3 || command -v python)"
"$PY" "$TESTS/verify.py" || true
exit 0
"""
VERIFY_MAIN = """

# ---------------------------------------------------------------- Harbor verifier entry point
def main() -> int:
    import os
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.environ.get("KNOT_TESTS_DIR", here)
    app_dir = os.environ.get("KNOT_APP_DIR", "/app")
    log_dir = os.environ.get("KNOT_LOG_DIR", "/logs/verifier")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(tests_dir, "expected.json"), encoding="utf-8") as f:
        expected = json.load(f)
    status, reward = "ok", 0.0
    try:
        with open(os.path.join(app_dir, "answer.json"), encoding="utf-8") as f:
            doc = json.load(f)
        if not isinstance(doc, dict) or "answer" not in doc:
            status = "invalid: no 'answer' key"
        else:
            reward = grade_prediction(doc["answer"], expected)
    except FileNotFoundError:
        status = "missing: answer.json not found"
    except (OSError, ValueError) as exc:
        status = "invalid: " + str(exc)[:200]
    with open(os.path.join(log_dir, "reward.txt"), "w", encoding="utf-8") as f:
        f.write(str(float(reward)) + "\\n")
    with open(os.path.join(log_dir, "reward.json"), "w", encoding="utf-8") as f:
        json.dump({"reward": float(reward)}, f)
    meta = {"grader_sha256": GRADER_SHA256, "bundle_id": BUNDLE_ID, "parse_status": status}
    with open(os.path.join(log_dir, "verifier_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, sort_keys=True)
    sys.stdout.write("reward=" + str(float(reward)) + " (" + status + ")\\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
FORMAT_HINT = {
    "label": 'a string label, e.g. {"answer": "7"}',
    "int": 'an integer, e.g. {"answer": 3}',
    "float": 'a number, e.g. {"answer": 47}',
    "zone": 'one of "green", "amber", "red", e.g. {"answer": "red"}',
    "premise": 'an object, e.g. {"answer": {"premise_holds": false, "value": 47}}',
}


def grader_source() -> str:
    return Path(grading.__file__).read_text(encoding="utf-8")


def native_answer(answer: dict[str, Any]) -> Any:
    """Oracle value in the native JSON form an agent writes (SPEC 13.3)."""
    v, t = answer["value"], answer["type"]
    if t == "int":
        return int(v)
    if t == "float":
        f = float(v)
        return int(f) if f.is_integer() else f
    if t == "premise":
        x = float(v["value"])
        return {"premise_holds": bool(v["premise_holds"]), "value": int(x) if x.is_integer() else x}
    return str(v)


def difficulty_score(task: dict[str, Any]) -> float:
    """Kernel difficulty: program depth x 10 + crossings / gauges in the scene (D-30)."""
    q = task["question"]
    d = q.get("difficulty", {})
    size = float(d.get("n", d.get("n_gauges", 1.0)))
    crossings = sum(
        len(s.get("evidence", {}).get("centers", []) or [])
        for s in task["certificate"]["steps"]
        if s["op"] == "CROSS"
    )
    return 10.0 * len(q["steps"]) + size + crossings


def tertiles(scores: list[float]) -> tuple[float, float]:
    s = sorted(scores)
    if not s:
        return 0.0, 0.0
    return s[len(s) // 3], s[(2 * len(s)) // 3]


def difficulty_label(score: float, cuts: tuple[float, float]) -> Literal["easy", "medium", "hard"]:
    return "easy" if score < cuts[0] else ("medium" if score < cuts[1] else "hard")


def instruction(task: dict[str, Any]) -> str:
    q, a = task["question"], task["answer"]
    opts = f"\nOptions: {', '.join(q['options'])}\n" if q.get("options") else ""
    return (
        f"# Knotwork visual reasoning task `{task['bundle_id']}`\n\n"
        f"The image is at `/app/image.png` ({CONSUMPTION_PX} px).\n\n"
        f"## Question\n\n{q['text'].split(chr(10) + 'End your reply')[0].strip()}\n{opts}\n"
        f"## How to answer\n\nWrite a JSON file `/app/answer.json` of the form "
        f'`{{"answer": <value>}}` where `<value>` is {FORMAT_HINT[a["type"]]}.\n'
        f"Answer from the image only.\n\n<!-- {task['canary']} -->\n"
    )


def task_toml(task: dict[str, Any], difficulty: str) -> str:
    fam, plugin, bid = task["family"], task["plugin"], task["bundle_id"]
    return f'''schema_version = "1.4"

[task]
name = "knotwork/{bid}"
description = "Knotwork {fam} visual reasoning task ({plugin})"
keywords = ["knotwork", "visual-reasoning", "{plugin}", "{fam}"]

[metadata]
category = "visual-reasoning"
difficulty = "{difficulty}"
family = "{fam}"
bundle_id = "{bid}"
canary = "{task["canary"]}"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 300.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 2048
'''


def consumption_png(beauty: Path, out: Path) -> None:
    with Image.open(beauty) as im:
        rgb = im.convert("RGB")
        scale = CONSUMPTION_PX / max(rgb.size)
        size = (round(rgb.size[0] * scale), round(rgb.size[1] * scale))
        rgb.resize(size, Image.Resampling.LANCZOS).save(out, optimize=False)


def _write(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    if executable:
        path.chmod(0o755)


def export_task(
    bundle: Path, task: dict[str, Any], out: Path, difficulty: Literal["easy", "medium", "hard"]
) -> HarborTask:
    """Write one Harbor task folder under ``out/<bundle_id>/`` and return its record."""
    bid = task["bundle_id"]
    d = out / bid
    src = grader_source()
    sha = hashlib.sha256(src.encode("utf-8")).hexdigest()
    instr = instruction(task)
    _write(d / "instruction.md", instr)
    _write(d / "task.toml", task_toml(task, difficulty))
    _write(d / "environment" / "Dockerfile", DOCKERFILE)
    consumption_png(bundle / "beauty.png", d / "environment" / "image.png")
    _write(d / "solution" / "answer.json", json.dumps({"answer": native_answer(task["answer"])}))
    _write(d / "solution" / "solve.sh", SOLVE_SH, executable=True)
    _write(d / "tests" / "test.sh", TEST_SH, executable=True)
    header = f'\nGRADER_SHA256 = "{sha}"\nBUNDLE_ID = "{bid}"\n'
    _write(d / "tests" / "verify.py", src + header + VERIFY_MAIN)
    _write(d / "tests" / "expected.json", json.dumps(task["answer"], indent=1, sort_keys=True))
    return HarborTask(
        task_name=f"knotwork/{bid}", bundle_id=bid, instruction_md=instr, expected=task["answer"],
        difficulty=difficulty, tags=["knotwork", task["plugin"], task["family"]], canary=task["canary"],
    )  # fmt: skip


__all__ = ["difficulty_label", "difficulty_score", "export_task", "grader_source", "native_answer"]
