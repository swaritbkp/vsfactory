"""Task 4.3: Harbor export -- exact layout; oracle 1.0, wrong 0.0, missing 0.0 for every family."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from knotwork.export.harbor import DOCKERFILE, grader_source, native_answer
from knotwork.knotcore.contracts import Family, HarborTask

LAYOUT = (
    "instruction.md", "task.toml", "environment/Dockerfile", "environment/image.png",
    "solution/answer.json", "solution/solve.sh", "tests/test.sh", "tests/verify.py",
    "tests/expected.json",
)  # fmt: skip
BASH = shutil.which("bash")


def wrong(a: dict[str, Any]) -> Any:
    t, v, tol = a["type"], a["value"], a.get("tolerance", 0.0)
    if t == "label":
        return "WRONG-" + str(v)
    if t == "int":
        return int(v) + 1
    if t == "float":
        return float(v) + 10 * tol + 7.0
    if t == "zone":
        return {"green": "red", "amber": "green", "red": "green"}[v]
    return {"premise_holds": not v["premise_holds"], "value": v["value"]}


def tasks(exported: Path) -> list[Path]:
    return sorted(p for p in (exported / "harbor").iterdir() if p.is_dir())


def run_mode(task: Path, mode: str, tmp: Path) -> tuple[float, float, dict[str, Any]]:
    app, logs = tmp / "app", tmp / "logs"
    app.mkdir(parents=True)
    logs.mkdir()
    env = {
        **os.environ,
        "KNOT_APP_DIR": str(app),
        "KNOT_LOG_DIR": str(logs),
        "KNOT_TESTS_DIR": str(task / "tests"),
        "KNOT_SOLUTION_DIR": str(task / "solution"),
    }
    if mode == "oracle":
        if BASH:
            subprocess.run([BASH, str(task / "solution" / "solve.sh")], env=env, check=True)
        else:
            shutil.copy(task / "solution" / "answer.json", app / "answer.json")
    elif mode == "wrong":
        exp = json.loads((task / "tests" / "expected.json").read_text(encoding="utf-8"))
        (app / "answer.json").write_text(json.dumps({"answer": wrong(exp)}), encoding="utf-8")
    elif mode == "garbage":
        (app / "answer.json").write_text("{not json", encoding="utf-8")
    cmd = (
        [BASH, str(task / "tests" / "test.sh")]
        if BASH
        else [sys.executable, str(task / "tests" / "verify.py")]
    )
    cp = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr
    reward = json.loads((logs / "reward.json").read_text(encoding="utf-8"))
    meta = json.loads((logs / "verifier_meta.json").read_text(encoding="utf-8"))
    return float(reward["reward"]), float((logs / "reward.txt").read_text().strip()), meta


def test_layout_and_metadata(exported: Path) -> None:
    ts = tasks(exported)
    assert len(ts) == 3 * 11  # 11 base items (every family) x (base, causal, nuisance)
    src = grader_source()
    for t in ts:
        for rel in LAYOUT:
            assert (t / rel).is_file(), f"{t.name}: {rel}"
        toml = tomllib.loads((t / "task.toml").read_text(encoding="utf-8"))
        assert toml["schema_version"] == "1.4" and toml["task"]["name"] == f"knotwork/{t.name}"
        assert toml["metadata"]["difficulty"] in ("easy", "medium", "hard")
        assert toml["verifier"]["timeout_sec"] == 120.0 and toml["agent"]["timeout_sec"] == 300.0
        assert toml["environment"]["memory_mb"] == 2048
        instr = (t / "instruction.md").read_text(encoding="utf-8")
        assert f"<!-- {toml['metadata']['canary']} -->" in instr and "/app/answer.json" in instr
        assert (t / "environment" / "Dockerfile").read_text() == DOCKERFILE
        assert (t / "tests" / "verify.py").read_text(encoding="utf-8").startswith(src)
        assert max(Image.open(t / "environment" / "image.png").size) == 768
    rows = [json.loads(x) for x in (exported / "harbor" / "index.jsonl").read_text().splitlines()]
    assert sorted(HarborTask.model_validate(r).bundle_id for r in rows) == [t.name for t in ts]


def test_verifier_is_stdlib_only(exported: Path) -> None:
    t = tasks(exported)[0]
    tree = ast.parse((t / "tests" / "verify.py").read_text(encoding="utf-8"))
    mods = {
        a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    mods |= {
        (n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
    }
    assert mods <= {"json", "math", "re", "unicodedata", "typing", "__future__", "os", "sys"}


@pytest.mark.parametrize(
    ("mode", "want"), [("oracle", 1.0), ("wrong", 0.0), ("missing", 0.0), ("garbage", 0.0)]
)
def test_rewards_every_family(exported: Path, tmp_path: Path, mode: str, want: float) -> None:
    seen = set()
    for t in tasks(exported):
        fam = tomllib.loads((t / "task.toml").read_text(encoding="utf-8"))["metadata"]["family"]
        reward, txt, meta = run_mode(t, mode, tmp_path / t.name)
        assert reward == txt == want, (t.name, fam, mode, meta)
        assert meta["bundle_id"] == t.name and len(meta["grader_sha256"]) == 64
        seen.add(fam)
    assert seen == {f.value for f in Family}


def test_native_answers() -> None:
    assert native_answer({"type": "float", "value": 47.0}) == 47
    assert native_answer({"type": "float", "value": 47.5}) == 47.5
    assert native_answer({"type": "int", "value": 3}) == 3
    prem = native_answer({"type": "premise", "value": {"premise_holds": False, "value": 31.0}})
    assert prem == {"premise_holds": False, "value": 31}
    assert native_answer({"type": "label", "value": "G2"}) == "G2"
