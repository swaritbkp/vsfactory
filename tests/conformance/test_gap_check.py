"""scripts/gap_check.py: locks, request body and report, using httpx.MockTransport (no sockets)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
from PIL import Image


def _load(repo_root: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "gap_check", repo_root / "scripts" / "gap_check.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dataset(root: Path) -> Path:
    data = root / "phase2"
    for plugin, label in (("tangle", "7"), ("gauge", "40")):
        ds = data / f"data_{plugin}"
        rows = []
        for i in range(2):
            bid = f"kw-{plugin[:1]}{i:019d}"
            d = ds / "bundles" / bid
            d.mkdir(parents=True)
            Image.new("RGB", (1024, 1024), (246, 244, 238)).save(d / "beauty.png")
            atype = "label" if plugin == "tangle" else "float"
            task = {
                "bundle_id": bid,
                "scene_id": "0" * 64,
                "family": "T1_TRACE" if plugin == "tangle" else "G1_READ",
                "question": {"text": "Follow the cable from terminal A. Where does it end?"},
                "answer": {"type": atype, "value": label, "tolerance": 1.6, "canonical": label},
            }
            (d / "task.json").write_text(json.dumps(task))
            rows.append({"bundle_id": bid, "role": "base", "dir": f"bundles/{bid}"})
        (ds / "index.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return data


@pytest.fixture
def env(tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.chdir(tmp_path)
    return {"mod": _load(repo_root), "data": _dataset(tmp_path), "tmp": tmp_path}


def _args(env: dict[str, Any], *extra: str) -> list[str]:
    return ["--data", str(env["data"]), "--n", "4", "--out", str(env["tmp"] / "gap.json"), *extra]


def test_budget_cap_refused(env: dict[str, Any]) -> None:
    with pytest.raises(SystemExit) as exc:
        env["mod"].main(_args(env, "--budget-usd", "0.51"))
    assert exc.value.code == 4


def test_requires_approval_file_and_key(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit) as exc:
        env["mod"].main(_args(env, "--budget-usd", "0.5"))
    assert exc.value.code == 4  # no approval file
    (env["tmp"] / ".checkpoints").mkdir()
    (env["tmp"] / ".checkpoints" / "GAPCHECK_APPROVED").write_text("ok")
    with pytest.raises(SystemExit) as exc2:
        env["mod"].main(_args(env, "--budget-usd", "0.5"))
    assert exc2.value.code == 4  # no API key


def test_dry_run_makes_no_request(env: dict[str, Any]) -> None:
    def _fail(request: httpx.Request) -> httpx.Response:
        raise AssertionError("dry run must not send requests")

    rc = env["mod"].main(
        _args(env, "--budget-usd", "0.5", "--dry-run"), transport=httpx.MockTransport(_fail)
    )
    assert rc == 0


def test_live_path_with_mock_transport(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    (env["tmp"] / ".checkpoints").mkdir()
    (env["tmp"] / ".checkpoints" / "GAPCHECK_APPROVED").write_text("ok")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append({"body": body, "auth": request.headers["Authorization"]})
        answer = "ANSWER: 7" if len(seen) % 2 else "I think it is forty.\nANSWER: 40"
        return httpx.Response(
            200, json={"choices": [{"message": {"content": answer}}], "usage": {"cost": 0.001}}
        )

    rc = env["mod"].main(
        _args(env, "--budget-usd", "0.5", "--slot", "vlm_b"), transport=httpx.MockTransport(handler)
    )
    assert rc == 0
    assert len(seen) == 4
    first = seen[0]["body"]
    assert first["provider"]["data_collection"] == "deny"
    assert first["usage"]["include"] is True
    assert first["max_tokens"] == 512
    assert first["reasoning"] == {"effort": "low"}
    assert first["messages"][1]["content"][0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert seen[0]["auth"] == "Bearer sk-or-test"
    rep = json.loads((env["tmp"] / "gap.json").read_text())
    assert rep["n_done"] == 4 and rep["accuracy"] == 1.0 and rep["parse_fail_rate"] == 0.0
    assert rep["cost_usd"] == pytest.approx(0.004)
    ledger = (env["tmp"] / "freshness" / "burn_ledger.jsonl").read_text().splitlines()
    assert len(ledger) == 4 and json.loads(ledger[0])["model"] == "google/gemini-2.5-flash"


def test_budget_stop(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    (env["tmp"] / ".checkpoints").mkdir()
    (env["tmp"] / ".checkpoints" / "GAPCHECK_APPROVED").write_text("ok")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ANSWER: 7"}}], "usage": {"cost": 0.2}}
        )

    env["mod"].main(_args(env, "--budget-usd", "0.3"), transport=httpx.MockTransport(handler))
    # first call costs 0.2; the next is assumed to cost >= 0.2, so 0.4 > 0.3 stops before overspending
    assert len(calls) == 1
    rep = json.loads((env["tmp"] / "gap.json").read_text())
    assert rep["cost_usd"] <= 0.3
