"""Task 5.3: OpenRouter + replay clients with httpx.MockTransport only (zero real sockets)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from knotwork import net
from knotwork.bench.budget import Budget
from knotwork.bench.client_base import BudgetExhausted, ReplayMiss
from knotwork.bench.client_openrouter import OpenRouterClient, OpenRouterError
from knotwork.bench.client_replay import ReplayClient, replay_key, replay_path
from knotwork.bench.prompts import image_sha256

REPO = Path(__file__).resolve().parents[2]
PNG = b"\x89PNG not really"
KEY = "sk-or-test-not-a-real-key"


def cfg() -> dict[str, Any]:
    return yaml.safe_load((REPO / "configs" / "models.yaml").read_text(encoding="utf-8"))


def ok_response(text: str = "thinking\nANSWER: 7", cost: float | None = 0.0021) -> dict[str, Any]:
    usage: dict[str, Any] = {"prompt_tokens": 1200, "completion_tokens": 80}
    if cost is not None:
        usage["cost"] = cost
    return {
        "id": "gen-1",
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": usage,
    }


class Recorder:
    def __init__(self, statuses: list[int], replay_dir: Path | None = None) -> None:
        self.statuses, self.requests, self.replay_dir = list(statuses), [], replay_dir
        self.cached_before_return: list[bool] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        status = self.statuses.pop(0) if self.statuses else 200
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json=ok_response())


async def _no_sleep(_: float) -> None:
    return None


def make(tmp: Path, rec: Recorder, budget: float = 5.0) -> OpenRouterClient:
    return OpenRouterClient(cfg(), tmp / "replay", KEY, Budget(budget, 0.10),
                            transport=httpx.MockTransport(rec), sleep=_no_sleep)  # fmt: skip


def test_body_headers_and_replay_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOTWORK_NETWORK", "openrouter")
    rec = Recorder([200])
    c = make(tmp_path, rec)
    reply = c.ask("vlm_b", PNG, "Which?", 3)
    assert reply.text.endswith("ANSWER: 7") and reply.cost_usd == pytest.approx(0.0021)
    assert c.network_calls == 1 and len(rec.requests) == 1
    req = rec.requests[0]
    assert str(req.url) == cfg()["endpoint"]
    assert req.headers["Authorization"] == f"Bearer {KEY}"
    body = json.loads(req.content)
    assert body["provider"]["data_collection"] == "deny" and body["usage"] == {"include": True}
    assert body["model"] == "google/gemini-2.5-flash" and body["temperature"] == 0.0
    key = replay_key("vlm_b", "google/gemini-2.5-flash", "Which?", image_sha256(PNG), 3)
    entry = json.loads(replay_path(tmp_path / "replay", "vlm_b", key).read_text())
    assert entry["text"] == reply.text and entry["cost_usd"] == pytest.approx(0.0021)
    blob = json.dumps(entry)
    assert KEY not in blob and "base64" not in blob and f"sha256:{image_sha256(PNG)}" in blob
    # the cached response replays offline, with zero network calls
    rc = ReplayClient(tmp_path / "replay", {s["slot"]: s["slug"] for s in cfg()["slots"]})
    again = rc.ask("vlm_b", PNG, "Which?", 3)
    assert again.text == reply.text and again.cached and rc.network_calls == 0


def test_429_retry_then_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOTWORK_NETWORK", "openrouter")
    rec = Recorder([429, 503, 200])
    c = make(tmp_path, rec)
    assert c.ask("vlm_a", PNG, "Q", 0).text.endswith("ANSWER: 7")
    assert c.network_calls == 3 and len(rec.requests) == 3


def test_retries_exhausted_and_client_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KNOTWORK_NETWORK", "openrouter")
    c = make(tmp_path, Recorder([429, 429, 429, 429]))
    with pytest.raises(OpenRouterError):
        c.ask("vlm_a", PNG, "Q", 0)
    assert c.network_calls == 4  # 1 + max_retries
    c2 = make(tmp_path, Recorder([401]))
    with pytest.raises(OpenRouterError):
        c2.ask("vlm_a", PNG, "Q", 0)
    assert c2.network_calls == 1


def test_policy_blocks_without_openrouter_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KNOTWORK_NETWORK", "deny")
    rec = Recorder([200])
    with pytest.raises(net.NetworkPolicyError):
        make(tmp_path, rec).ask("vlm_a", PNG, "Q", 0)
    assert rec.requests == []


def test_budget_stops_before_exceeding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOTWORK_NETWORK", "openrouter")
    rec = Recorder([])
    c = make(
        tmp_path, rec, budget=0.12
    )  # stop line 0.02; each call costs 0.0021 (est ~0.0097 vlm_a)
    n = 0
    with pytest.raises(BudgetExhausted):
        for _ in range(100):
            c.ask("vlm_a", PNG, f"Q{n}", 0)
            n += 1
    assert c.budget.spent <= 0.12 - 0.10 and n == len(rec.requests)


def test_missing_key_and_replay_miss(tmp_path: Path) -> None:
    with pytest.raises(OpenRouterError):
        OpenRouterClient(cfg(), tmp_path, "", Budget(1.0, 0.1))
    rc = ReplayClient(tmp_path / "empty", {"vlm_a": "x/y"})
    with pytest.raises(ReplayMiss):
        rc.ask("vlm_a", PNG, "never asked", 0)


def test_ask_many_concurrent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOTWORK_NETWORK", "openrouter")
    rec = Recorder([200] * 6)
    c = make(tmp_path, rec)
    res = c.ask_many([("vlm_c", PNG, f"Q{i}", 0) for i in range(6)])
    assert len(res) == 6 and all(not isinstance(r, BaseException) for r in res)
    assert len(rec.requests) == 6 == c.network_calls
    assert len(list((tmp_path / "replay" / "vlm_c").glob("*.json"))) == 6


def test_cost_falls_back_to_tokens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOTWORK_NETWORK", "openrouter")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_response(cost=None))

    c = OpenRouterClient(
        cfg(), tmp_path, KEY, Budget(5.0, 0.1), transport=httpx.MockTransport(handler)
    )
    r = c.ask("vlm_b", PNG, "Q", 0)
    assert r.cost_usd == pytest.approx((1200 * 0.30 + 80 * 2.50) / 1e6)
