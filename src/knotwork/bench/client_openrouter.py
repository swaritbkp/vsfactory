"""OpenRouter client (SPEC 12.4) -- the only paid path; triple-locked by `knot bench --live`.

Async ``httpx`` through `knotwork.net.apost_json` (policy ``openrouter``), concurrency from
``request_defaults.concurrency``, exponential backoff with ``max_retries`` on 429/5xx. Every response
is written to the replay cache *before* it is returned, and the budget is checked before every
call. Tests inject ``httpx.MockTransport``; nothing here runs in tests or gates against the network.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from knotwork import net
from knotwork.bench.budget import Budget, actual_cost, est_call_usd
from knotwork.bench.client_base import Reply
from knotwork.bench.client_replay import elide_images, replay_key, write_entry
from knotwork.bench.prompts import image_sha256, request_body

RETRY_STATUS = {429, 500, 502, 503, 504}


class OpenRouterError(RuntimeError):
    pass


class OpenRouterClient:
    mode = "live"

    def __init__(self, cfg: dict[str, Any], replay_dir: Path, api_key: str, budget: Budget,
                 transport: Any = None, sleep: Callable[[float], Awaitable[None]] | None = None,
                 on_call: Callable[[str, dict[str, Any], str], None] | None = None) -> None:  # fmt: skip
        if not api_key:
            raise OpenRouterError("E_API_KEY_MISSING")
        self.cfg, self.replay_dir, self.budget = cfg, Path(replay_dir), budget
        self.slots = {s["slot"]: s for s in cfg["slots"]}
        self.defaults, self.routing = cfg["request_defaults"], cfg["routing"]
        self._key, self._transport = api_key, transport
        self._sleep = sleep or asyncio.sleep
        self._on_call = on_call
        self.network_calls = 0
        self.margin = float(cfg["budget"]["per_call_safety_margin"])

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json",
                "X-Title": "knotwork-bench"}  # fmt: skip

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        retries = int(self.defaults.get("max_retries", 3))
        for attempt in range(retries + 1):
            self.network_calls += 1
            status, data = await net.apost_json(
                self.cfg["endpoint"], body, self._headers(), float(self.defaults.get("timeout_s", 90)),
                self._transport,
            )  # fmt: skip
            if status == 200 and isinstance(data, dict):
                return data
            if status in RETRY_STATUS and attempt < retries:
                await self._sleep(2.0**attempt)
                continue
            raise OpenRouterError(f"HTTP {status}: {str(data)[:300]}")
        raise OpenRouterError("unreachable")

    async def aask(self, slot: str, image_png: bytes, prompt: str, seed: int) -> Reply:
        s = self.slots[slot]
        self.budget.check(est_call_usd(s, self.cfg["plan"], self.margin))
        body = request_body(s, self.defaults, self.routing, image_png, prompt)
        body["seed"] = int(seed) % (2**31)
        data = await self._post(body)
        choice = (data.get("choices") or [{}])[0]
        text = str((choice.get("message") or {}).get("content") or "")
        usage = data.get("usage") or {}
        cost = actual_cost(s, usage)
        sha = image_sha256(image_png)
        key = replay_key(slot, s["slug"], prompt, sha, seed)
        write_entry(self.replay_dir, slot, key, {"request": elide_images(body, sha), "response": data,
                                                 "text": text, "usage": usage, "cost_usd": cost})  # fmt: skip
        self.budget.add(cost)
        if self._on_call is not None:
            self._on_call(slot, body, sha)
        return Reply(text=text, usage_in=int(usage.get("prompt_tokens", 0)),
                     usage_out=int(usage.get("completion_tokens", 0)), cost_usd=cost)  # fmt: skip

    def ask(
        self,
        slot: str,
        image_png: bytes,
        prompt: str,
        seed: int,
        meta: dict[str, Any] | None = None,
    ) -> Reply:
        return asyncio.run(self.aask(slot, image_png, prompt, seed))

    def ask_many(self, calls: list[tuple[str, bytes, str, int]]) -> list[Reply | BaseException]:
        """Run independent calls with bounded concurrency; failures are returned, not raised."""
        limit = asyncio.Semaphore(int(self.defaults.get("concurrency", 4)))

        async def one(c: tuple[str, bytes, str, int]) -> Reply:
            async with limit:
                return await self.aask(*c)

        async def run() -> list[Reply | BaseException]:
            return list(await asyncio.gather(*(one(c) for c in calls), return_exceptions=True))

        return asyncio.run(run())


__all__ = ["OpenRouterClient", "OpenRouterError"]
