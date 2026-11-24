"""VLM client interface (SPEC 12.4): ``ask(slot, image_png, prompt, seed) -> Reply``.

``meta`` carries the trial's ground truth; only the mock client reads it (real clients ignore it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Reply:
    text: str
    usage_in: int = 0
    usage_out: int = 0
    cost_usd: float = 0.0
    cached: bool = False


class ReplayMiss(KeyError):
    """No cached response for this request (replay never falls back to the network)."""


class BudgetExhausted(RuntimeError):
    """The next call would exceed the budget (the run stops cleanly, SPEC 12.5)."""


@runtime_checkable
class VLMClient(Protocol):
    mode: str
    network_calls: int

    def ask(
        self,
        slot: str,
        image_png: bytes,
        prompt: str,
        seed: int,
        meta: dict[str, Any] | None = None,
    ) -> Reply: ...


__all__ = ["BudgetExhausted", "Reply", "ReplayMiss", "VLMClient"]
