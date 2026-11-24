"""Budget guard and preflight estimate (SPEC 12.5).

Preflight: calls x est_tokens x est_price x per_call_safety_margin per slot. Live: before each call,
``spent + estimate(call) > budget_usd - stop_when_remaining_usd_below`` stops the run cleanly.
Actual cost = OpenRouter ``usage.cost`` when present, else tokens x price.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from knotwork.bench.client_base import BudgetExhausted


def est_call_usd(slot: dict[str, Any], plan: dict[str, Any], margin: float) -> float:
    tin, tout = float(plan["est_tokens_in_per_call"]), float(plan["est_tokens_out_per_call"])
    usd = (
        tin * float(slot["est_usd_per_mtok_in"]) + tout * float(slot["est_usd_per_mtok_out"])
    ) / 1e6
    return usd * margin


def preflight(
    cfg: dict[str, Any], calls_per_slot: dict[str, int], budget_usd: float | None
) -> dict[str, Any]:
    margin = float(cfg["budget"]["per_call_safety_margin"])
    per_slot = {
        s["slot"]: round(calls_per_slot.get(s["slot"], 0) * est_call_usd(s, cfg["plan"], margin), 4)
        for s in cfg["slots"]
    }
    return {
        "estimated_usd": round(sum(per_slot.values()), 4),
        "calls": int(sum(calls_per_slot.values())),
        "calls_per_slot": dict(calls_per_slot),
        "per_slot": per_slot,
        "slugs": {s["slot"]: s["slug"] for s in cfg["slots"]},
        "hard_cap_usd": float(cfg["budget"]["hard_cap_usd"]),
        "budget_usd": budget_usd,
        "per_call_safety_margin": margin,
    }


def actual_cost(slot: dict[str, Any], usage: dict[str, Any] | None) -> float:
    u = usage or {}
    if u.get("cost") is not None:
        return float(u["cost"])
    tin, tout = float(u.get("prompt_tokens", 0)), float(u.get("completion_tokens", 0))
    return (
        tin * float(slot["est_usd_per_mtok_in"]) + tout * float(slot["est_usd_per_mtok_out"])
    ) / 1e6


@dataclass
class Budget:
    budget_usd: float
    stop_below: float
    spent: float = 0.0
    calls: int = 0
    log: list[float] = field(default_factory=list)

    def check(self, estimate: float) -> None:
        """Raise BudgetExhausted when the next call could cross the stop line."""
        if self.spent + estimate > self.budget_usd - self.stop_below:
            raise BudgetExhausted(
                f"spent {self.spent:.4f} + next {estimate:.4f} > budget {self.budget_usd:.2f} - "
                f"{self.stop_below:.2f}"
            )

    def add(self, cost: float) -> None:
        self.spent += float(cost)
        self.calls += 1
        self.log.append(float(cost))


__all__ = ["Budget", "actual_cost", "est_call_usd", "preflight"]
