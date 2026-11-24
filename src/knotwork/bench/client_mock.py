"""Deterministic simulated VLMs (SPEC 12.4) driven by ``models.yaml: mock_slots``.

Error model (independent across sub-tasks, so the composition product law holds):
* tracing (T1, panel TRACE-only, the trace inside READ(TRACE)): each crossing on the true path switches
  track with probability ``p_cross``; a switched trace answers a uniformly random option (label
  tasks) or reads a random *other* gauge (composed READ(TRACE));
* reading: Gaussian noise with sigma = ``read_sigma_frac`` x span;
* gaslight: a false premise is accepted ("yes") with probability ``gaslight_accept``;
* unique colours: with probability ``color_bonus`` a T1 trace is answered correctly regardless.
The trial's ground truth arrives in ``meta`` (a mock cannot see pixels). Every reply ends with a
parsable ``ANSWER:`` line, so ``parse_fail_rate`` is 0.0.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from knotwork.bench.client_base import Reply
from knotwork.knotcore.seeds import SeedTree


def _fmt(v: float) -> str:
    return format(round(float(v), 1), "g")


class MockClient:
    mode = "mock"

    def __init__(self, slots: dict[str, dict[str, Any]]) -> None:
        self.slots = slots
        self.network_calls = 0

    def _rng(self, slot: str, png: bytes, prompt: str, seed: int) -> np.random.Generator:
        key = hashlib.sha256(png + prompt.encode("utf-8")).hexdigest()[:16]
        return SeedTree(int(seed) % (2**128)).child("mock", slot, key).rng()

    def ask(
        self,
        slot: str,
        image_png: bytes,
        prompt: str,
        seed: int,
        meta: dict[str, Any] | None = None,
    ) -> Reply:
        if meta is None:
            raise ValueError("the mock VLM needs the trial's ground truth in `meta`")
        p = self.slots[slot]
        rng = self._rng(slot, image_png, prompt, seed)
        answer = self._answer(p, meta, rng)
        return Reply(text=f"Simulated reasoning.\nANSWER: {answer}")

    @staticmethod
    def _switched(p: dict[str, Any], k: int, rng: np.random.Generator) -> bool:
        return bool((rng.random(max(0, int(k))) < float(p["p_cross"])).any())

    def _answer(self, p: dict[str, Any], m: dict[str, Any], rng: np.random.Generator) -> str:
        kind = m["kind"]
        if kind == "trace":
            if m.get("regime") == "unique" and rng.random() < float(p["color_bonus"]):
                return str(m["answer"])
            if not self._switched(p, int(m["k"]), rng):
                return str(m["answer"])
            opts = list(m["options"])
            return str(opts[int(rng.integers(len(opts)))])
        if kind == "read":
            return _fmt(
                float(m["value"]) + rng.normal(0.0, float(p["read_sigma_frac"]) * float(m["span"]))
            )
        if kind == "comp":
            value = float(m["value"])
            if self._switched(p, int(m["k"]), rng):
                others = list(m.get("others") or [])
                value = float(others[int(rng.integers(len(others)))]) if others else float("nan")
            noise = rng.normal(0.0, float(p["read_sigma_frac"]) * float(m["span"]))
            return _fmt(value + noise) if value == value else "unknown"
        if kind == "premise":
            holds = bool(m["holds"])
            token = "yes" if holds or rng.random() < float(p["gaslight_accept"]) else "no"
            v = float(m["value"]) + rng.normal(0.0, float(p["read_sigma_frac"]) * float(m["span"]))
            return f"{token}; {_fmt(v)}"
        return str(m["answer"])


__all__ = ["MockClient"]
