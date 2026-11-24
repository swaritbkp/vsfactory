"""Task 5.2: prompts, client interface and the deterministic mock VLMs (SPEC 12.1, 12.4)."""

from __future__ import annotations

import base64
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from knotwork.bench.client_base import VLMClient
from knotwork.bench.client_mock import MockClient
from knotwork.bench.prompts import (
    FORMAT,
    SYSTEM,
    messages,
    read_only,
    request_body,
    trace_only,
    with_format,
)
from knotwork.knotcore.grading import extract

PNG = b"\x89PNG fake image bytes"
P = {
    "slot": "m",
    "p_cross": 0.1,
    "read_sigma_frac": 0.02,
    "gaslight_accept": 0.3,
    "color_bonus": 0.6,
}


def _ask(client: MockClient, meta: dict, prompt: str, seed: int = 0) -> str:
    return extract(client.ask("m", PNG, prompt, seed, meta).text).strip()


def test_prompts() -> None:
    assert SYSTEM.endswith("'ANSWER: <answer>'.")
    t = with_format("How many?", "int")
    assert t.endswith("'ANSWER: <integer>'.") and with_format(t, "int") == t
    assert "G1-G3" in trace_only("K", 3) and FORMAT["label"] in trace_only("K", 3)
    assert "gauge G2" in read_only("G2") and FORMAT["float"] in read_only("G2")
    msgs = messages(PNG, "Q?")
    assert msgs[0] == {"role": "system", "content": SYSTEM}
    url = msgs[1]["content"][0]["image_url"]["url"]
    assert base64.b64decode(url.split(",", 1)[1]) == PNG and msgs[1]["content"][1]["text"] == "Q?"
    cfg = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))
    body = request_body(cfg["slots"][0], cfg["request_defaults"], cfg["routing"], PNG, "Q?")
    assert body["provider"]["data_collection"] == "deny" and body["usage"] == {"include": True}
    assert body["temperature"] == 0.0 and body["max_tokens"] == cfg["slots"][0]["max_tokens"]
    assert body["reasoning"] == {"effort": "low"}
    assert "reasoning" not in request_body(
        cfg["slots"][2], cfg["request_defaults"], cfg["routing"], PNG, "Q"
    )


def test_deterministic_per_seed() -> None:
    c = MockClient({"m": P})
    assert isinstance(c, VLMClient) and c.mode == "mock" and c.network_calls == 0
    meta = {"kind": "trace", "answer": "3", "options": [str(i) for i in range(1, 7)], "k": 8}
    a = [_ask(c, meta, f"q{i}") for i in range(50)]
    assert a == [_ask(c, meta, f"q{i}") for i in range(50)]
    assert a != [_ask(c, meta, f"q{i}", seed=1) for i in range(50)]
    with pytest.raises(ValueError):
        c.ask("m", PNG, "q", 0, None)


def _rate(c: MockClient, meta: dict, n: int, ok: object) -> float:
    return float(np.mean([_ask(c, meta, f"q{i}") == ok for i in range(n)]))


def test_trace_error_model() -> None:
    c = MockClient({"m": P})
    opts = [str(i) for i in range(1, 6)]
    for k in (0, 3, 8):
        meta = {"kind": "trace", "answer": "2", "options": opts, "k": k}
        want = (1 - 0.1) ** k + (1 - (1 - 0.1) ** k) / 5
        assert _rate(c, meta, 3000, "2") == pytest.approx(want, abs=0.03)
    uniq = {"kind": "trace", "answer": "2", "options": opts, "k": 8, "regime": "unique"}
    s = 0.9**8
    assert _rate(c, uniq, 3000, "2") == pytest.approx(0.6 + 0.4 * (s + (1 - s) / 5), abs=0.03)


def test_read_noise_and_gaslight() -> None:
    c = MockClient({"m": P})
    vals = np.array(
        [
            float(_ask(c, {"kind": "read", "value": 40.0, "span": 80.0}, f"q{i}"))
            for i in range(3000)
        ]
    )
    assert vals.mean() == pytest.approx(40.0, abs=0.1)
    assert vals.std() == pytest.approx(0.02 * 80.0, rel=0.06)
    gas = {"kind": "premise", "holds": False, "value": 30.0, "span": 80.0}
    yes = np.mean([_ask(c, gas, f"q{i}").startswith("yes") for i in range(3000)])
    assert yes == pytest.approx(0.3, abs=0.03)
    ctrl = {"kind": "premise", "holds": True, "value": 30.0, "span": 80.0}
    assert all(_ask(c, ctrl, f"q{i}").startswith("yes;") for i in range(50))


def test_composition_independence() -> None:
    """Composed READ(TRACE) is correct iff the trace survived and the read is within tolerance."""
    c = MockClient({"m": P})
    meta = {"kind": "comp", "value": 40.0, "others": [10.0, 75.0], "k": 4, "span": 80.0}
    vals = np.array([float(_ask(c, meta, f"q{i}")) for i in range(4000)])
    ok = np.abs(vals - 40.0) <= 1.6
    p_read = math.erf(1.6 / (0.02 * 80.0 * math.sqrt(2)))
    assert ok.mean() == pytest.approx(0.9**4 * p_read, abs=0.03)
