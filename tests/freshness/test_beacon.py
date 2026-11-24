"""Network gateway policy + drand/NIST beacon parsing with httpx.MockTransport (no real network)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from knotwork import net
from knotwork.freshness import beacon
from knotwork.freshness.receipt import build_receipt, receipt_id_ok
from knotwork.knotcore.contracts import FreshnessReceipt
from knotwork.knotcore.seeds import offline_randomness

REPO = Path(__file__).resolve().parents[2]
PROFILE = {
    "freshness": {
        "drand_url": "https://api.drand.sh/public/latest",
        "nist_url": "https://beacon.nist.gov/beacon/2.0/pulse/last",
        "protocol_files": ["src/knotwork/knotcore/contracts.py", "configs/default_profile.yaml"],
        "emission_salt_path": ".secrets/emission_salt",
    }
}
DRAND = {"round": 4242, "randomness": "ab" * 32, "signature": "00"}
NIST = {"pulse": {"pulseIndex": 777, "outputValue": "cd" * 64}}


def _transport(calls: list[str]) -> httpx.MockTransport:
    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(str(req.url))
        if req.url.host == "api.drand.sh":
            return httpx.Response(200, json=DRAND)
        if req.url.host == "beacon.nist.gov":
            return httpx.Response(200, json=NIST)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_get_json_refused_under_deny(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOTWORK_NETWORK", "deny")
    calls: list[str] = []
    with pytest.raises(net.NetworkPolicyError):
        net.get_json(PROFILE["freshness"]["drand_url"], transport=_transport(calls))
    assert calls == []


def test_beacon_policy_only_allows_beacon_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOTWORK_NETWORK", "beacon")
    calls: list[str] = []
    with pytest.raises(net.NetworkPolicyError):
        net.get_json("https://openrouter.ai/api/v1/models", transport=_transport(calls))
    with pytest.raises(net.NetworkPolicyError):
        net.post_json("https://example.com/x", {}, {}, transport=_transport(calls))
    assert calls == []
    monkeypatch.setenv("KNOTWORK_NETWORK", "openrouter")
    with pytest.raises(net.NetworkPolicyError):
        net.get_json(PROFILE["freshness"]["drand_url"], transport=_transport(calls))


def test_drand_parsed_from_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOTWORK_NETWORK", "beacon")
    calls: list[str] = []
    b = beacon.fetch("drand", 0, PROFILE, transport=_transport(calls))
    assert calls == [PROFILE["freshness"]["drand_url"]]
    assert (b.source, b.randomness_hex, b.round) == ("drand", "ab" * 32, 4242)
    assert b.url == PROFILE["freshness"]["drand_url"] and b.fetched_at_utc


def test_nist_parsed_from_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOTWORK_NETWORK", "beacon")
    b = beacon.fetch("nist", 0, PROFILE, transport=_transport([]))
    assert b.source == "nist" and b.round == 777
    assert b.randomness_hex == hashlib.sha256(bytes.fromhex("cd" * 64)).hexdigest()


def test_bad_beacon_payloads_rejected() -> None:
    with pytest.raises(ValueError):
        beacon.parse_drand({"round": 1, "randomness": "zz" * 32}, "u")
    with pytest.raises(ValueError):
        beacon.parse_nist({"pulse": {"pulseIndex": 1, "outputValue": "ab" * 10}}, "u")
    with pytest.raises(ValueError):
        beacon.fetch("carrier-pigeon", 0, PROFILE)


def test_offline_beacon_and_receipt() -> None:
    b = beacon.fetch("offline", 7, PROFILE)
    assert (b.source, b.round, b.randomness_hex) == ("offline", 7, offline_randomness(7))
    r = build_receipt(b, PROFILE)
    assert r.fetched_at_utc is None and receipt_id_ok(r)
    assert build_receipt(beacon.fetch("offline", 7, PROFILE), PROFILE) == r


def test_receipt_id_excludes_fetched_at(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOTWORK_NETWORK", "beacon")
    b = beacon.fetch("drand", 0, PROFILE, transport=_transport([]))
    r = build_receipt(b, PROFILE)
    other = r.model_copy(update={"fetched_at_utc": "1999-01-01T00:00:00+00:00"})
    assert receipt_id_ok(r) and receipt_id_ok(other)
    assert not receipt_id_ok(r.model_copy(update={"round": 4243}))


def test_generate_with_drand_beacon_mocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`knot generate --beacon drand` path: every HTTP call is served by a MockTransport."""
    from knotwork.pipeline.generate import run_generate

    calls: list[str] = []
    mock = _transport(calls)
    real_client = httpx.Client

    def client(*args: Any, **kwargs: Any) -> httpx.Client:
        if kwargs.get("transport") is None:
            kwargs["transport"] = mock
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setenv("KNOTWORK_NETWORK", "beacon")
    out = tmp_path / "gen"
    rc = run_generate(
        "gauge",
        1,
        out,
        0,
        "drand",
        REPO / "configs" / "default_profile.yaml",
        "dev",
        False,
        None,
        1,
    )
    assert rc == 0
    assert calls == [PROFILE["freshness"]["drand_url"]]
    rec = FreshnessReceipt.model_validate_json((out / "freshness_receipt.json").read_text())
    assert rec.source == "drand" and rec.round == 4242 and rec.fetched_at_utc
    assert receipt_id_ok(rec)
    row = json.loads((out / "index.jsonl").read_text().splitlines()[0])
    task = json.loads((out / row["dir"] / "task.json").read_text())
    assert task["canary"] == rec.canary and task["freshness_receipt_id"] == rec.receipt_id
