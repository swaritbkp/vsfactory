"""Public-randomness beacons (SPEC 11.1): drand, NIST, or the offline stand-in."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from knotwork import net
from knotwork.knotcore.seeds import offline_randomness


@dataclass(frozen=True)
class BeaconValue:
    source: str  # drand | nist | offline
    randomness_hex: str  # 64 hex chars
    round: int | None
    url: str | None
    fetched_at_utc: str | None


def offline(seed: int) -> BeaconValue:
    return BeaconValue("offline", offline_randomness(seed), seed, None, None)


def parse_drand(doc: dict[str, Any], url: str) -> BeaconValue:
    rnd = str(doc["randomness"]).lower()
    if len(rnd) != 64 or any(c not in "0123456789abcdef" for c in rnd):
        raise ValueError("drand randomness must be 64 hex chars")
    return BeaconValue("drand", rnd, int(doc["round"]), url, _now())


def parse_nist(doc: dict[str, Any], url: str) -> BeaconValue:
    pulse = doc["pulse"]
    out = str(pulse["outputValue"]).lower()
    if len(out) != 128:
        raise ValueError("NIST outputValue must be 128 hex chars")
    rnd = hashlib.sha256(bytes.fromhex(out)).hexdigest()
    return BeaconValue("nist", rnd, int(pulse["pulseIndex"]), url, _now())


def fetch(source: str, seed: int, profile: dict[str, Any], transport: Any = None) -> BeaconValue:
    """Beacon value for ``source``; network sources go through `knotwork.net` (policy `beacon`)."""
    if source == "offline":
        return offline(seed)
    fr = profile["freshness"]
    if source == "drand":
        return parse_drand(net.get_json(fr["drand_url"], transport=transport), fr["drand_url"])
    if source == "nist":
        return parse_nist(net.get_json(fr["nist_url"], transport=transport), fr["nist_url"])
    raise ValueError(f"unknown beacon {source!r}")


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
