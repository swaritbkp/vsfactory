"""The single network gateway. Every outbound HTTP call in Knotwork goes through `get_json` / `post_json`
and honours the process-wide policy `KNOTWORK_NETWORK` (SPEC 1.1):

* ``deny`` (default) -- no request is ever made;
* ``beacon``         -- GET to the configured drand / NIST beacon hosts only;
* ``openrouter``     -- POST/GET to openrouter.ai only (paid path, triple-locked by the CLI).
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

ALLOWED_HOSTS = {
    "beacon": {"api.drand.sh", "drand.cloudflare.com", "beacon.nist.gov"},
    "openrouter": {"openrouter.ai"},
}


class NetworkPolicyError(PermissionError):
    """A request was refused by the KNOTWORK_NETWORK policy."""


def policy() -> str:
    return os.environ.get("KNOTWORK_NETWORK", "deny")


def check(url: str) -> None:
    host = urlparse(url).hostname or ""
    allowed = ALLOWED_HOSTS.get(policy(), set())
    if host not in allowed:
        raise NetworkPolicyError(f"KNOTWORK_NETWORK={policy()!r} forbids a request to {host!r}")


def get_json(url: str, timeout: float = 10.0, transport: Any = None) -> Any:
    """GET ``url`` and decode JSON, after the policy check. ``transport`` lets tests inject
    ``httpx.MockTransport``."""
    check(url)
    import httpx

    with httpx.Client(timeout=timeout, transport=transport) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.json()


def post_json(
    url: str, body: Any, headers: dict[str, str], timeout: float = 90.0, transport: Any = None
) -> Any:
    """POST JSON after the policy check; returns the decoded JSON response (raises on HTTP errors)."""
    check(url)
    import httpx

    with httpx.Client(timeout=timeout, transport=transport) as client:
        resp = client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()
