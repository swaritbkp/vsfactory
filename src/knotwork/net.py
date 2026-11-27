"""The single network gateway. Every outbound HTTP call in Knotwork goes through `get_json` / `post_json`
and honours the process-wide policy `KNOTWORK_NETWORK` (SPEC 1.1):

* ``deny`` (default) -- no request is ever made;
* ``beacon``         -- GET to the configured drand / NIST beacon hosts only;
* ``openrouter``     -- POST/GET to openrouter.ai only (paid path, triple-locked by the CLI);
* ``custom``         -- POST/GET to the one host named by ``KNOTWORK_CUSTOM_HOST`` only (a custom
  OpenAI-compatible endpoint a person configured for a Model Arena run or a Test Endpoint click).
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

ALLOWED_HOSTS = {
    "beacon": {"api.drand.sh", "drand.cloudflare.com", "beacon.nist.gov"},
    "openrouter": {"openrouter.ai"},
}
CUSTOM_HOST_ENV = "KNOTWORK_CUSTOM_HOST"


class NetworkPolicyError(PermissionError):
    """A request was refused by the KNOTWORK_NETWORK policy."""


def policy() -> str:
    return os.environ.get("KNOTWORK_NETWORK", "deny")


def check(url: str, policy_override: str | None = None, custom_host: str | None = None) -> None:
    """Refuse ``url`` unless the policy allows its host. ``policy_override`` is passed only by a call
    a person started with an explicit click (Model Arena: refresh catalog, AI brief, Test Endpoint);
    it still only reaches the hosts of that policy. Under ``custom`` the single allowed host is
    ``custom_host`` (a click) or else ``KNOTWORK_CUSTOM_HOST`` (a runner process)."""
    active = policy_override or policy()
    host = (urlparse(url).hostname or "").lower()
    if active == "custom":
        named = (custom_host or os.environ.get(CUSTOM_HOST_ENV, "")).strip().lower()
        allowed = {named} if named else set()
    else:
        allowed = ALLOWED_HOSTS.get(active, set())
    if host not in allowed:
        raise NetworkPolicyError(f"KNOTWORK_NETWORK={active!r} forbids a request to {host!r}")


def get_json(
    url: str,
    timeout: float = 10.0,
    transport: Any = None,
    headers: dict[str, str] | None = None,
    policy_override: str | None = None,
) -> Any:
    """GET ``url`` and decode JSON, after the policy check. ``transport`` lets tests inject
    ``httpx.MockTransport``."""
    check(url, policy_override)
    import httpx

    with httpx.Client(timeout=timeout, transport=transport) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()


def post_json(
    url: str,
    body: Any,
    headers: dict[str, str],
    timeout: float = 90.0,
    transport: Any = None,
    policy_override: str | None = None,
) -> Any:
    """POST JSON after the policy check; returns the decoded JSON response (raises on HTTP errors)."""
    check(url, policy_override)
    import httpx

    with httpx.Client(timeout=timeout, transport=transport) as client:
        resp = client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def apost_json(
    url: str, body: Any, headers: dict[str, str], timeout: float = 90.0, transport: Any = None
) -> tuple[int, Any]:
    """Async POST after the policy check. Returns (status_code, decoded JSON or None); never raises
    on HTTP status so callers can retry 429/5xx (SPEC 12.4)."""
    check(url)
    import httpx

    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        resp = await client.post(url, json=body, headers=headers)
        try:
            data = resp.json()
        except ValueError:
            data = None
        return resp.status_code, data
