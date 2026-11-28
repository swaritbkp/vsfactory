"""Validation of a custom OpenAI-compatible `base_url` (stdlib only: `ipaddress`, `socket`,
`urllib.parse`), and its normalisation to the chat-completions endpoint.

Two policies:

* **Public demo** (`public_demo=True`, a hosted Studio that strangers can reach): the URL must be
  `https://`, carry no credentials, name no internal host (`localhost`, `*.localhost`, `*.local`,
  `*.internal`, the GCP metadata names), and every A / AAAA record it resolves to must be a public
  unicast address: loopback, private (RFC 1918 / ULA), shared CGNAT, link-local (incl.
  169.254.169.254 and fe80::/10), multicast, unspecified, reserved and the GCP IPv6 metadata
  address fd20:ce::254 are refused. Callers must also send the request with redirects disabled, so
  a public host cannot 302 the server to an internal address after the check.
* **Local / self-hosted** (`public_demo=False`): `http://` and `https://` to any host, including
  `localhost`, `127.0.0.1` and LAN addresses (vLLM, SGLang, Ollama on the same machine or network).
  Other schemes and embedded credentials are still refused.

The check returns a result instead of raising, so the UI can show the reason next to the field.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

CHAT_PATH = "/chat/completions"
BLOCKED_HOSTS = frozenset({"localhost", "metadata.google.internal", "metadata.goog", "metadata"})
BLOCKED_SUFFIXES = (".localhost", ".local", ".internal", ".localdomain", ".home.arpa")
GCP_METADATA_V6 = ipaddress.ip_address("fd20:ce::254")
CGNAT = ipaddress.ip_network("100.64.0.0/10")
MAX_URL = 2048
Resolver = Callable[..., list[tuple[Any, ...]]]


@dataclass(frozen=True)
class UrlCheck:
    ok: bool
    reason: str = ""
    endpoint: str = ""  # the normalised .../chat/completions URL when ok
    host: str = ""


def normalise(base_url: str) -> str:
    """`http://h:8000/v1`, `.../v1/` and `.../v1/chat/completions` all map to
    `http://h:8000/v1/chat/completions`; query strings and fragments are dropped."""
    parts = urlsplit(base_url.strip())
    path = parts.path.rstrip("/")
    if path.endswith(CHAT_PATH):
        path = path[: -len(CHAT_PATH)]
    return urlunsplit((parts.scheme.lower(), parts.netloc, path.rstrip("/") + CHAT_PATH, "", ""))


def _ip_problem(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return _ip_problem(ip.ipv4_mapped)
    if ip == GCP_METADATA_V6:
        return "the cloud metadata address"
    checks = (
        (ip.is_loopback, "a loopback address"),
        (ip.is_link_local, "a link-local address"),
        (ip.is_multicast, "a multicast address"),
        (ip.is_unspecified, "an unspecified address"),
        (ip.is_reserved, "a reserved address"),
        (isinstance(ip, ipaddress.IPv4Address) and ip in CGNAT, "a shared (CGNAT) address"),
        (ip.is_private, "a private network address"),
        (not ip.is_global, "a non-public address"),
    )
    return next((why for bad, why in checks if bad), None)


def _host_problem(host: str) -> str | None:
    name = host.lower().rstrip(".")
    if name in BLOCKED_HOSTS or name.endswith(BLOCKED_SUFFIXES):
        return f"{host} is an internal host name"
    if "." not in name and ":" not in name:
        return f"{host} is not a fully qualified public host name"
    return None


def validate_custom_base_url(
    url: str, public_demo: bool, resolver: Resolver = socket.getaddrinfo
) -> UrlCheck:
    """Whether `url` may be called as a custom OpenAI-compatible endpoint (see module docstring).
    `resolver` is injectable so tests never touch DNS."""
    text = (url or "").strip()
    if not text:
        return UrlCheck(False, "Enter the endpoint's base URL, for example https://host/v1.")
    if len(text) > MAX_URL or any(c.isspace() or ord(c) < 32 for c in text):
        return UrlCheck(False, "The URL is too long or contains spaces or control characters.")
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError:
        return UrlCheck(False, "The URL could not be parsed.")
    scheme = parts.scheme.lower()
    allowed = ("https",) if public_demo else ("http", "https")
    if scheme not in allowed:
        need = "https://" if public_demo else "http:// or https://"
        return UrlCheck(False, f"Use an {need} URL (got {scheme or 'no scheme'}).")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        return UrlCheck(False, "Put the API key in the key field, not in the URL.")
    host = parts.hostname or ""
    if not host:
        return UrlCheck(False, "The URL has no host.")
    endpoint = normalise(text)
    if not public_demo:
        return UrlCheck(True, endpoint=endpoint, host=host)
    try:
        literal: ipaddress.IPv4Address | ipaddress.IPv6Address | None = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:  # an IP in the URL: judged as written, no lookup needed
        why = _ip_problem(literal)
        if why:
            return UrlCheck(False, f"The public demo cannot call {host}: it is {why}.", host=host)
        return UrlCheck(True, endpoint=endpoint, host=host)
    why = _host_problem(host)
    if why:
        return UrlCheck(False, f"The public demo cannot call {why}.", host=host)
    try:
        infos = resolver(host, port or 443, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as exc:
        return UrlCheck(False, f"{host} does not resolve ({type(exc).__name__}).", host=host)
    addresses = {str(info[4][0]).split("%", 1)[0] for info in infos if info and len(info) > 4}
    if not addresses:
        return UrlCheck(False, f"{host} does not resolve to any address.", host=host)
    for raw in sorted(addresses):
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError:
            return UrlCheck(False, f"{host} resolves to an unreadable address.", host=host)
        why = _ip_problem(ip)
        if why:
            return UrlCheck(False, f"The public demo cannot call {host}: it resolves to {why}.",
                            host=host)  # fmt: skip
    return UrlCheck(True, endpoint=endpoint, host=host)


__all__ = ["CHAT_PATH", "UrlCheck", "normalise", "validate_custom_base_url"]
