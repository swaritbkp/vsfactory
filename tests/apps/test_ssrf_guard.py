"""The custom-endpoint SSRF guard: strict in the public demo (HTTPS, public unicast addresses
only, no credentials, no internal names), permissive for local / self-hosted servers. DNS is
never touched: every check runs against an injected resolver."""

from __future__ import annotations

import socket
from typing import Any

import httpx
import pytest

from knotwork.arena import client
from knotwork.studio.ssrf_guard import normalise, validate_custom_base_url

PUBLIC_V4 = "93.184.216.34"


def resolver(table: dict[str, list[str]]) -> Any:
    def resolve(host: str, port: object, *args: object, **kwargs: object) -> list[tuple[Any, ...]]:
        if host not in table:
            raise socket.gaierror(socket.EAI_NONAME, "unknown host")
        out = []
        for ip in table[host]:
            family = socket.AF_INET6 if ":" in ip else socket.AF_INET
            out.append((family, socket.SOCK_STREAM, 6, "", (ip, port)))
        return out

    return resolve


PUBLIC = resolver({"api.example.com": [PUBLIC_V4], "api.openai.com": [PUBLIC_V4],
                   "rebind.example.com": [PUBLIC_V4, "10.1.2.3"],
                   "v6.example.com": ["2606:4700:4700::1111"],
                   "meta6.example.com": ["fd20:ce::254"],
                   "mapped.example.com": ["::ffff:169.254.169.254"]})  # fmt: skip


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/v1",
        "http://metadata.google.internal/computeMetadata/v1",
        "http://127.0.0.1:8000/v1",
        "http://localhost:8000/v1",
        "https://localhost/v1",
        "https://api.localhost/v1",
        "https://printer.local/v1",
        "https://metadata.google.internal/v1",
        "https://metadata.goog/v1",
        "https://10.0.0.1/v1",
        "https://192.168.1.1/v1",
        "https://172.16.0.5/v1",
        "https://100.64.0.1/v1",
        "https://169.254.169.254/v1",
        "https://[fd20:ce::254]/v1",
        "https://[::1]/v1",
        "https://[fe80::1]/v1",
        "https://0.0.0.0/v1",
        "https://224.0.0.1/v1",
        "https://user:pass@api.example.com/v1",
        "ftp://api.example.com/v1",
        "file:///etc/passwd",
        "https://rebind.example.com/v1",  # one private record is enough to refuse
        "https://meta6.example.com/v1",
        "https://mapped.example.com/v1",
        "https://intranet/v1",
        "https://no-such-host.example/v1",
        "",
        "https://api.example.com/v1 extra",
    ],
)
def test_public_demo_refuses_internal_and_unsafe_urls(url: str) -> None:
    check = validate_custom_base_url(url, public_demo=True, resolver=PUBLIC)
    assert not check.ok and check.reason and not check.endpoint


@pytest.mark.parametrize(
    ("url", "endpoint"),
    [
        ("https://api.example.com/v1", "https://api.example.com/v1/chat/completions"),
        ("https://api.openai.com/v1/", "https://api.openai.com/v1/chat/completions"),
        ("https://api.example.com/v1/chat/completions", "https://api.example.com/v1/chat/completions"),
        ("https://v6.example.com/v1beta/openai/", "https://v6.example.com/v1beta/openai/chat/completions"),
        ("https://93.184.216.34/v1", "https://93.184.216.34/v1/chat/completions"),
    ],
)  # fmt: skip
def test_public_demo_allows_public_https(url: str, endpoint: str) -> None:
    check = validate_custom_base_url(url, public_demo=True, resolver=PUBLIC)
    assert check.ok, check.reason
    assert check.endpoint == endpoint


@pytest.mark.parametrize(
    "url",
    ["http://localhost:8000/v1", "http://127.0.0.1:8000/v1", "http://localhost:11434/v1",
     "http://192.168.1.20:30000/v1", "https://api.openai.com/v1"],
)  # fmt: skip
def test_self_hosted_allows_local_servers_without_dns(url: str) -> None:
    def never(*args: object, **kwargs: object) -> list[tuple[Any, ...]]:
        raise AssertionError("local mode does not resolve names")

    check = validate_custom_base_url(url, public_demo=False, resolver=never)
    assert check.ok, check.reason
    assert check.endpoint.endswith("/v1/chat/completions")


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "ftp://host/v1", "http://u:p@localhost:8000/v1", "localhost:8000"]
)
def test_self_hosted_still_refuses_other_schemes_and_credentials(url: str) -> None:
    assert not validate_custom_base_url(url, public_demo=False).ok


def test_normalisation_variants() -> None:
    want = "http://localhost:8000/v1/chat/completions"
    for url in ("http://localhost:8000/v1", "http://localhost:8000/v1/",
                "http://localhost:8000/v1/chat/completions", "http://localhost:8000/v1/chat/completions/",
                "HTTP://localhost:8000/v1?x=1#frag"):  # fmt: skip
        assert normalise(url) == want


def test_redirects_are_never_followed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A public endpoint answering 302 to the metadata address is reported, not followed."""
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(302, headers={"Location": "http://169.254.169.254/latest/"})

    monkeypatch.setenv("KNOTWORK_NETWORK", "custom")
    monkeypatch.setenv("KNOTWORK_CUSTOM_HOST", "localhost")
    endpoint = client.custom_endpoint("http://localhost:9/v1", public_demo=False)
    status, data, error = client.post(endpoint, {"model": "m"}, httpx.MockTransport(handle))
    assert status == 302 and "redirected" in error and data is None
    assert seen == ["http://localhost:9/v1/chat/completions"]  # exactly one request, no hop


def test_public_demo_rechecks_the_url_right_before_each_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from knotwork.studio import demo_guard

    monkeypatch.setenv(demo_guard.ENV, "1")
    endpoint = client.Endpoint("openai_compat", "http://127.0.0.1:9/v1/chat/completions", "",
                               "127.0.0.1")  # fmt: skip
    with pytest.raises(client.EndpointError):
        client.post(endpoint, {"model": "m"}, httpx.MockTransport(lambda r: httpx.Response(200)))
