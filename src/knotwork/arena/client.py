"""Chat-completions client for the Model Arena's three providers.

* `mock`          -- Offline Mock (no key, no network; `runner.mock_reply`);
* `openrouter`    -- Live · OpenRouter (`OPENROUTER_API_KEY`);
* `openai_compat` -- Live · any OpenAI-compatible server: vLLM, SGLang, Ollama, OpenAI, the Gemini
  OpenAI endpoint. The key is optional (local servers often need none) and travels only in memory
  or in the runner's environment as `KNOTWORK_CUSTOM_API_KEY`; it is never written to a file.

Every call goes through `knotwork.net.check` (the single network gateway), uses explicit timeouts
(connect 10 s, read 45 s for custom endpoints), never follows redirects, and reads at most 2 MB of
response body, so a hostile or broken endpoint cannot redirect the server, hang it or fill memory.
In the public demo a custom URL is re-validated against `ssrf_guard` right before each request.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from knotwork import net
from knotwork.studio import demo_guard
from knotwork.studio.ssrf_guard import normalise, validate_custom_base_url

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
CUSTOM_KEY_ENV = "KNOTWORK_CUSTOM_API_KEY"
PROVIDERS: dict[str, str] = {
    "mock": "Offline Mock (No Key)",
    "openrouter": "Live · OpenRouter",
    "openai_compat": "Live · Custom endpoint",  # any OpenAI-compatible server
}
OPENROUTER_TIMEOUT_S = 60.0
CONNECT_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 45.0
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
NETWORK_ERROR = 599  # status used for timeouts, transport errors and oversized replies


class EndpointError(ValueError):
    """A custom endpoint that may not be called (bad URL, or refused by the SSRF guard)."""


@dataclass(frozen=True)
class Endpoint:
    provider: str
    url: str
    key: str = ""
    host: str = ""

    def headers(self) -> dict[str, str]:
        out = {"Content-Type": "application/json"}
        if self.key:
            out["Authorization"] = f"Bearer {self.key}"
        if self.provider == "openrouter":
            out["X-Title"] = "knotwork-model-arena"
        return out

    @property
    def timeout(self) -> Any:
        import httpx

        if self.provider == "openrouter":
            return httpx.Timeout(OPENROUTER_TIMEOUT_S)
        return httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)


def custom_endpoint(base_url: str, key: str = "", public_demo: bool | None = None) -> Endpoint:
    """A validated custom endpoint. Raises EndpointError with a readable reason."""
    demo = demo_guard.is_public_demo() if public_demo is None else public_demo
    check = validate_custom_base_url(base_url, public_demo=demo)
    if not check.ok:
        raise EndpointError(check.reason)
    return Endpoint("openai_compat", check.endpoint, key.strip(), check.host)


def endpoint_for(provider: str, base_url: str, key: str) -> Endpoint:
    if provider == "openai_compat":
        return custom_endpoint(base_url, key)
    return Endpoint("openrouter", OPENROUTER_ENDPOINT, key, "openrouter.ai")


def body(endpoint: Endpoint, model: str, messages: list[dict[str, Any]], temperature: float,
         max_tokens: int) -> dict[str, Any]:  # fmt: skip
    """The standard OpenAI chat-completions body; OpenRouter also gets its usage / routing hints."""
    out: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature,
                           "max_tokens": max_tokens}  # fmt: skip
    if endpoint.provider == "openrouter":
        out["usage"] = {"include": True}
        out["provider"] = {"data_collection": "deny", "allow_fallbacks": True}
    return out


def _bounded(resp: Any) -> bytes:
    chunks, size = [], 0
    for chunk in resp.iter_bytes():
        size += len(chunk)
        if size > MAX_RESPONSE_BYTES:
            raise OverflowError(f"the reply is larger than {MAX_RESPONSE_BYTES // 1024} KB")
        chunks.append(chunk)
    return b"".join(chunks)


def _send(method: str, endpoint: Endpoint, url: str, transport: Any,
          payload: dict[str, Any] | None = None) -> tuple[int, bytes]:  # fmt: skip
    """One request with redirects off and a bounded body: (status, raw bytes)."""
    import httpx

    kwargs: dict[str, Any] = {"headers": endpoint.headers()}
    if payload is not None:
        kwargs["json"] = payload
    with (
        httpx.Client(timeout=endpoint.timeout, transport=transport, follow_redirects=False) as c,
        c.stream(method, url, **kwargs) as resp,
    ):
        return resp.status_code, _bounded(resp)


def post(endpoint: Endpoint, payload: dict[str, Any], transport: Any = None,
         policy_override: str | None = None) -> tuple[int, Any, str]:  # fmt: skip
    """One POST. Returns (status, decoded JSON or None, error text); never raises for HTTP or
    network failures (status 599), so the runner can retry them. Raises NetworkPolicyError or
    EndpointError when the call is not allowed at all."""
    import httpx

    if endpoint.provider == "openai_compat" and demo_guard.is_public_demo():
        custom_endpoint(endpoint.url, public_demo=True)  # re-check right before the request
    net.check(endpoint.url, policy_override, custom_host=endpoint.host or None)
    try:
        status, raw = _send("POST", endpoint, endpoint.url, transport, payload)
    except (httpx.TimeoutException, httpx.TransportError, OverflowError) as exc:
        return NETWORK_ERROR, None, f"{type(exc).__name__}: {exc}"[:200]
    try:
        data = json.loads(raw) if raw else None
    except ValueError:
        data = None
    if 300 <= status < 400:
        return status, data, f"HTTP {status}: the endpoint redirected; use its final URL"
    error = "" if status == 200 else f"HTTP {status}: {raw[:160].decode('utf-8', 'replace')}"
    return status, data, error


def reply_text(data: Any) -> tuple[str, int, int, float | None]:
    """(text, prompt tokens, completion tokens, provider-reported cost) of a chat reply."""
    choice = ((data or {}).get("choices") or [{}])[0] if isinstance(data, dict) else {}
    content = (choice.get("message") or {}).get("content") or ""
    if isinstance(content, list):  # some servers return content parts
        content = "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    usage = (data or {}).get("usage") or {} if isinstance(data, dict) else {}
    cost = usage.get("cost")
    return (str(content), int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            float(cost) if isinstance(cost, (int, float)) else None)  # fmt: skip


@dataclass(frozen=True)
class PingResult:
    ok: bool
    message: str
    latency_ms: float = 0.0
    models: tuple[str, ...] = ()


def ping(base_url: str, model: str, key: str = "", transport: Any = None,
         public_demo: bool | None = None) -> PingResult:  # fmt: skip
    """`Test Endpoint`: GET `<base>/models` (free on vLLM, SGLang, Ollama, OpenAI and Gemini).
    Reports whether the server answers and, when it lists models, whether `model` is among them."""
    import httpx

    try:
        endpoint = custom_endpoint(base_url, key, public_demo)
    except EndpointError as exc:
        return PingResult(False, str(exc))
    url = endpoint.url.removesuffix("/chat/completions") + "/models"
    try:
        net.check(url, "custom", custom_host=endpoint.host)
    except net.NetworkPolicyError as exc:
        return PingResult(False, str(exc))
    t0 = time.perf_counter()
    try:
        status, raw = _send("GET", endpoint, url, transport)
    except (httpx.TimeoutException, httpx.TransportError, OverflowError) as exc:
        return PingResult(False, f"No answer from {endpoint.host}: {type(exc).__name__}.")
    ms = (time.perf_counter() - t0) * 1000
    if status in (401, 403):
        return PingResult(False, f"{endpoint.host} refused the key (HTTP {status}).", ms)
    if 300 <= status < 400:
        return PingResult(False, f"{endpoint.host} redirected (HTTP {status}); use its final URL.")
    if status != 200:
        return PingResult(True, f"{endpoint.host} answered HTTP {status} on /models; chat "
                          "completions may still work.", ms)  # fmt: skip
    try:
        listed = tuple(str(m.get("id")) for m in json.loads(raw).get("data", []) if m.get("id"))
    except (ValueError, AttributeError):
        listed = ()
    if listed and model and model not in listed:
        return PingResult(True, f"Reachable in {ms:.0f} ms, but {model} is not in its "
                          f"{len(listed)} listed models.", ms, listed)  # fmt: skip
    found = f"; {model} is listed" if listed and model else ""
    return PingResult(True, f"Reachable in {ms:.0f} ms{found}.", ms, listed)


def runner_env(
    provider: str, base_url: str, key: str | None, env: dict[str, str]
) -> dict[str, str]:
    """The environment of a background runner: the network policy for its provider and the key
    (only in memory of the child process). Strips every inherited key first."""
    out = {k: v for k, v in env.items() if k not in ("OPENROUTER_API_KEY", CUSTOM_KEY_ENV,
                                                     net.CUSTOM_HOST_ENV)}  # fmt: skip
    if provider == "openai_compat":
        out["KNOTWORK_NETWORK"] = "custom"
        out[net.CUSTOM_HOST_ENV] = custom_endpoint(base_url).host
        if key:
            out[CUSTOM_KEY_ENV] = key
    else:
        out["KNOTWORK_NETWORK"] = "openrouter"
        if key:
            out["OPENROUTER_API_KEY"] = key
    return out


def env_key(provider: str) -> str:
    name = CUSTOM_KEY_ENV if provider == "openai_compat" else "OPENROUTER_API_KEY"
    return os.environ.get(name, "").strip()


__all__ = [
    "CUSTOM_KEY_ENV", "MAX_RESPONSE_BYTES", "NETWORK_ERROR", "OPENROUTER_ENDPOINT", "PROVIDERS",
    "Endpoint", "EndpointError", "PingResult", "body", "custom_endpoint", "endpoint_for",
    "env_key", "normalise", "ping", "post", "reply_text", "runner_env",
]  # fmt: skip
