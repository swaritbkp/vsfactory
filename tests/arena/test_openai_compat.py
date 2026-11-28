"""`provider: openai_compat` end to end against a real local HTTP server (loopback only): the
standard vision payload, base-URL normalisation, optional bearer key, retries and clean errors,
bounded replies, redirects refused, secret-free campaign files and no key on disk."""

from __future__ import annotations

import base64
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from arena_fixtures import config
from knotwork import net
from knotwork.arena import client, runner, storage
from knotwork.arena.campaign_yaml import export_campaign_yaml, load_campaign_yaml
from knotwork.arena.schemas import BlockSpec

KEY = "sk-local-0123456789abcdef-test-key"


class Server:
    """A tiny OpenAI-compatible server. `behaviour` picks the reply; every request is recorded."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.behaviour = "ok"
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                return

            def _send(self, code: int, body: bytes, headers: dict[str, str] | None = None) -> None:
                self.send_response(code)
                for k, v in (headers or {"Content-Type": "application/json"}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 (http.server API)
                outer.requests.append({"method": "GET", "path": self.path,
                                       "auth": self.headers.get("Authorization")})  # fmt: skip
                body = {"data": [{"id": "Qwen/Qwen2.5-VL-7B-Instruct"}, {"id": "gemma4:31b"}]}
                self._send(200, json.dumps(body).encode())

            def do_POST(self) -> None:  # noqa: N802 (http.server API)
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                outer.requests.append({"method": "POST", "path": self.path, "body": json.loads(raw),
                                       "auth": self.headers.get("Authorization")})  # fmt: skip
                if outer.behaviour == "error":
                    self._send(500, b'{"error": "boom"}')
                elif outer.behaviour == "bad_request":
                    self._send(400, b'{"error": "unsupported image"}')
                elif outer.behaviour == "redirect":
                    self._send(302, b"", {"Location": "http://169.254.169.254/latest/"})
                elif outer.behaviour == "huge":
                    self._send(200, b"x" * (client.MAX_RESPONSE_BYTES + 10))
                else:
                    reply = {"choices": [{"message": {"content": "Tracing.\nANSWER: PP-01"}}],
                             "usage": {"prompt_tokens": 900, "completion_tokens": 12}}  # fmt: skip
                    self._send(200, json.dumps(reply).encode())

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = int(self.httpd.server_address[1])
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def posts(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["method"] == "POST"]


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[Server]:
    srv = Server()
    monkeypatch.setenv("KNOTWORK_NETWORK", "custom")
    monkeypatch.setenv(net.CUSTOM_HOST_ENV, "127.0.0.1")
    yield srv
    srv.httpd.shutdown()


def _cfg(root: Path, base_url: str, **extra: Any) -> Any:
    if "blocks" in extra:
        extra["blocks"] = [BlockSpec.model_validate(b) for b in extra["blocks"]]
    return config(root, mode="live", provider="openai_compat", base_url=base_url,
                  spend_cap_usd=5.0, max_concurrency=1, **extra)  # fmt: skip


def _run(cfg: Any) -> runner.Runner:
    results = storage.resolve(cfg.results_dir)
    storage.write_session(cfg)
    r = runner.Runner(cfg, results, sleep=lambda s: None)
    assert r.run() == 0
    return r


def _all_text(*roots: Path) -> str:
    return "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for root in roots for p in root.rglob("*") if p.is_file() and p.suffix != ".png"
    )  # fmt: skip


@pytest.mark.parametrize("suffix", ["", "/", "/chat/completions"])
def test_vision_payload_and_normalised_url(arena_root, frozen, server, monkeypatch, suffix) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv(client.CUSTOM_KEY_ENV, KEY)
    cfg = _cfg(arena_root, server.base + suffix, default_models=["Qwen/Qwen2.5-VL-7B-Instruct"],
               blocks=[{"block_id": "b1", "domain": "rack_patch", "preset": "High stress",
                        "n": 2, "seed": 0}])  # fmt: skip
    r = _run(cfg)
    posts = server.posts()
    assert len(posts) == 6 and {p["path"] for p in posts} == {"/v1/chat/completions"}  # block b1
    body = posts[0]["body"]
    assert body["model"] == "Qwen/Qwen2.5-VL-7B-Instruct" and "provider" not in body
    image = body["messages"][1]["content"][0]["image_url"]["url"]
    assert image.startswith("data:image/png;base64,")
    assert base64.b64decode(image.split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"
    assert posts[0]["auth"] == f"Bearer {KEY}"
    trials = storage.load_trials(storage.resolve(cfg.results_dir))
    assert all(t.status == "ok" and t.extracted == "01" for t in trials.values())
    assert r.status.state == "done"
    assert KEY not in _all_text(storage.resolve(cfg.results_dir), arena_root)


def test_key_is_optional_for_local_servers(arena_root, frozen, server, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv(client.CUSTOM_KEY_ENV, raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = _cfg(arena_root, server.base, default_models=["gemma4:31b"],
               blocks=[{"block_id": "b1", "domain": "rack_patch", "preset": "Clean", "n": 1,
                        "seed": 0}])  # fmt: skip
    _run(cfg)
    assert server.posts() and {p["auth"] for p in server.posts()} == {None}


@pytest.mark.parametrize(
    ("behaviour", "attempts", "text"),
    [("error", runner.MAX_RETRIES + 1, "HTTP 500"), ("bad_request", 1, "HTTP 400"),
     ("redirect", 1, "redirected"), ("huge", runner.MAX_RETRIES + 1, "larger than")],
)  # fmt: skip
def test_failures_become_clean_api_errors(
    arena_root, frozen, server, behaviour, attempts, text
) -> None:  # type: ignore[no-untyped-def]
    server.behaviour = behaviour
    cfg = _cfg(arena_root, server.base, default_models=["m"],
               blocks=[{"block_id": "b1", "domain": "rack_patch", "preset": "Clean", "n": 1,
                        "seed": 0}])  # fmt: skip
    _run(cfg)
    trials = list(storage.load_trials(storage.resolve(cfg.results_dir)).values())
    assert trials and all(t.status == "api_error" for t in trials)
    assert all(t.attempts == attempts and text in t.error for t in trials)
    if behaviour == "redirect":
        assert all("169.254" not in r["path"] for r in server.requests)


def test_network_policy_allows_only_the_configured_host(
    arena_root, frozen, server, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv(net.CUSTOM_HOST_ENV, "other.example.com")
    with pytest.raises(net.NetworkPolicyError):
        net.check(server.base + "/chat/completions")
    monkeypatch.setenv("KNOTWORK_NETWORK", "deny")
    with pytest.raises(net.NetworkPolicyError):
        net.check(server.base + "/chat/completions", custom_host="127.0.0.1")


def test_ping_lists_models(server) -> None:  # type: ignore[no-untyped-def]
    found = client.ping(server.base, "gemma4:31b", public_demo=False)
    assert found.ok and "gemma4:31b is listed" in found.message
    missing = client.ping(server.base + "/", "llava:7b", KEY, public_demo=False)
    assert missing.ok and "not in its 2 listed models" in missing.message
    assert server.requests[-1]["path"] == "/v1/models" and server.requests[-1]["auth"] == (
        f"Bearer {KEY}"
    )
    refused = client.ping("http://169.254.169.254/v1", "m", public_demo=True)
    assert not refused.ok and "https" in refused.message


def test_runner_env_carries_only_its_own_key_and_host() -> None:
    env = {"OPENROUTER_API_KEY": "sk-or-server-secret-0000", client.CUSTOM_KEY_ENV: "stale",
           "PATH": "x"}  # fmt: skip
    custom = client.runner_env("openai_compat", "http://localhost:8000/v1", KEY, env)
    assert custom["KNOTWORK_NETWORK"] == "custom" and custom[net.CUSTOM_HOST_ENV] == "localhost"
    assert custom[client.CUSTOM_KEY_ENV] == KEY and "OPENROUTER_API_KEY" not in custom
    local = client.runner_env("openai_compat", "http://localhost:8000/v1", None, env)
    assert client.CUSTOM_KEY_ENV not in local
    routed = client.runner_env("openrouter", "", "sk-or-visitor-1111", env)
    assert (
        routed["OPENROUTER_API_KEY"] == "sk-or-visitor-1111" and net.CUSTOM_HOST_ENV not in routed
    )


def test_campaign_yaml_round_trip_is_secret_free(arena_root: Path) -> None:
    cfg = _cfg(arena_root, "http://localhost:11434/v1", default_models=["gemma4:31b"])
    text = export_campaign_yaml(cfg)
    assert "provider: openai_compat" in text and "base_url: http://localhost:11434/v1" in text
    assert KEY not in text and "api_key" not in text
    again = load_campaign_yaml(text)
    assert (again.provider, again.base_url, again.default_models) == (
        "openai_compat", "http://localhost:11434/v1", ["gemma4:31b"])  # fmt: skip
    short = load_campaign_yaml(
        "session_name: Local VLM\nmode: live\nprovider: openai_compat\n"
        "base_url: http://localhost:8000/v1/chat/completions\nmodel_id: Qwen/Qwen2.5-VL-7B-Instruct\n"
        "blocks:\n  - {block_id: b1, domain: tangle, preset: Clean, n: 2, seed: 0}\n"
    )
    assert short.default_models == ["Qwen/Qwen2.5-VL-7B-Instruct"]
    for bad, why in (
        ("api_key: sk-abcdefghijklmnop", "credentials"),
        (f"notes: {KEY}", "secret"),
        ("base_url: http://user:pw@localhost:8000/v1", "key field"),
        ("base_url: file:///etc/passwd", "http"),
    ):
        body = ("session_name: X\nmode: live\nprovider: openai_compat\nmodel_id: m\n"
                f"{bad}\nblocks:\n  - {{block_id: b1, domain: tangle, preset: Clean, n: 1, seed: 0}}\n")  # fmt: skip
        if "base_url" not in bad:
            body = body.replace("model_id: m\n", "model_id: m\nbase_url: http://localhost:1/v1\n")
        with pytest.raises(ValueError, match=why):
            load_campaign_yaml(body)


def test_campaign_yaml_rejects_google_aiza_literals() -> None:
    aiza = "AIzaSyDummyAuditKey1234567890abcd"
    for field in (f"model_id: {aiza}", f"notes: key={aiza}", f"session_name: {aiza}"):
        body = (
            "session_name: Gemini Test\nmode: live\nprovider: openai_compat\n"
            "base_url: https://generativelanguage.googleapis.com/v1beta/openai/\n"
            f"{field}\n"
            "blocks:\n  - {block_id: b1, domain: tangle, preset: Clean, n: 1, seed: 0}\n"
        )
        if field.startswith("session_name:"):
            body = (
                body.replace("session_name: Gemini Test\n", "") + "\nmodel_id: gemini-2.5-flash\n"
            )
        elif not field.startswith("model_id:"):
            body += "model_id: gemini-2.5-flash\n"
        with pytest.raises(ValueError, match="secret"):
            load_campaign_yaml(body)


def test_openrouter_campaign_files_keep_the_pre_v2_1_shape(arena_root: Path) -> None:
    text = export_campaign_yaml(config(arena_root, mode="live"))
    body = " ".join(line for line in text.splitlines() if not line.startswith("#"))
    assert "provider" not in body and "base_url" not in body
    assert load_campaign_yaml(text).provider == "openrouter"
