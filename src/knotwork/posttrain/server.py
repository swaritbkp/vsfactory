"""Stateless local HTTP RLVR verifier (`knot serve-verifier`), stdlib `http.server` only.

Endpoints (JSON in, JSON out):
* `GET  /health`      -> {"status": "ok", "kernel_sha256", "weights", "version"}
* `POST /score`       -> one rollout {data_source, solution_str, ground_truth, extra_info?}
* `POST /score_batch` -> {"items": [rollout, ...]} -> {"results": [...], "mean_score"}
* `GET  /llms.txt`    -> the repository's `llms.txt` (text/plain), a machine-readable summary for
  agents; an embedded copy is served when the server runs outside the repository

The server keeps no state between requests and makes no outbound connection, so it runs unchanged
under `KNOTWORK_NETWORK=deny`. It binds 127.0.0.1 by default; GPU rollout workers on other hosts
reach it through an SSH tunnel or an explicit `--host`.
"""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from knotwork import __version__
from knotwork.posttrain.reward import DEFAULT_WEIGHTS, compute_score

MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_BATCH = 4096
LOOPBACK = ("127.0.0.1", "localhost", "::1")
TEXT_ROUTES = ("/llms.txt",)
#: served when no repository `llms.txt` is next to the package (an installed wheel)
LLMS_FALLBACK = """# Knotwork

> Proof-carrying synthetic visual task factory: seeded 2-D scenes whose answers an isolated kernel
> re-derives from renderer object masks before an item is emitted. Offline by default.

## Verifier endpoints

- `GET /health`: status, kernel digest, reward weights
- `POST /score`: {data_source, solution_str, ground_truth, extra_info?} -> {score, ...}
- `POST /score_batch`: {"items": [...]} -> {results, n, mean_score}
- `GET /llms.txt`: this summary

## verl reward interface

- `knotwork.posttrain.reward.compute_score(data_source, solution_str, ground_truth, extra_info)`

## CLI

- `knot selftest`, `knot arena-run`, `knot gate`, `knot plugins`, `knot serve-verifier`,
  `knot posttrain-pack`, `knot bench-throughput`
"""


def kernel_digest() -> str:
    try:
        from knotwork.knotcore import engine

        return engine.kernel_sha256()
    except Exception:  # informative only: a broken checkout must not stop the verifier
        return "unknown"


def _score_one(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict) or "solution_str" not in item or "ground_truth" not in item:
        raise ValueError("a rollout needs solution_str and ground_truth")
    extra = item.get("extra_info")
    if extra is not None and not isinstance(extra, dict):
        raise ValueError("extra_info must be an object")
    gt = item["ground_truth"]
    gt_text = gt if isinstance(gt, str) else json.dumps(gt, sort_keys=True)
    return compute_score(
        str(item.get("data_source", "")), str(item["solution_str"]), gt_text, extra
    )


def llms_text() -> str:
    """The repository's `llms.txt`, or the embedded summary when it is not available."""
    from knotwork.render.canvas import REPO_ROOT

    path = REPO_ROOT / "llms.txt"
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else LLMS_FALLBACK
    except OSError:
        return LLMS_FALLBACK


def handle_text(method: str, path: str) -> tuple[int, str] | None:
    """Plain-text routes (GET /llms.txt); None when `path` is a JSON route."""
    route = path.split("?", 1)[0].rstrip("/") or "/"
    if method == "GET" and route in TEXT_ROUTES:
        return 200, llms_text()
    return None


def handle(method: str, path: str, body: bytes, kernel_sha: str) -> tuple[int, dict[str, Any]]:
    """Pure request router (unit-testable without sockets)."""
    route = path.split("?", 1)[0].rstrip("/") or "/"
    if method == "GET" and route == "/health":
        return 200, {"status": "ok", "kernel_sha256": kernel_sha, "version": __version__,
                     "weights": DEFAULT_WEIGHTS, "endpoints": ["/health", "/score", "/score_batch", "/llms.txt"]}  # fmt: skip
    if method != "POST" or route not in ("/score", "/score_batch"):
        return 404, {"error": f"no route {method} {route}"}
    try:
        payload = json.loads(body.decode("utf-8") or "null")
        if route == "/score":
            return 200, _score_one(payload)
        items = payload.get("items") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise ValueError("score_batch expects {'items': [...]} or a JSON list")
        if len(items) > MAX_BATCH:
            raise ValueError(f"at most {MAX_BATCH} rollouts per batch")
        results = [_score_one(x) for x in items]
        mean = sum(r["score"] for r in results) / len(results) if results else None
        return 200, {"results": results, "n": len(results), "mean_score": mean}
    except (ValueError, UnicodeDecodeError, AttributeError) as exc:
        return 400, {"error": str(exc)[:300]}


def make_server(host: str = "127.0.0.1", port: int = 8090) -> ThreadingHTTPServer:
    """A bound (not yet serving) verifier; port 0 picks a free port."""
    kernel_sha = kernel_digest()

    class Handler(BaseHTTPRequestHandler):
        server_version = "knotwork-verifier/" + __version__

        def _reply(self, code: int, data: dict[str, Any]) -> None:
            raw = json.dumps(data, sort_keys=True).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _reply_text(self, code: int, text: str) -> None:
            raw = text.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _dispatch(self, method: str) -> None:
            text = handle_text(method, self.path)
            if text is not None:
                self._reply_text(*text)
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                self._reply(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "body too large"})
                return
            body = self.rfile.read(length) if length else b""
            self._reply(*handle(method, self.path, body, kernel_sha))

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return  # quiet: rollout bodies may be large; access logs are not needed

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def serve_in_thread(host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    """Start a verifier on a background thread; returns (server, base_url). Call
    `server.shutdown(); server.server_close()` to stop it."""
    server = make_server(host, port)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                     daemon=True).start()  # fmt: skip
    host_s, port_n = str(server.server_address[0]), int(server.server_address[1])
    return server, f"http://{host_s}:{port_n}"


def run(host: str, port: int) -> int:
    server = make_server(host, port)
    host_s, port_n = str(server.server_address[0]), int(server.server_address[1])
    print(f"knotwork RLVR verifier on http://{host_s}:{port_n} (kernel {kernel_digest()[:12]})")
    if host not in LOOPBACK:
        print("warning: bound to a non-loopback address; any host that can reach it may score.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


__all__ = ["handle", "handle_text", "llms_text", "make_server", "run", "serve_in_thread"]
