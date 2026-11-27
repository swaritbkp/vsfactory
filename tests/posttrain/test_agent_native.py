"""Agent-native surface: root `llms.txt`, `GET /llms.txt` on the verifier, and external plugin
discovery through `entry_points(group="knotwork.plugins")` (mocked; nothing is installed)."""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner

from knotwork import cli as K
from knotwork.plugins import registry
from knotwork.posttrain import server

REPO = Path(__file__).resolve().parents[2]


def test_llms_txt_follows_the_spec_and_links_resolve() -> None:
    text = (REPO / "llms.txt").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "# Knotwork" and lines[2].startswith("> ")
    sections = re.findall(r"^## (.+)$", text, re.M)
    assert {"Docs", "CLI", "verl reward interface"} <= set(sections)
    for target in re.findall(r"\]\(([^)]+)\)", text):
        assert (REPO / target).is_file(), target
    for cmd in ("knot selftest", "knot arena-run", "knot gate", "knot serve-verifier",
                "knot posttrain-pack", "knot bench-throughput", "knot plugins"):  # fmt: skip
        assert cmd in text
    for doc in (
        "README.md",
        "VERIFICATION.md",
        "DATA_CARD.md",
        "BLOCKERS.md",
        "docs/PLUGIN_SDK.md",
    ):
        assert f"]({doc})" in text


def test_llms_route_is_text_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    assert server.handle_text("GET", "/llms.txt") == (200, server.llms_text())
    assert server.handle_text("GET", "/health") is None
    assert server.handle_text("POST", "/llms.txt") is None
    import knotwork.render.canvas as canvas

    monkeypatch.setattr(canvas, "REPO_ROOT", Path("/nonexistent-knotwork-root"))
    assert server.llms_text() == server.LLMS_FALLBACK


@pytest.fixture
def base_url() -> Iterator[str]:
    srv, url = server.serve_in_thread("127.0.0.1", 0)
    try:
        yield url
    finally:
        srv.shutdown()
        srv.server_close()


def test_get_llms_txt_over_http(base_url: str) -> None:
    with urllib.request.urlopen(base_url + "/llms.txt", timeout=10) as resp:
        assert resp.status == 200
        assert resp.headers["Content-Type"] == "text/plain; charset=utf-8"
        assert resp.read().decode("utf-8").startswith("# Knotwork")
    with urllib.request.urlopen(base_url + "/health", timeout=10) as resp:
        assert "/llms.txt" in json.loads(resp.read())["endpoints"]


# ------------------------------------------------------------------ external plugins
class _Good:
    name, version, families = "ext_dials", "0.3.1", ()

    def __getattr__(self, attr: str) -> Any:
        if attr in registry.REQUIRED_METHODS:
            return lambda *a, **k: None
        raise AttributeError(attr)


class _EP:
    def __init__(self, name: str, obj: Any = None, error: Exception | None = None) -> None:
        self.name, self.value, self._obj, self._error = name, f"pkg_{name}:PLUGIN", obj, error
        self.dist = SimpleNamespace(name=f"knotwork-{name}", version="0.3.1")

    def load(self) -> Any:
        if self._error:
            raise self._error
        return self._obj


@pytest.fixture
def fake_eps(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    half = SimpleNamespace(name="ext_half", version="1", families=(), render=lambda: None)
    eps = [_EP("ext_dials", _Good()), _EP("tangle", _Good()), _EP("ext_half", half),
           _EP("ext_boom", error=ImportError("no module named boom")),
           _EP("ext_alias", _Good())]  # fmt: skip
    monkeypatch.setattr(registry, "_entry_points", lambda: eps)
    registry.refresh()
    yield
    registry.refresh()


def test_entry_point_discovery_and_rejections(fake_eps: None) -> None:
    assert registry.available()[-1] == "ext_dials" and "ext_half" not in registry.available()
    assert registry.get_plugin("ext_dials").version == "0.3.1"
    rows = {(r.name, r.source): r for r in registry.list_plugins()}
    assert rows[("tangle", "builtin")].status == "ok"
    assert rows[("ext_dials", "external")].status == "ok"
    assert "collides" in rows[("tangle", "external")].detail
    assert "missing method claims()" in rows[("ext_half", "external")].detail
    assert "load failed: ImportError" in rows[("ext_boom", "external")].detail
    assert "!= PLUGIN.name" in rows[("ext_alias", "external")].detail
    assert {"rack_patch", "tangle", "gauge", "panel"} <= {n for n, s in rows if s == "builtin"}


def test_knot_plugins_cli(fake_eps: None) -> None:
    runner = CliRunner()
    res = runner.invoke(K.cli, ["plugins"])
    assert res.exit_code == 0, res.output
    assert res.output.isascii() and "ext_dials" in res.output and "rejected" in res.output
    res = runner.invoke(K.cli, ["plugins", "--json"])
    rows = json.loads(res.output)
    assert {r["name"] for r in rows if r["status"] == "ok"} >= {"tangle", "ext_dials"}
    assert set(rows[0]) == {"name", "source", "version", "status", "origin", "detail"}
