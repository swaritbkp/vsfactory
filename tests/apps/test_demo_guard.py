"""`KNOTWORK_PUBLIC_DEMO=1` sandbox guard: confinement, caps and no server key when on; no effect
at all when unset or `0`. Pure functions, no rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from arena_fixtures import config
from knotwork.arena import keys, storage
from knotwork.arena.schemas import BlockSpec
from knotwork.studio import demo_guard as G

KEY = "sk-or-v1-server-side-secret-0123456789"


@pytest.fixture
def demo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv(G.ENV, "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    root = tmp_path / "knotwork_demo"
    monkeypatch.setattr(G, "SANDBOX_ROOT", root)
    return G.sandbox_root()


def test_confinement_and_traversal(demo: Path) -> None:
    assert G.confine("runs/a/results") == demo / "runs" / "a" / "results"
    assert G.confine(demo / "x") == demo / "x"
    for bad in ("../escape", "runs/../../escape", str(demo.parent / "elsewhere"), "/etc"):
        with pytest.raises(G.SandboxViolation):
            G.confine(bad)
        assert G.path_problem(bad)
    with pytest.raises(G.SandboxViolation):
        storage.resolve("../outside")  # the Arena's own path resolution is confined too
    assert storage.resolve("s/results") == demo / "s" / "results"


def test_caps_and_no_server_key(demo: Path) -> None:
    assert G.limits() == G.Limits(10, 10, 3, 3, 1.0)
    assert G.cap(100_000, 10) == 10 and G.cap(5, 10) == 5
    assert G.server_api_key() is None and keys.detect() == (None, "")
    ok = config(demo, data_dir=demo / "data", blocks=[
        BlockSpec(block_id="b1", domain="tangle", preset="Clean", n=10, seed=0)])  # fmt: skip
    assert G.config_problems(ok) == []
    big = config(demo, data_dir=demo / "data", repeats=4, spend_cap_usd=5.0, blocks=[
        BlockSpec(block_id=f"b{i}", domain="tangle", preset="Clean", n=11, seed=0)
        for i in range(4)])  # fmt: skip
    found = " ".join(G.config_problems(big))
    for word in ("3 blocks", "10 items per block", "k = 3", "$1.00"):
        assert word in found
    outside = config(demo.parent, data_dir=demo.parent / "data")
    assert any("Dataset folder" in p for p in G.config_problems(outside))


@pytest.mark.parametrize("value", [None, "0", "true", ""])
def test_default_imposes_nothing(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv(G.ENV, raising=False)
    else:
        monkeypatch.setenv(G.ENV, value)
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    assert not G.is_public_demo() and G.limits() is None
    assert G.cap(100_000, 10) == 100_000
    assert G.confine("../anywhere") == Path("../anywhere") and G.path_problem("/etc") is None
    assert G.server_api_key() == KEY and keys.detect() == (KEY, "environment")
    huge = config(Path("/any"), repeats=5, spend_cap_usd=200.0)
    assert G.config_problems(huge) == []


# ------------------------------------------------------------------ hardened public demo
def test_offline_mock_only_is_enforced_server_side(demo: Path) -> None:
    modes = {"Offline Mock (No Key)": "mock", "Live (OpenRouter)": "live"}
    assert G.execution_modes(modes) == {"Offline Mock (No Key)": "mock"}
    live = config(demo, data_dir=demo / "data", mode="live")
    assert G.offline_only(live).mode == "mock"
    assert "The sandbox runs Offline Mock only." in G.config_problems(live)
    assert not G.campaign_import_allowed()


def test_live_resume_is_refused_in_the_demo(demo: Path) -> None:
    from knotwork.arena import planning

    cfg = config(demo, data_dir=demo / "data", mode="live")
    storage.write_session(cfg)
    ws = storage.Workspace(demo / "ws")
    problems, process = planning.resume(storage.resolve(cfg.results_dir), ws, KEY)
    assert process is None and "Offline Mock" in problems[0]


def test_external_plugins_are_not_loaded_in_the_demo(
    demo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from knotwork.plugins import registry

    def scan() -> list[object]:
        raise AssertionError("entry points must not be scanned in the public demo")

    monkeypatch.setattr(registry, "_entry_points", scan)
    registry.refresh()
    try:
        assert registry.available() == ("tangle", "gauge", "panel", "rack_patch")
        assert all(r.source != "external" for r in registry.list_plugins())
    finally:
        registry.refresh()


def test_demo_arena_page_hides_live_mode_keys_and_campaign_import(
    demo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv(storage.ROOT_ENV, str(demo / "arena"))
    monkeypatch.setenv("KNOTWORK_NETWORK", "deny")
    repo = Path(__file__).resolve().parents[2]
    app = AppTest.from_file(str(repo / "apps" / "inspector.py"), default_timeout=120).run()
    app.switch_page("app_pages/arena.py").run()
    assert not app.exception, app.exception
    assert any(G.BANNER in i.value for i in app.info)
    assert app.session_state["ar_mode"] == "Offline Mock (No Key)"
    mode = next(s for s in app.get("button_group") if "Offline Mock (No Key)" in str(s.proto))
    assert "Live (OpenRouter)" not in str(mode.proto)
    assert not [t for t in app.text_input if "API key" in t.label]
    assert not app.get("file_uploader") and not [e for e in app.expander
                                                 if e.label == "Import campaign.yaml"]  # fmt: skip


def test_local_arena_page_keeps_live_mode_and_campaign_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv(G.ENV, raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv(storage.ROOT_ENV, str(tmp_path / "ws"))
    monkeypatch.setenv("KNOTWORK_NETWORK", "deny")
    repo = Path(__file__).resolve().parents[2]
    app = AppTest.from_file(str(repo / "apps" / "inspector.py"), default_timeout=120).run()
    app.switch_page("app_pages/arena.py").run()
    assert not app.exception, app.exception
    mode = next(s for s in app.get("button_group") if "Offline Mock (No Key)" in str(s.proto))
    assert "Live (OpenRouter)" in str(mode.proto)
    assert [e for e in app.expander if e.label == "Import campaign.yaml"]
