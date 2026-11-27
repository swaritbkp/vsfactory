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
    monkeypatch.delenv(G.BYOK_ENV, raising=False)  # the default demo: Offline Mock only
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
    assert "Live · OpenRouter" in str(mode.proto)
    assert [e for e in app.expander if e.label == "Import campaign.yaml"]


# ------------------------------------------------------------------ opt-in BYOK tier
LIVE_MODES = {"Offline Mock (No Key)": "mock", "Live · OpenRouter": "openrouter",
              "Live · Custom endpoint": "openai_compat"}  # fmt: skip


@pytest.fixture
def byok(demo: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(G.BYOK_ENV, "1")
    return demo


@pytest.mark.parametrize("value", [None, "0", "true", ""])
def test_byok_needs_both_switches(
    demo: Path, monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is not None:
        monkeypatch.setenv(G.BYOK_ENV, value)
    assert not G.byok_allowed() and G.banner() == G.BANNER
    assert G.execution_modes(LIVE_MODES) == {"Offline Mock (No Key)": "mock"}
    monkeypatch.setenv(G.BYOK_ENV, "1")
    monkeypatch.delenv(G.ENV)
    assert not G.byok_allowed()  # BYOK alone (no public demo) changes nothing


def _small(demo: Path, **extra: object) -> object:
    blocks = extra.pop("blocks", None) or [
        BlockSpec(block_id="b1", domain="tangle", preset="Clean", n=5, seed=0)
    ]
    return config(demo, data_dir=demo / "data", mode="live", blocks=blocks, repeats=1,
                  spend_cap_usd=0.5, **extra)  # fmt: skip


def test_byok_caps_are_enforced_server_side(byok: Path) -> None:
    assert G.byok_allowed() and G.banner() == G.BYOK_BANNER
    assert G.execution_modes(LIVE_MODES) == LIVE_MODES
    ok = _small(byok)
    assert G.offline_only(ok).mode == "live" and G.config_problems(ok) == []
    two = _small(byok, blocks=[BlockSpec(block_id=f"b{i}", domain="tangle", preset="Clean",
                                         n=2, seed=0) for i in range(2)])  # fmt: skip
    six = _small(byok, blocks=[BlockSpec(block_id="b1", domain="tangle", preset="Clean", n=6,
                                         seed=0)])  # fmt: skip
    found = " ".join(G.config_problems(two) + G.config_problems(six))
    assert "1 block" in found and "at most 5 items" in found
    found = " ".join(G.config_problems(config(byok, data_dir=byok / "data", mode="live",
                                              repeats=2, spend_cap_usd=0.75, blocks=[
        BlockSpec(block_id="b1", domain="tangle", preset="Clean", n=1, seed=0)])))  # fmt: skip
    assert "k = 1" in found and "$0.50" in found
    mock = config(byok, data_dir=byok / "data", blocks=[
        BlockSpec(block_id="b1", domain="tangle", preset="Clean", n=10, seed=0)])  # fmt: skip
    assert G.config_problems(mock) == []  # Offline Mock keeps the ordinary demo caps


def test_byok_custom_endpoint_must_pass_the_ssrf_guard(byok: Path) -> None:
    local = _small(byok, provider="openai_compat", base_url="http://localhost:8000/v1")
    assert any("Endpoint:" in p and "https" in p for p in G.config_problems(local))
    meta = _small(byok, provider="openai_compat", base_url="https://169.254.169.254/v1")
    assert any("Endpoint:" in p for p in G.config_problems(meta))


def test_byok_never_reads_the_server_key(byok: Path, tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"OPENROUTER_API_KEY={KEY}\n", encoding="utf-8")
    assert G.server_api_key() is None and keys.detect(dotenv) == (None, "")


def test_byok_resume_is_capped_not_refused(byok: Path) -> None:
    from knotwork.arena import planning

    big = config(byok, data_dir=byok / "data", mode="live", repeats=3)
    storage.write_session(big)
    problems, process = planning.resume(storage.resolve(big.results_dir),
                                        storage.Workspace(byok / "ws"), KEY)  # fmt: skip
    assert process is None and any("Small Live BYOK Test" in p for p in problems)


def test_byok_arena_page_offers_live_modes_with_session_only_keys(
    byok: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv(storage.ROOT_ENV, str(byok / "arena"))
    monkeypatch.setenv("KNOTWORK_NETWORK", "deny")
    repo = Path(__file__).resolve().parents[2]
    app = AppTest.from_file(str(repo / "apps" / "inspector.py"), default_timeout=120).run()
    app.switch_page("app_pages/arena.py").run()
    assert not app.exception, app.exception
    assert any(G.BYOK_BANNER in i.value for i in app.info)
    mode = next(s for s in app.get("button_group") if "Offline Mock (No Key)" in str(s.proto))
    assert "Live · OpenRouter" in str(mode.proto) and "Live · Custom endpoint" in str(mode.proto)
    assert not app.get("file_uploader")  # campaign import stays off in every demo tier
    mode.set_value("Live · OpenRouter").run()
    assert not app.exception, app.exception
    assert any(G.PRIVACY_NOTE in i.value for i in app.info)
    assert [t for t in app.text_input if t.label == "OpenRouter API key"]
    assert not [b for b in app.get("badge") if ".env" in str(b.proto)]  # no server key shown
    assert any("paste your key above" in w.value for w in app.warning)  # server key not used
    files = " ".join(p.read_text("utf-8", "ignore") for p in byok.rglob("*") if p.is_file()
                     and p.suffix in (".json", ".jsonl", ".yaml", ".log"))  # fmt: skip
    assert KEY not in files
