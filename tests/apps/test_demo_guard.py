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
