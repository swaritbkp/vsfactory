"""Statistics are exact, the catalog never needs the network, and configs validate at scale."""

from __future__ import annotations

import math

import httpx
import pytest
from pydantic import ValidationError

from arena_fixtures import SECRET, config
from knotwork.arena import catalog, planning, report
from knotwork.arena.schemas import BlockSpec, SessionConfig, find_secret


def test_wilson_matches_closed_form_values() -> None:
    lo, hi = report.wilson(8, 10)
    assert (round(lo, 4), round(hi, 4)) == (0.4902, 0.9433)
    lo, hi = report.wilson(0, 10)
    assert lo == 0.0 and round(hi, 4) == 0.2775
    assert report.wilson(0, 0) == (0.0, 1.0)
    lo, hi = report.wilson(50, 100)
    assert math.isclose(lo + hi, 1.0, abs_tol=1e-12)  # symmetric around one half


def test_fisher_and_z_tests_match_reference_values() -> None:
    # R: fisher.test(matrix(c(8, 2, 1, 5), 2)) -> p = 0.03497
    assert round(report.fisher_exact(8, 10, 1, 6), 5) == 0.03497
    assert report.fisher_exact(5, 10, 5, 10) == pytest.approx(1.0)
    z, p = report.two_proportion_z(80, 100, 60, 100)
    assert round(z, 3) == 3.086 and round(p, 4) == 0.0020
    assert report.two_proportion_z(0, 0, 1, 2) == (0.0, 1.0)


def test_percentiles_and_pareto_front() -> None:
    values = [float(v) for v in range(1, 11)]
    assert report.percentile(values, 50) == 5.5
    assert report.percentile(values, 95) == pytest.approx(9.55)
    assert report.percentile([], 50) is None
    front = report.pareto_front([(1.0, 0.5, "a"), (2.0, 0.4, "b"), (3.0, 0.9, "c")])
    assert [p[2] for p in front] == ["a", "c"]


def test_catalog_is_curated_offline_and_live_falls_back_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KNOTWORK_NETWORK", "deny")
    assert catalog.fetch().source == "curated"

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    cat = catalog.fetch(live=True, transport=httpx.MockTransport(boom))
    assert cat.source == "curated" and "unavailable" in cat.note
    assert set(catalog.DEFAULT_MODELS) <= set(cat.ids())
    assert all(m.free for m in cat.models if m.id in cat.ids(free_only=True))


def test_live_catalog_parses_vision_models_and_prices() -> None:
    payload = {"data": [
        {"id": "x/vision", "name": "X", "pricing": {"prompt": "0.000001", "completion": "0.000004"},
         "architecture": {"input_modalities": ["text", "image"]}, "context_length": 1000},
        {"id": "x/text", "pricing": {"prompt": "0", "completion": "0"},
         "architecture": {"input_modalities": ["text"]}},
    ]}  # fmt: skip
    cat = catalog.fetch(
        live=True, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    )
    assert cat.source == "live" and cat.ids() == ["x/vision"]
    assert cat.get("x/vision:nitro") is not None  # routing suffixes map to their base model
    assert catalog.prices("x/vision", cat) == pytest.approx((1e-6, 4e-6))
    assert catalog.prices("anything:free") == (0.0, 0.0)


def test_thousand_item_blocks_and_per_block_models(tmp_path) -> None:  # type: ignore[no-untyped-def]
    blocks = [
        BlockSpec(block_id="b1", domain="rack_patch", preset="High stress", n=1000, seed=0),
        BlockSpec(block_id="b2", domain="panel", preset="Hard logic", n=1000, seed=7,
                  models=["qwen/qwen3-vl-235b-a22b-instruct", "custom/model:nitro"]),
    ]  # fmt: skip
    cfg = config(tmp_path, data_mode="generate", blocks=blocks)
    assert cfg.total_items == 2000
    assert cfg.total_trials == 1000 * 2 + 1000 * 2
    assert cfg.models_for(cfg.blocks[1]) == [
        "qwen/qwen3-vl-235b-a22b-instruct",
        "custom/model:nitro",
    ]
    pf = planning.preflight(cfg)
    assert pf.images_to_generate == 2000 and pf.trials == 4000 and pf.max_cost_usd > 0
    live = cfg.model_copy(update={"mode": "live", "spend_cap_usd": 0.5})
    assert planning.preflight(live).over_cap


def test_configs_refuse_secrets_and_duplicate_blocks(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValidationError):
        config(tmp_path, notes=f"key {SECRET}")
    with pytest.raises(ValidationError):
        config(
            tmp_path,
            blocks=[BlockSpec(block_id="b1", domain="gauge", preset="Clean", n=1, seed=0)] * 2,
        )
    assert find_secret({"a": ["x", {"b": f"Bearer {SECRET}"}]})
    with pytest.raises(ValidationError):
        BlockSpec(block_id="b1", domain="tangle", preset="Clean", n=0, seed=0)
    assert isinstance(config(tmp_path), SessionConfig)
