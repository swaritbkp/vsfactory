"""The v2.1 journey on the live Streamlit API: Studio hands a certified item to the Working Set
and to Model Arena; Arena opens on the read-only showcase and the reward sandbox; Forensics,
Human Eval and Export open on the Working Set with real numbers and no dead ends."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from knotwork.arena import storage
from knotwork.studio import calibration, review
from knotwork.studio import working_set as W

REPO = Path(__file__).resolve().parents[2]


def _walk(node: Any, out: list[Any]) -> list[Any]:
    for child in getattr(node, "children", {}).values():
        out.append(child)
        _walk(child, out)
    return out


def _frame(app: AppTest, column: str) -> Any:
    """The first dataframe that has `column` (AppTest does not look dataframes up by key)."""
    return next(d.value for d in app.dataframe if column in d.value.columns)


def _text(app: AppTest) -> str:
    return " ".join(str(getattr(e, "proto", "")) for e in _walk(app.main, []))


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    root = tmp_path_factory.mktemp("v21")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(W.ENV, str(root / "working_set"))
        mp.setenv(storage.ROOT_ENV, str(root / "arena"))
        mp.setenv("KNOTWORK_NETWORK", "deny")
        mp.delenv("OPENROUTER_API_KEY", raising=False)
        mp.setattr(calibration, "LOG", root / "human_calibration.jsonl")
        yield root


@pytest.fixture(scope="module")
def app(env: Path) -> AppTest:
    return AppTest.from_file(str(REPO / "apps" / "inspector.py"), default_timeout=120).run()


def test_cold_start_seeds_the_working_set_and_header(app: AppTest, env: Path) -> None:
    assert not app.exception, app.exception
    ws = W.WorkingSet(env / "working_set")
    assert len(ws.rows()) == 4
    text = _text(app)
    assert "Working Set: 4 certified" in text
    assert "detached @ unknown" not in text and "(untracked)" not in text


def test_studio_adds_tags_and_hands_off_to_the_arena(app: AppTest, env: Path) -> None:
    app.switch_page("app_pages/studio.py").run()
    ws = W.WorkingSet(env / "working_set")
    assert any(b.label == "In the Working Set" and b.disabled for b in app.button)
    next(b for b in app.button if b.label == "Generate & Certify Task").click().run()
    assert not app.exception, app.exception
    bid = app.session_state["item"]["task"]["bundle_id"]
    seed = app.session_state["item"]["seed"]
    item_key = f"{bid}_{seed}"
    assert any(d.label == "Download bundle (.zip)" for d in app.get("download_button"))
    next(b for b in app.button if b.label == "Add to Working Set").click().run()
    assert not app.exception, app.exception
    assert bid in ws.ids() and len(ws.rows()) == 5
    assert any(b.label == "In the Working Set" and b.disabled for b in app.button)
    app.button_group(key=f"studio_qa_tags_{item_key}").set_value(["hard-case", "stress"]).run()
    assert review.load_review_states(ws.review_log)[bid].tags == ["hard-case", "stress"]
    app.button_group(key=f"studio_qa_decision_{item_key}").set_value("Approve").run()
    assert not app.exception, app.exception
    rec = review.load_review_states(ws.review_log)[bid]
    assert rec.status == "approved" and rec.tags == ["hard-case", "stress"]
    next(b for b in app.button if b.label == "Generate & Certify Task").click().run()
    assert not app.exception, app.exception
    next_item = app.session_state["item"]
    next_key = f"{next_item['task']['bundle_id']}_{next_item['seed']}"
    assert next_key != item_key
    assert app.button_group(key=f"studio_qa_tags_{next_key}").value == []
    next(b for b in app.button if b.label == "Quick test in Arena").click().run()
    assert not app.exception, app.exception
    assert app.session_state["arena_home_view"] == "New Session"
    assert app.session_state["ar_source"] == "Working Set"
    assert app.multiselect(key="ar_ws_pick").value == [next_item["task"]["bundle_id"]]
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Certified images"] == "1"  # one frozen item, nothing to render


def test_arena_showcase_and_reward_sandbox(app: AppTest) -> None:
    app.switch_page("app_pages/arena.py").run()
    assert not app.exception, app.exception
    app.button_group(key="arena_home_view").set_value("Showcase · 531-Call Audit").run()
    assert not app.exception, app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Live spend"] == "$0.6727" and metrics["Certified items"] == "59 / 60"
    assert metrics["GRPO frontier cells"] == "8" and metrics["SFT cold-start cells"] == "7"
    board = _frame(app, "$ / correct")
    assert list(board["Model"]) == ["GPT-5.4", "Gemini 2.5 Flash", "Qwen3-VL-235B"]
    assert abs(board.loc[0, "Accuracy"] - 44.6) < 0.1
    assert any("statistical tie" in c.value for c in app.caption)
    app.button_group(key="arena_home_view").set_value("Reward Verifier Sandbox").run()
    assert not app.exception, app.exception
    group = next(g for g in app.button_group if "Grounded trace" in g.options)
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Composite"] == "1.000"
    group.set_value("Full-image box cheat [0,0,1000,1000]").run()
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Composite"] == "0.150" and "Reward-hack flag" in _text(app)
    group = next(g for g in app.button_group if "Grounded trace" in g.options)
    group.set_value("Ungrounded lucky guess").run()
    assert {m.label: m.value for m in app.metric}["Composite"] == "0.150"


def test_forensics_opens_on_the_working_set_with_real_kpis(app: AppTest) -> None:
    app.switch_page("app_pages/forensics.py").run()
    assert not app.exception, app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Yield"] != "n/a" and metrics["Fault matrix coverage"] == "14 / 14"
    assert any(b.label == "Generate counterfactual twins" for b in app.button)
    assert "not built yet" not in _text(app)


def test_human_eval_opens_with_an_answerable_item_and_qa_strip(app: AppTest, env: Path) -> None:
    ws = W.WorkingSet(env / "working_set")
    ws.review(ws.ids()[0], "approved")  # each test sets up its own state: xdist may split them
    qa = ws.qa()
    app.switch_page("app_pages/human_eval.py").run()
    assert not app.exception, app.exception
    assert "Working Set" in str(app.selectbox(key="he_dataset").proto)
    assert any(b.label == "Submit" for b in app.button)
    text = _text(app)
    assert f"{qa['approved']} approved" in text and f"{qa['unreviewed']} pending" in text
    v2 = [d.value for d in app.dataframe if list(d.value.columns) == ["Model", "Accuracy"]]
    assert v2 and v2[0]["Model"].tolist()[0] == "GPT-5.4"


def test_export_filters_by_tag_and_offers_a_zip(app: AppTest, env: Path) -> None:
    ws = W.WorkingSet(env / "working_set")
    ws.review(ws.ids()[-1], "approved", ["regression"])
    total = len(ws.rows())
    tagged = sum("regression" in r["tags"] for r in ws.table())
    app.switch_page("app_pages/export.py").run()
    assert not app.exception, app.exception
    table = _frame(app, "QA status")
    assert len(table) == total and "Approved" in set(table["QA status"])
    app.button_group(key="ex_formats").set_value(["manifest"]).run()
    app.button_group(key="ex_tags").set_value(["regression"]).run()
    assert any(f"{tagged} of {total} items match" in c.value for c in app.caption)
    next(b for b in app.button if b.label == "Export dataset").click().run()
    assert not app.exception, app.exception
    assert any(d.label == "Download export (.zip)" for d in app.get("download_button"))
    where = next(c.value for c in app.caption if c.value.startswith("Writes to `"))
    folder = Path(where.split("`")[1])
    folder = folder if folder.is_absolute() else REPO / folder
    manifest = json.loads((folder / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tag_filter"] == ["regression"] and manifest["selected_count"] == tagged
    assert manifest["qa_filter"] == "none" and manifest["candidate_count"] == total


def test_segmented_controls_handle_none_deselection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(W.ENV, str(tmp_path / "working_set"))
    monkeypatch.setenv(storage.ROOT_ENV, str(tmp_path / "arena"))
    monkeypatch.setenv("KNOTWORK_NETWORK", "deny")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(calibration, "LOG", tmp_path / "human_calibration.jsonl")

    fresh = AppTest.from_file(str(REPO / "apps" / "inspector.py"), default_timeout=120).run()
    assert not fresh.exception, fresh.exception

    # 1. Studio View deselection -> falls back to Beauty
    fresh.button_group(key="view").set_value(None).run()
    assert not fresh.exception, fresh.exception
    assert any("What a model sees" in c.value for c in fresh.caption)

    # 2. Studio QA Decision deselection -> falls back to unreviewed without error
    item = fresh.session_state["item"]
    item_key = f"{item['task']['bundle_id']}_{item['seed']}"
    fresh.button_group(key=f"studio_qa_decision_{item_key}").set_value("Approve").run()
    assert not fresh.exception, fresh.exception
    fresh.button_group(key=f"studio_qa_decision_{item_key}").set_value(None).run()
    assert not fresh.exception, fresh.exception
    assert "Unreviewed" in _text(fresh)

    # 3. Arena view deselection -> falls back to default showcase view
    fresh.switch_page("app_pages/arena.py").run()
    fresh.button_group(key="arena_home_view").set_value(None).run()
    assert not fresh.exception, fresh.exception
    assert "v2 4-domain live audit" in _text(fresh)

    # 4. Human Eval Your answer deselection -> clean no-exception render
    fresh.switch_page("app_pages/human_eval.py").run()
    ans_group = next((g for g in fresh.button_group if g.label == "Your answer"), None)
    if ans_group is not None:
        ans_group.set_value(ans_group.options[0]).run()
        assert not fresh.exception, fresh.exception
        next(g for g in fresh.button_group if g.label == "Your answer").set_value(None).run()
        assert not fresh.exception, fresh.exception


def test_studio_qa_rapid_clicks_and_seed_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(W.ENV, str(tmp_path / "working_set"))
    monkeypatch.setenv(storage.ROOT_ENV, str(tmp_path / "arena"))
    monkeypatch.setenv("KNOTWORK_NETWORK", "deny")

    fresh = AppTest.from_file(str(REPO / "apps" / "inspector.py"), default_timeout=120).run()
    ws = W.WorkingSet(tmp_path / "working_set")
    next(b for b in fresh.button if b.label == "Generate & Certify Task").click().run()
    assert not fresh.exception, fresh.exception

    item_a = fresh.session_state["item"]
    bid_a = item_a["task"]["bundle_id"]
    seed_a = int(item_a["seed"])
    key_a = f"{bid_a}_{seed_a}"

    fresh.button_group(key=f"studio_qa_decision_{key_a}").set_value("Approve").run()
    assert not fresh.exception, fresh.exception
    fresh.button_group(key=f"studio_qa_decision_{key_a}").set_value("Flag").run()
    assert not fresh.exception, fresh.exception
    fresh.button_group(key=f"studio_qa_tags_{key_a}").set_value(["hard-case", "monochrome"]).run()
    assert not fresh.exception, fresh.exception

    next(b for b in fresh.button if b.label == "Generate & Certify Task").click().run()
    assert not fresh.exception, fresh.exception
    item_b = fresh.session_state["item"]
    bid_b = item_b["task"]["bundle_id"]
    key_b = f"{bid_b}_{item_b['seed']}"
    assert bid_b != bid_a
    assert fresh.button_group(key=f"studio_qa_tags_{key_b}").value == []
    assert fresh.button_group(key=f"studio_qa_decision_{key_b}").value is None

    # Switch back to seed_a and re-generate: latest decision + tags persist without duplicates
    fresh.number_input(key="seed").set_value(seed_a).run()
    next(b for b in fresh.button if b.label == "Generate & Certify Task").click().run()
    assert not fresh.exception, fresh.exception
    assert fresh.session_state["item"]["task"]["bundle_id"] == bid_a
    assert ws.ids().count(bid_a) == 1
    rec = review.load_review_states(ws.review_log)[bid_a]
    assert rec.status == "flagged" and rec.tags == ["hard-case", "monochrome"]


def test_studio_cold_start_and_domain_mismatch_cue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from knotwork.studio import factory

    monkeypatch.setenv(W.ENV, str(tmp_path / "working_set"))
    monkeypatch.setenv(storage.ROOT_ENV, str(tmp_path / "arena"))
    monkeypatch.setenv("KNOTWORK_NETWORK", "deny")

    def forbid_generate(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("cold start must load Starter Working Set without generating")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(factory, "generate_one", forbid_generate)
        fresh = AppTest.from_file(str(REPO / "apps" / "inspector.py"), default_timeout=120).run()

    assert not fresh.exception, fresh.exception
    item = fresh.session_state["item"]
    assert item["plugin"] == "tangle" and item["family"] == "T1_TRACE"
    assert fresh.session_state["plugin"] == "tangle" and fresh.session_state["seed"] == item["seed"]
    assert any(b.label == "In the Working Set" and b.disabled for b in fresh.button)

    # Selecting Rack patch-bay without Generate keeps preview and surfaces mismatch cue
    fresh.button_group(key="plugin").set_value("rack_patch").run()
    assert not fresh.exception, fresh.exception
    assert fresh.session_state["item"]["plugin"] == "tangle"
    assert any("Rack patch-bay" in i.value and "Tangle" in i.value for i in fresh.info)

    # Switching back to Tangle clears the cue immediately
    fresh.button_group(key="plugin").set_value("tangle").run()
    assert not any("currently previewing" in i.value for i in fresh.info)

    # Switching to Rack patch-bay and generating renders Rack patch-bay and clears the cue
    fresh.button_group(key="plugin").set_value("rack_patch").run()
    assert any("currently previewing" in i.value for i in fresh.info)
    next(b for b in fresh.button if b.label == "Generate & Certify Task").click().run()
    assert not fresh.exception, fresh.exception
    assert fresh.session_state["item"]["plugin"] == "rack_patch"
    assert not any("currently previewing" in i.value for i in fresh.info)


def test_batch_factory_all_domains_and_no_mismatch_cue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from knotwork.pipeline.generate import run_generate
    from knotwork.studio import factory

    monkeypatch.setenv(W.ENV, str(tmp_path / "working_set"))
    monkeypatch.setenv(storage.ROOT_ENV, str(tmp_path / "arena"))
    monkeypatch.setenv("KNOTWORK_NETWORK", "deny")

    fresh = AppTest.from_file(str(REPO / "apps" / "inspector.py"), default_timeout=120).run()
    fresh.button_group(key="studio_mode").set_value("Batch Factory").run()
    assert not fresh.exception, fresh.exception

    for plugin in ("tangle", "gauge", "panel", "rack_patch"):
        fresh.button_group(key="plugin").set_value(plugin).run()
        assert not fresh.exception, fresh.exception
        assert any(b.label == "Launch batch build" for b in fresh.button), plugin
        assert not any("Press **Generate & Certify Task**" in i.value for i in fresh.info), plugin

    # Real batch generation on rack_patch certifies base + twins cleanly
    out = tmp_path / "rack_batch"
    rc = run_generate(
        "rack_patch", 1, out, 7, "offline", REPO / "configs" / "default_profile.yaml",
        "dev", True, None, 1,
    )  # fmt: skip
    assert rc == 0
    rows = [json.loads(line) for line in (out / "index.jsonl").read_text("utf-8").splitlines()]
    assert {r["role"] for r in rows} == {"base", "causal", "nuisance"}

    # Defensive fallback guard when a custom/future plugin lacks a profile section
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(factory, "batch_ready", lambda _p: False)
        fresh.run()
        assert not fresh.exception, fresh.exception
        next(b for b in fresh.button if b.label == "Switch to Single Task").click().run()
        assert not fresh.exception, fresh.exception
        assert fresh.session_state["studio_mode"] == "Single Task"


def test_arena_session_view_escape_and_back_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from arena_fixtures import config, make_dataset
    from knotwork.arena import runner
    from knotwork.arena.schemas import BlockSpec

    root = tmp_path / "arena"
    monkeypatch.setenv(W.ENV, str(tmp_path / "working_set"))
    monkeypatch.setenv(storage.ROOT_ENV, str(root))
    monkeypatch.setenv("KNOTWORK_NETWORK", "deny")
    make_dataset(root / "data", {"b1": ("tangle", "Clean", 2)})
    cfg = config(root, blocks=[BlockSpec(block_id="b1", domain="tangle", preset="Clean", n=2,
                                         seed=0)])  # fmt: skip
    storage.write_session(cfg)
    runner.Runner(cfg, storage.resolve(cfg.results_dir), sleep=lambda _s: None).run()
    storage.Workspace().register(cfg)

    fresh = AppTest.from_file(str(REPO / "apps" / "inspector.py"), default_timeout=120).run()
    fresh.switch_page("app_pages/arena.py").run()
    next(b for b in fresh.sidebar.button if b.label == cfg.session_name).click().run()
    assert fresh.session_state["ar_selected"] == cfg.slug
    next(b for b in fresh.main.button if b.label == "Back to Arena home").click().run()
    assert not fresh.exception, fresh.exception
    assert fresh.session_state["ar_selected"] is None
    assert fresh.session_state["arena_home_view"] == "Showcase · 531-Call Audit"

    # Re-enter session view, then escape via Hero Stepper "2 · Arena"
    next(b for b in fresh.sidebar.button if b.label == cfg.session_name).click().run()
    assert fresh.session_state["ar_selected"] == cfg.slug
    fresh.button(key="hero_nav_model_arena").click().run()
    assert not fresh.exception, fresh.exception
    assert fresh.session_state["ar_selected"] is None
    assert fresh.session_state["arena_home_view"] == "Showcase · 531-Call Audit"


def test_public_demo_renders_zero_tmp_strings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from knotwork.studio import demo_guard as G

    monkeypatch.setenv(G.ENV, "1")
    monkeypatch.delenv(G.BYOK_ENV, raising=False)
    monkeypatch.setenv("KNOTWORK_NETWORK", "deny")
    monkeypatch.delenv(W.ENV, raising=False)
    monkeypatch.delenv(storage.ROOT_ENV, raising=False)
    monkeypatch.setattr(G, "SANDBOX_ROOT", tmp_path / "knotwork_demo")

    try:
        fresh = AppTest.from_file(str(REPO / "apps" / "inspector.py"), default_timeout=120).run()
        assert not fresh.exception, fresh.exception
        assert "Offline Mock Sandbox · 10-item cap · Ephemeral" in _text(fresh)
        assert any(b.label == "Reset my sandbox" for b in fresh.button)
        assert "/tmp" not in _text(fresh)

        fresh.button_group(key="studio_mode").set_value("Batch Factory").run()
        assert "/tmp" not in _text(fresh)
        assert not any(t.label == "Output folder" for t in fresh.text_input)

        fresh.switch_page("app_pages/arena.py").run()
        fresh.button_group(key="arena_home_view").set_value("New Session").run()
        assert "/tmp" not in _text(fresh)
        assert not any("data_dir" in t.label or "results_dir" in t.label for t in fresh.text_input)
        assert all("/tmp" not in str(t.value or "") for t in fresh.text_input)

        for page in ("forensics", "human_eval", "export"):
            fresh.switch_page(f"app_pages/{page}.py").run()
            assert not fresh.exception, fresh.exception
            assert "/tmp" not in _text(fresh)
            assert all("/tmp" not in str(t.value or "") for t in fresh.text_input)
    finally:
        G.set_demo_session_id(None)
