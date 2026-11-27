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
    bid = app.session_state["item"]["task"]["bundle_id"]
    assert any(d.label == "Download bundle (.zip)" for d in app.get("download_button"))
    next(b for b in app.button if b.label == "Add to Working Set").click().run()
    assert not app.exception, app.exception
    assert bid in ws.ids() and len(ws.rows()) == 5
    assert any(b.label == "In the Working Set" and b.disabled for b in app.button)
    app.button_group(key=f"qa_dec_st_{bid}").set_value("Approve").run()
    app.button_group(key=f"qa_tags_st_{bid}").set_value(["hard-case", "stress"]).run()
    next(b for b in app.button if b.label == "Save review").click().run()
    assert not app.exception, app.exception
    rec = review.load_review_states(ws.review_log)[bid]
    assert rec.status == "approved" and rec.tags == ["hard-case", "stress"]
    next(b for b in app.button if b.label == "Quick test in Arena").click().run()
    assert not app.exception, app.exception
    assert app.session_state["ar_source"] == "Working Set"
    assert app.multiselect(key="ar_ws_pick").value == [bid]
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Certified images"] == "1"  # one frozen item, nothing to render


def test_arena_showcase_and_reward_sandbox(app: AppTest) -> None:
    app.switch_page("app_pages/arena.py").run()
    assert not app.exception, app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Live spend"] == "$0.6727" and metrics["Certified items"] == "59 / 60"
    assert metrics["GRPO frontier cells"] == "8" and metrics["SFT cold-start cells"] == "7"
    board = _frame(app, "$ / correct")
    assert list(board["Model"]) == ["GPT-5.4", "Gemini 2.5 Flash", "Qwen3-VL-235B"]
    assert abs(board.loc[0, "Accuracy"] - 44.6) < 0.1
    assert any("statistical tie" in c.value for c in app.caption)
    group = next(g for g in app.button_group if "Grounded trace" in g.options)
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
