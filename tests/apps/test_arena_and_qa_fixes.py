"""Model Arena loads offline with its sessions rail, and the four video-QA fixes hold: instant fault
proofs, the Human Eval scoreboard above the fold with the verdict under the picture, the batch CLI
command collapsed, and export paths relative to the output folder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from arena_fixtures import config, make_dataset
from knotwork.arena import runner, storage
from knotwork.studio import calibration, fault_proofs, present

REPO = Path(__file__).resolve().parents[2]


def _walk(node: Any, out: list[Any]) -> list[Any]:
    for child in getattr(node, "children", {}).values():
        out.append(child)
        _walk(child, out)
    return out


def _app(monkeypatch: pytest.MonkeyPatch, root: Path) -> AppTest:
    monkeypatch.setenv(storage.ROOT_ENV, str(root))
    monkeypatch.setenv("KNOTWORK_NETWORK", "deny")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return AppTest.from_file(str(REPO / "apps" / "inspector.py"), default_timeout=120).run()


# ------------------------------------------------------------------ Model Arena
def test_model_arena_opens_offline_with_sessions_rail(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    app = _app(monkeypatch, tmp_path / "ws")
    app.switch_page("app_pages/arena.py").run()
    assert not app.exception, app.exception
    assert app.session_state["arena_home_view"] == "Showcase · 531-Call Audit"
    assert [b.label for b in app.sidebar.button][0] == "v2 4-domain live audit"
    assert any("No saved sessions yet" in c.value for c in app.sidebar.caption)
    app.button_group(key="arena_home_view").set_value("New Session").run()
    assert not app.exception, app.exception
    labels = {m.label: m.value for m in app.metric}
    assert labels["Images to generate"] == "20" and labels["Model trials"] == "60"
    assert "Cost if run live" in labels  # mock mode by default: nothing is charged
    domains = [s for s in app.selectbox if s.label == "Domain"]
    assert [d.value for d in domains] == ["rack_patch", "panel"]
    assert any("BilotaAI" in c.value for c in app.caption)
    launch = next(b for b in app.button if b.label == "Launch campaign")
    assert not launch.disabled


def test_model_arena_session_view_and_sidebar_card(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    root = tmp_path / "ws"
    monkeypatch.setenv(storage.ROOT_ENV, str(root))
    make_dataset(
        root / "data", {"b1": ("rack_patch", "High stress", 6), "b2": ("panel", "Hard logic", 4)}
    )
    cfg = config(root, client="Acme Robotics")
    storage.write_session(cfg)
    runner.Runner(cfg, storage.resolve(cfg.results_dir), sleep=lambda s: None).run()
    storage.Workspace().register(cfg)
    app = _app(monkeypatch, root)
    app.switch_page("app_pages/arena.py").run()
    open_button = next(b for b in app.sidebar.button if b.label == cfg.session_name)
    open_button.click().run()
    assert not app.exception, app.exception
    assert app.session_state["ar_selected"] == cfg.slug
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Completed trials"] == f"{cfg.total_trials} / {cfg.total_trials}"
    assert metrics["Certification yield"] == "100.0%"
    assert any(e.label == "Copy headless CLI command" for e in app.expander)
    assert len(app.dataframe) >= 1 and "Accuracy" in app.dataframe[0].value.columns
    app.sidebar.text_input[0].set_value("no-such-session").run()
    assert any("No session matches" in c.value for c in app.sidebar.caption)


# ------------------------------------------------------------------ Pillar 0 fixes
def test_fault_proofs_open_instantly_from_the_committed_pair(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    sample = fault_proofs.load_sample()
    assert sample is not None
    fault, seed, sides = sample
    assert [s.label for s in sides] == list(fault_proofs.LABELS)
    assert sides[0].verdict == "CERTIFIED" and fault_proofs.caught(fault, sides[1])
    app = _app(monkeypatch, tmp_path / "ws")
    app.switch_page("app_pages/forensics.py").run()
    app.session_state["fx_section"] = "Fault proofs"
    app.run()
    assert not app.exception, app.exception
    badges = " ".join(str(getattr(e, "proto", "")) for e in _walk(app.main, []))
    assert f"{fault_proofs.expected_code(fault)} raised" in badges
    assert not any("Inject and certify" in str(s.label) for s in app.status)  # nothing was built


def test_human_eval_scoreboard_first_and_verdict_under_the_picture(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(calibration, "LOG", tmp_path / "human_calibration.jsonl")
    app = _app(monkeypatch, tmp_path / "ws")
    app.switch_page("app_pages/human_eval.py").run()
    board = app.dataframe[0].value
    assert list(board["Model"])[:3] == ["GPT-5.4", "Gemini 2.5 Flash", "Qwen3-VL-235B"]
    assert app.metric[0].label == "Your session"
    submit = next(b for b in app.button if b.label == "Submit")
    assert submit.proto.type == "primary"
    answer = next((t for t in app.text_input if t.label == "Your answer"), None)
    group = next((g for g in app.button_group if g.label == "Your answer"), None)
    if answer is not None:
        answer.set_value("1")
    elif group is not None:  # up to 8 options: buttons
        group.set_value(group.options[0])
    else:  # more options (a rack patch-bay item): a select box
        box = next(s for s in app.selectbox if s.label == "Your answer")
        box.set_value(box.options[0])
    next(b for b in app.button if b.label == "Submit").click().run()
    assert not app.exception, app.exception
    order = [(type(e).__name__, getattr(e, "label", None)) for e in _walk(app.main, [])]
    image = order.index(("Image", None))
    verdict = order.index(("Metric", "Your answer"))
    nxt = order.index(("Button", "Next item"))
    assert image < verdict < nxt  # the verdict sits where Submit was, under the picture


def test_batch_cli_command_is_collapsed(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    app = _app(monkeypatch, tmp_path / "ws")
    assert not app.exception, app.exception
    app.button_group(key="studio_mode").set_value("Batch Factory").run()
    assert not app.exception, app.exception
    cli = next(e for e in app.expander if e.label == "Show CLI command")
    assert any("uv run knot generate" in c.value for c in cli.code)
    assert not cli.proto.expanded


def test_export_paths_are_relative_to_the_output_folder(tmp_path: Path) -> None:
    out = tmp_path / "export" / "tangle"
    out.mkdir(parents=True)
    (out / "dataset_manifest.json").write_text("{}", encoding="utf-8")
    data = {"artifacts": {"harbor/index.json": {"bytes": 10},
                          "parquet/train-00000.parquet": {"bytes": 2048}}}  # fmt: skip
    rows = present.export_rows(out, data)
    assert [r["Path"] for r in rows] == [
        "harbor/index.json", "parquet/train-00000.parquet", "dataset_manifest.json",
    ]  # fmt: skip
    assert all(str(tmp_path) not in r["Path"] for r in rows)
    json.dumps(rows)


def test_arena_temperature_control_and_post_training_tab(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from posttrain_fixtures import session

    root = tmp_path / "ws"
    monkeypatch.setenv(storage.ROOT_ENV, str(root))
    cfg = session(root)
    storage.Workspace().register(cfg)
    app = _app(monkeypatch, root)
    app.switch_page("app_pages/arena.py").run()
    app.button_group(key="arena_home_view").set_value("New Session").run()
    temp = next(s for s in app.slider if s.label == "Temperature")
    assert temp.value == 0.0 and not app.exception
    next(b for b in app.sidebar.button if b.label == cfg.session_name).click().run()
    assert not app.exception, app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["GRPO frontier"] == "2" and metrics["SFT cold-start"] == "1"
    assert metrics["Saturated"] == "1" and metrics["Train seeds from"] == "10,000"
    assert float(metrics["Composite"]) >= 0.99 and metrics["Grounding IoU"] == "1.00"
    examples = next(g for g in app.button_group if "Lucky guess (no boxes)" in g.options)
    examples.set_value("Lucky guess (no boxes)").run()
    assert "Reward-hack flag" in " ".join(str(getattr(e, "proto", "")) for e in _walk(app.main, []))
    next(b for b in app.button if b.label == "Export post-training pack").click().run()
    assert not app.exception, app.exception
    assert (storage.resolve(cfg.results_dir) / "posttrain_pack.zip").is_file()
    assert any(d.label == "Download posttrain_pack.zip" for d in app.get("download_button"))
