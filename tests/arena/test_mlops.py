"""MLOps layer of the Model Arena: regression pinning, the `knot gate` CI gate, declarative
`campaign.yaml` round trips and the headless `knot arena-run` runner. Frozen fixtures only: no
rendering, no network (mock models)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from arena_fixtures import SECRET, config
from knotwork import cli as K
from knotwork.arena import regression, storage
from knotwork.arena.campaign_yaml import export_campaign_yaml, load_campaign_yaml
from knotwork.arena.schemas import BlockSpec, ItemRecord

MODEL = "openai/gpt-5.4"


def _suite(path: Path, items: list[ItemRecord], **over: object) -> Path:
    for item in items:
        case = regression.case_from_item(item, "fixture", MODEL, "2026-11-27T00:00:00Z")
        regression.pin_case_to_suite(path, case, "fixture regressions")
    if over:
        suite = regression.load_suite(path).model_copy(update=over)
        regression.write_suite(path, suite)
    return path


def test_pin_dedups_by_domain_seed_family(tmp_path: Path, frozen: tuple[Path, list]) -> None:
    _, items = frozen
    path = tmp_path / regression.SUITE_FILE
    first = regression.case_from_item(items[0], "s", MODEL, "t0")
    suite, added = regression.pin_case_to_suite(path, first)
    assert added and len(suite.cases) == 1
    again = first.model_copy(update={"note": "seen twice", "pinned_from_model": "m/other"})
    suite, added = regression.pin_case_to_suite(path, again)
    assert not added and len(suite.cases) == 1 and suite.cases[0].note == "seen twice"
    suite, added = regression.pin_case_to_suite(
        path, regression.case_from_item(items[1], "s", MODEL, "t1")
    )
    assert added and len(suite.cases) == 2
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["version"] == 1 and raw["min_accuracy"] == 0.85
    assert set(raw["cases"][0]) >= {"case_id", "domain", "seed", "preset_or_config",
                                    "question_family", "expected_answer", "source_session",
                                    "pinned_from_model", "pinned_at_utc", "note"}  # fmt: skip
    assert regression.pinned_keys(path) == {c.dedup_key for c in suite.cases}


def test_gate_exit_codes_and_json(tmp_path: Path, frozen: tuple[Path, list]) -> None:
    _, items = frozen
    suite = _suite(tmp_path / "suite.yaml", items)
    runner = CliRunner()
    out = tmp_path / "gate.json"
    res = runner.invoke(K.cli, ["gate", "--suite", str(suite), "--model", MODEL,
                                "--min-accuracy", "0", "--json-out", str(out)])  # fmt: skip
    assert res.exit_code == 0, res.output
    assert "knot gate: PASS" in res.output and res.output.isascii()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["n"] == len(items) and report["exit_code"] == 0 and report["mode"] == "mock"
    assert 0.0 < report["accuracy"] < 1.0  # the deterministic mock model misses some items
    res = runner.invoke(K.cli, ["gate", "--suite", str(suite), "--model", MODEL,
                                "--min-accuracy", "1.0"])  # fmt: skip
    assert res.exit_code == 1 and "FAIL" in res.output
    bad = tmp_path / "bad.yaml"
    bad.write_text("suite_name: x\nversion: 1\ncases: [{case_id: 1}]\n", encoding="utf-8")
    empty = tmp_path / "empty.yaml"
    empty.write_text("suite_name: x\nversion: 1\ncases: []\n", encoding="utf-8")
    for path in (bad, empty, tmp_path / "missing.yaml"):
        res = runner.invoke(K.cli, ["gate", "--suite", str(path), "--model", MODEL])
        assert res.exit_code == 2, (path, res.output)


def test_gate_live_needs_a_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    res = CliRunner().invoke(K.cli, ["gate", "--suite", str(tmp_path / "s.yaml"), "--model",
                                     MODEL, "--mode", "live"])  # fmt: skip
    assert res.exit_code == K.EXIT_SAFETY and "E_API_KEY_MISSING" in res.output


def test_campaign_yaml_round_trip_never_holds_a_key(arena_root: Path) -> None:
    cfg = config(arena_root, run_id="abc123", created_utc="2026-11-27T00:00:00Z", repeats=2)
    text = export_campaign_yaml(cfg)
    assert SECRET not in text and "api_key" not in text and "run_id" not in text
    back = load_campaign_yaml(text)
    assert back == cfg.model_copy(update={"run_id": "", "created_utc": ""})
    with pytest.raises(ValueError, match="credentials"):
        load_campaign_yaml(text + f"api_key: {SECRET}\n")
    with pytest.raises(ValueError, match="secret"):
        load_campaign_yaml(text.replace("notes: ''", f"notes: {SECRET}"))
    with pytest.raises(ValueError, match="blocks"):
        load_campaign_yaml("session_name: x\ndefault_models: [a/b]\nblocks: []\n")
    minimal = load_campaign_yaml(
        "session_name: Tiny run\ndefault_models: [a/b]\n"
        "blocks: [{block_id: b1, domain: tangle, preset: Clean, n: 2, seed: 0}]\n"
    )
    assert minimal.slug == "tiny-run"
    assert Path(minimal.results_dir).parts[-2:] == ("tiny-run", "results")


def test_arena_run_headless_mock(arena_root: Path, frozen: tuple[Path, list]) -> None:
    data, _ = frozen
    one = BlockSpec(block_id="b1", domain="rack_patch", preset="High stress", n=6, seed=0)
    cfg = config(arena_root, data_dir=data, blocks=[one], rpm=1000, max_concurrency=8)
    cfg = cfg.model_copy(update={"default_models": [MODEL]})
    campaign = arena_root / "campaign.yaml"
    campaign.write_text(export_campaign_yaml(cfg), encoding="utf-8")
    results = arena_root / "headless" / "results"
    args = ["arena-run", "--campaign", str(campaign), "--results-dir", str(results), "--report"]
    res = CliRunner().invoke(K.cli, args)
    assert res.exit_code == 0, res.output
    assert "state=done" in res.output and "| ALL" in res.output and res.output.isascii()
    assert len(storage.load_trials(results)) == cfg.total_trials
    for name in ("report.html", "report.md", "report_metrics.json"):
        assert (results / "reports" / name).is_file()
    assert storage.Workspace().get(cfg.slug) is not None
    again = CliRunner().invoke(K.cli, args[:-1])  # resumes: nothing left to ask
    assert again.exit_code == 0 and len(storage.load_trials(results)) == cfg.total_trials
