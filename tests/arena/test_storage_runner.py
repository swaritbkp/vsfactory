"""Sessions index, separate data and results folders, and the resumable runner: results.jsonl,
cooperative pause and stop, resume without duplicates, retry of failed calls only, the spend cap,
and no API key in any file."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from arena_fixtures import SECRET, config, make_dataset
from knotwork.arena import planning, runner, storage
from knotwork.arena.schemas import ItemRecord, TrialRecord


def _lines(results: Path) -> list[dict[str, object]]:
    path = results / storage.RESULTS
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _run(cfg, **kw) -> runner.Runner:  # type: ignore[no-untyped-def]
    results = storage.resolve(cfg.results_dir)
    assert storage.check_folders(cfg, resume=(results / storage.RESULTS).is_file()) == []
    storage.write_session(cfg)
    r = runner.Runner(cfg, results, sleep=lambda s: None, **kw)
    assert r.run() == 0
    return r


def test_slugs_index_rename_archive_keep_folders(arena_root: Path, frozen) -> None:  # type: ignore[no-untyped-def]
    assert storage.slugify("Rack Monochrome Audit!", "20261127") == "rack-monochrome-audit-20261127"
    ws = storage.Workspace()
    assert ws.root == arena_root
    cfg = config(arena_root)
    storage.write_session(cfg)
    ws.register(cfg)
    ws.rename(cfg.slug, "Renamed", "Acme Robotics", ["q4", "rack"])
    entry = ws.get(cfg.slug)
    assert entry and entry.name == "Renamed" and entry.client == "Acme Robotics"
    assert storage.load_config(Path(entry.results_dir)).client == "Acme Robotics"  # type: ignore[union-attr]
    assert ws.unique_slug(cfg.slug) == f"{cfg.slug}-2"
    ws.archive(cfg.slug)
    assert ws.get(cfg.slug) is None and Path(entry.results_dir, "config.json").is_file()
    again = ws.register_existing(Path(entry.results_dir))
    assert again is not None and ws.get(cfg.slug) is not None
    clone = storage.clone_config(cfg, "Rack audit v2", ws)
    assert clone.slug.startswith("rack-audit-v2") and clone.data_mode == "generate"
    assert [b.seed for b in clone.blocks] != [b.seed for b in cfg.blocks]
    assert clone.results_dir != cfg.results_dir and clone.blocks[1].models == cfg.blocks[1].models


def test_atomic_writes_refuse_secrets(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        storage.write_json_atomic(tmp_path / "x.json", {"key": SECRET})
    storage.write_json_atomic(tmp_path / "x.json", {"ok": 1})
    assert json.loads((tmp_path / "x.json").read_text()) == {"ok": 1}
    assert not list(tmp_path.glob(".*.tmp"))


def test_mock_run_writes_one_line_per_trial_and_resumes_without_duplicates(
    arena_root, frozen
) -> None:  # type: ignore[no-untyped-def]
    cfg = config(arena_root)
    r = _run(cfg)
    lines = _lines(storage.resolve(cfg.results_dir))
    assert len(lines) == cfg.total_trials == 6 * 2 + 4 * 1
    assert r.status.state == "done" and r.status.completed_trials == cfg.total_trials
    assert all(line["simulated"] for line in lines)
    _run(cfg)  # a second run is a resume: nothing left to do, nothing repeated
    again = _lines(storage.resolve(cfg.results_dir))
    assert len(again) == len(lines)
    assert len({line["trial_id"] for line in again}) == len(again)


def test_second_session_reuses_frozen_data_into_its_own_results(arena_root, frozen) -> None:  # type: ignore[no-untyped-def]
    data, items = frozen
    first = config(arena_root, "First session")
    second = config(arena_root, "Second session").model_copy(
        update={"default_models": ["meta-llama/llama-4-maverick"]}
    )
    _run(first)
    _run(second)
    a, b = storage.resolve(first.results_dir), storage.resolve(second.results_dir)
    assert a != b and (a / storage.RESULTS).is_file() and (b / storage.RESULTS).is_file()
    assert {t.bundle_id for t in storage.load_trials(a).values()} == {
        t.bundle_id for t in storage.load_trials(b).values()
    }  # the same frozen items
    assert storage.dataset_fingerprint(storage.load_manifest(data)) == storage.dataset_fingerprint(
        items
    )
    clash = first.model_copy(update={"results_dir": str(data)})
    assert any("different folders" in p for p in storage.check_folders(clash, resume=True))
    assert any("already holds results" in p for p in storage.check_folders(first, resume=False))


def test_cooperative_pause_then_resume(arena_root, frozen) -> None:  # type: ignore[no-untyped-def]
    cfg = config(arena_root)
    results = storage.resolve(cfg.results_dir)
    storage.write_session(cfg)
    calls = {"n": 0}

    def sleepy(_: float) -> None:
        calls["n"] += 1
        if calls["n"] == 3:
            storage.send_control(results, "pause")

    paused = runner.Runner(cfg.model_copy(update={"mock_delay_s": 0.01, "max_concurrency": 1}),
                           results, sleep=sleepy)  # fmt: skip
    assert paused.run() == 0
    assert storage.read_status(results).state == "paused"
    done_before = len(_lines(results))
    assert 0 < done_before < cfg.total_trials
    storage.send_control(results, "run")
    _run(cfg)
    lines = _lines(results)
    assert len(lines) == cfg.total_trials and len({x["trial_id"] for x in lines}) == len(lines)


def test_stop_halts_before_any_work(arena_root, frozen) -> None:  # type: ignore[no-untyped-def]
    cfg = config(arena_root)
    results = storage.resolve(cfg.results_dir)
    storage.write_session(cfg)
    storage.send_control(results, "stop")
    assert runner.Runner(cfg, results, sleep=lambda s: None).run() == 0
    assert storage.read_status(results).state == "stopped"
    assert not (results / storage.RESULTS).is_file()


def _transport(fail: set[str], cost: float = 0.001):  # type: ignore[no-untyped-def]
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {SECRET}"
        model = json.loads(request.content)["model"]
        if model in fail:
            return httpx.Response(503, json={"error": "overloaded"})
        body = {"choices": [{"message": {"content": "ANSWER: PP-01"}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 20, "cost": cost}}  # fmt: skip
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handle)


def _all_text(*roots: Path) -> str:
    return "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for root in roots for p in root.rglob("*") if p.is_file() and p.suffix != ".png"
    )  # fmt: skip


def test_live_retry_failed_only_and_no_key_on_disk(arena_root, frozen, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("KNOTWORK_NETWORK", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    cfg = config(arena_root, mode="live", spend_cap_usd=5.0)
    results = storage.resolve(cfg.results_dir)
    storage.Workspace().register(cfg)
    _run(cfg, transport=_transport({"google/gemini-2.5-flash"}))
    trials = storage.load_trials(results)
    failed = [t for t in trials.values() if t.status == "api_error"]
    assert len(failed) == 6 and all(t.attempts == runner.MAX_RETRIES + 1 for t in failed)
    assert storage.read_status(results).failed_trials == 6
    ok = [t for t in trials.values() if t.status == "ok" and t.block_id == "b1"]
    assert ok and all(t.extracted == "01" for t in ok)  # "PP-01" graded as port 01
    before = len(_lines(results))
    _run(cfg, transport=_transport(set()), retry_failed=True)
    after = storage.load_trials(results)
    assert len(_lines(results)) == before + 6  # only the failed calls were asked again
    assert all(t.status == "ok" for t in after.values())
    assert storage.read_status(results).failed_trials == 0
    assert SECRET not in _all_text(results, Path(cfg.data_dir), arena_root)


def test_spend_cap_stops_live_runs(arena_root, frozen, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("KNOTWORK_NETWORK", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    cfg = config(arena_root, mode="live", spend_cap_usd=0.03, max_concurrency=1)
    r = _run(cfg, transport=_transport(set(), cost=0.01))
    assert r.status.state == "budget_stop"
    spent = sum(t.cost_usd for t in storage.load_trials(storage.resolve(cfg.results_dir)).values())
    assert 0 < spent <= cfg.spend_cap_usd
    assert r.status.completed_trials < cfg.total_trials


def test_live_without_key_fails_cleanly(arena_root, frozen, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = config(arena_root, mode="live")
    storage.write_session(cfg)
    results = storage.resolve(cfg.results_dir)
    assert runner.Runner(cfg, results).run() == 2
    assert storage.read_status(results).state == "failed"


def test_rack_answers_are_graded_on_the_port_number() -> None:
    item = ItemRecord(block_id="b", item_index=0, seed=0, domain="rack_patch", preset="Clean",
                      family="T1_TRACE", certified=True, truth="07",
                      answer={"type": "label", "value": "07", "canonical": "07"})  # fmt: skip
    assert runner.grade("It ends at PP-07.\nANSWER: PP-07", item)[0]
    assert runner.grade("ANSWER: port 7", item)[0]
    assert runner.grade("ANSWER: 08", item)[2] == "wrong_port_or_strand"
    assert runner.grade("no idea", item)[2] == "format_violation"
    assert runner.grade("Sorry, I cannot determine this.", item)[2] == "refusal_or_empty"
    assert runner.grade("ANSWER: 07\nANSWER: 08", item)[2] == "format_violation"


# Real detached subprocess, ~25 s on CI: nightly. The in-process pause/resume path stays in tier1
# (`test_cooperative_pause_then_resume`).
@pytest.mark.tier2
def test_background_runner_pauses_with_no_orphan_and_resumes(
    arena_root: Path, tmp_path: Path
) -> None:
    data = arena_root / "data"
    make_dataset(data, {"b1": ("tangle", "Clean", 12)})
    cfg = config(arena_root, "Background", mock_delay_s=0.15, max_concurrency=1, blocks=[
        __import__("knotwork.arena.schemas", fromlist=["BlockSpec"]).BlockSpec(
            block_id="b1", domain="tangle", preset="Clean", n=12, seed=0)])  # fmt: skip
    problems, process = planning.start(cfg, storage.Workspace(), None)
    assert problems == [] and process is not None
    results = storage.resolve(cfg.results_dir)
    deadline = time.monotonic() + 60
    while len(storage.load_trials(results)) < 2 and time.monotonic() < deadline:
        time.sleep(0.1)
    storage.send_control(results, "pause")
    assert process.wait(timeout=60) == 0  # type: ignore[attr-defined]
    status = storage.read_status(results)
    assert status.state == "paused" and not runner.pid_alive(status.pid)
    partial = len(storage.load_trials(results))
    assert 0 < partial < cfg.total_trials
    problems, again = planning.resume(results, storage.Workspace(), None)
    assert problems == [] and again.wait(timeout=90) == 0  # type: ignore[union-attr]
    trials = storage.load_trials(results)
    assert len(trials) == cfg.total_trials == len(_lines(results))
    assert storage.read_status(results).state == "done"
    assert all(isinstance(t, TrialRecord) for t in trials.values())
