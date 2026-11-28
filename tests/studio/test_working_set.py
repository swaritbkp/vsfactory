"""The Working Set: starter bootstrap, add / de-duplication, QA tags in review_log.jsonl, atomic
writes, per-process isolation under pytest-xdist, and the in-memory download archives."""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from knotwork.arena import storage
from knotwork.studio import present, review
from knotwork.studio import working_set as W

STARTER_DOMAINS = {"tangle", "gauge", "panel", "rack_patch"}


@pytest.fixture
def ws(tmp_path: Path) -> W.WorkingSet:
    return W.WorkingSet(tmp_path / "ws")


def test_committed_starter_is_four_certified_domains() -> None:
    rows = W.WorkingSet(W.STARTER).rows()
    assert {r["plugin"] for r in rows} == STARTER_DOMAINS and len(rows) == 4
    for row in rows:
        task = json.loads((W.STARTER / row["dir"] / "task.json").read_text(encoding="utf-8"))
        assert W.certified(task) and task["bundle_id"] == row["bundle_id"]
        assert row["answer_canonical"] == task["answer"]["canonical"]
    meta = json.loads((W.STARTER / W.META).read_text(encoding="utf-8"))
    assert meta["certified"] == 4 and meta["candidates"] >= 4
    size = sum(p.stat().st_size for p in W.STARTER.rglob("*") if p.is_file())
    assert size < 2_500_000  # small enough to commit and to copy on a cold start


@pytest.mark.tier2
def test_committed_starter_re_verifies_with_the_live_kernel(tmp_path: Path) -> None:
    """File-only re-verification (no plugin code): the committed copy matches today's kernel."""
    from knotwork.pipeline.verify import run_verify

    assert run_verify(W.STARTER, tmp_path / "verify_report.json") == 0


def test_cold_start_seeds_fast_and_only_once(ws: W.WorkingSet) -> None:
    assert not ws.valid()
    t0 = time.perf_counter()
    assert ws.ensure() is True
    assert time.perf_counter() - t0 < 2.5
    assert ws.valid() and len(ws.rows()) == 4
    assert ws.ensure() is False  # already valid: never re-seeded over the user's work
    assert ws.kpis()["yield_rate"] == pytest.approx(1.0) and ws.qa()["unreviewed"] == 4
    assert present.run_kpis(ws.root)["source"] == W.META
    assert not list(ws.root.rglob("*.tmp"))  # atomic writes leave no temporary files behind


def test_add_deduplicates_and_refuses_uncertified(ws: W.WorkingSet, tmp_path: Path) -> None:
    ws.ensure()
    source = W.WorkingSet(W.STARTER)
    first = source.rows()[0]
    again = ws.add(W.STARTER / first["dir"])
    assert not again.added and again.reason.startswith("Already")
    new_root = tmp_path / "other"
    other = W.WorkingSet(new_root)
    added = other.add(W.STARTER / first["dir"], "studio", seed=7, attempts=3)
    assert added.added and added.count == 1 and other.ids() == [first["bundle_id"]]
    assert other.rows()[0]["source"] == "studio" and other.rows()[0]["seed"] == 7
    assert other.meta()["candidates"] == 3 and other.kpis()["yield_rate"] == pytest.approx(1 / 3)
    bad = tmp_path / "bad"
    shutil.copytree(W.STARTER / first["dir"], bad)
    task = json.loads((bad / "task.json").read_text(encoding="utf-8"))
    task["certificate"]["verdict"] = "QUARANTINED"
    (bad / "task.json").write_text(json.dumps(task), encoding="utf-8")
    refused = other.add(bad)
    assert (
        not refused.added and "certified" in refused.reason and other.ids() == [first["bundle_id"]]
    )


def test_qa_tags_are_appended_and_old_logs_still_load(ws: W.WorkingSet) -> None:
    ws.ensure()
    a, b = ws.ids()[:2]
    ws.review(a, "approved", ["Hard Case", "stress", "stress"], note="  clean   trace ")
    ws.review(b, "flagged", ["needs-review"])
    ws.review(a, "approved", ["regression"])  # a later line supersedes the earlier one
    states = review.load_review_states(ws.review_log)
    assert review.tags_of(states, a) == ["regression"] and states[b].tags == ["needs-review"]
    assert ws.qa() == {"unreviewed": 2, "approved": 1, "flagged": 1}
    lines = ws.review_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["tags"] == ["hard-case", "stress"]
    assert json.loads(lines[0])["note"] == "clean trace"
    with ws.review_log.open("a", encoding="utf-8") as handle:  # a v2.0 line: no tags field
        handle.write(json.dumps({"item_id": b, "status": "approved", "domain": "gauge"}) + "\n")
    assert review.load_review_states(ws.review_log)[b].tags == []
    table = {r["bundle_id"]: r for r in ws.table()}
    assert table[a]["status"] == "approved" and table[a]["tags"] == ["regression"]


def test_concurrent_adds_keep_one_row_per_bundle(tmp_path: Path) -> None:
    ws = W.WorkingSet(tmp_path / "ws")
    bundles = [W.STARTER / r["dir"] for r in W.WorkingSet(W.STARTER).rows()]
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda b: ws.add(b), bundles * 4))
    assert sorted(ws.ids()) == sorted(r["bundle_id"] for r in W.WorkingSet(W.STARTER).rows())
    assert all((ws.root / r["dir"] / "task.json").is_file() for r in ws.rows())


def test_default_root_is_isolated_per_process_under_pytest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(W.ENV, raising=False)
    root = W.default_root()
    assert root == Path(tempfile.gettempdir()) / f"knotwork-ws-{os.getpid()}"
    assert "artifacts" not in root.parts  # never the repository's shared folder under xdist
    monkeypatch.setenv(W.ENV, str(tmp_path / "explicit"))
    assert W.default_root() == tmp_path / "explicit"


def test_working_set_reads_as_a_studio_batch_for_the_arena(ws: W.WorkingSet) -> None:
    ws.ensure()
    items = storage.studio_batch_items(ws.root)
    assert {i.domain for i in items} == STARTER_DOMAINS and all(i.certified for i in items)
    assert all((ws.root / i.dir / "beauty.png").is_file() for i in items)


def test_bundle_and_folder_zips_are_built_in_memory(ws: W.WorkingSet) -> None:
    ws.ensure()
    bundle = ws.root / ws.rows()[0]["dir"]
    line = W.vcot_line(bundle)
    assert line is not None and json.loads(line)["response"].startswith("<think>")
    data = W.bundle_zip(bundle, b"\x89PNG fake", line)
    names = set(zipfile.ZipFile(io.BytesIO(data)).namelist())
    assert {"beauty.png", "witness_overlay.png", "manifest.json", "vcot_sft.jsonl"} <= names
    archive = zipfile.ZipFile(io.BytesIO(W.folder_zip(ws.root, "ws")))
    assert "ws/index.jsonl" in archive.namelist() and W.folder_zip(ws.root) == W.folder_zip(
        ws.root
    )  # sorted entries: the same folder gives the same archive bytes


def test_build_label_never_shows_detached_unknown() -> None:
    base: present.RepoStatus = {"branch": "detached", "sha": "unknown", "dirty": False,
                                "kernel": "cc915c558ec0", "tcb_ok": True}  # fmt: skip
    assert present.build_label(base, {}) == "package 0.1.0"
    assert present.build_label(base, {"KNOTWORK_BUILD_SHA": "4af541bdeadbeef"}) == (
        "v2 · build 4af541b"
    )
    assert present.build_label({**base, "sha": "4af541b"}, {}) == "v2 · build 4af541b"
    named: present.RepoStatus = {**base, "branch": "main", "sha": "4af541b", "dirty": True}
    assert present.build_label(named, {}) == "main @ 4af541b + local"
    for label in (present.build_label(base, {}), present.build_label(base, {"X": "y"})):
        assert "unknown" not in label and "detached" not in label and "untracked" not in label


def test_fault_coverage_falls_back_to_the_committed_report(tmp_path: Path) -> None:
    found = present.fault_coverage(tmp_path / "no_artifacts")
    assert found["caught"] == found["total"] == 14
    assert found["report"] is not None and "samples" in Path(found["report"]).parts
