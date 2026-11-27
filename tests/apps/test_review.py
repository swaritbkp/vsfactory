"""Human QA review states: the append-only review log, and the *Approved items only* filter of
Export and of the Post-Training Pack (default off, filtered on, and the 0-approved edge case)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from knotwork.arena import storage
from knotwork.export.run import run_export
from knotwork.posttrain import curriculum
from knotwork.studio import review
from posttrain_fixtures import session

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "tests" / "bench" / "fixtures" / "data" / "tangle"
PROV = {"git_sha": "f" * 40, "kernel_sha256": "a" * 64}


def test_review_log_appends_and_latest_state_wins(tmp_path: Path) -> None:
    log = tmp_path / "qa" / review.LOG_NAME
    assert review.load_review_states(log) == {}
    review.record_review(log, "kw-a", "approved", "tangle", 3)
    review.record_review(log, "kw-b", "flagged", "gauge", 4, note="  needle  hidden\n by label ")
    review.record_review(log, "kw-a", "flagged", "tangle", 3, reviewer="")
    review.record_review(log, "kw-b", "approved", "gauge", 4)
    with log.open("a", encoding="utf-8") as f:
        f.write('{"item_id": "kw-c", "status": "appro')  # a torn last line is skipped
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5  # append-only: every decision kept
    assert json.loads(lines[1])["note"] == "needle hidden by label"
    states = review.load_review_states(log)
    assert {k: v.status for k, v in states.items()} == {"kw-a": "flagged", "kw-b": "approved"}
    assert states["kw-a"].reviewer == "operator" and states["kw-a"].timestamp_utc.endswith("Z")
    assert review.approved_ids(log) == {"kw-b"}
    assert review.counts(states, ["kw-a", "kw-b", "kw-z"]) == {
        "unreviewed": 1, "approved": 1, "flagged": 1}  # fmt: skip
    with pytest.raises(ValueError):
        review.record_review(log, "kw-a", "maybe")


def _bundle_ids() -> list[str]:
    rows = (DATASET / "index.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(r)["bundle_id"] for r in rows if r.strip()]


def test_export_approved_only(tmp_path: Path) -> None:
    ids = _bundle_ids()
    out = tmp_path / "all"
    assert run_export(DATASET, "manifest", out, None) == 0
    manifest = json.loads((out / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_bundles"] == len(ids) and "qa_filter" not in manifest
    out = tmp_path / "approved"
    assert run_export(DATASET, "manifest", out, None, only_ids=set(ids[:2])) == 0
    manifest = json.loads((out / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_bundles"] == 2
    assert manifest["qa_filter"] == review.QA_FILTER and manifest["approved_count"] == 2
    assert json.loads((out / "export_summary.json").read_text())["candidate_count"] == len(ids)
    assert run_export(DATASET, "manifest", tmp_path / "none", None, only_ids=set()) == 1
    assert not (tmp_path / "none" / "dataset_manifest.json").exists()


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("qa_pack")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(storage.ROOT_ENV, str(root))
        cfg = session(root)
        return storage.resolve(cfg.results_dir)


def test_posttrain_pack_approved_only(results: Path) -> None:
    log = results / review.LOG_NAME
    with pytest.raises(ValueError, match="0 of 8 items are approved"):
        curriculum.export_pack(results, prov=PROV, approved_only=True)
    items = curriculum.load_session(results)[1]
    for item in items[:3]:
        review.record_review(log, item.bundle_id, "approved", item.domain, item.seed)
    review.record_review(log, items[3].bundle_id, "flagged", items[3].domain, items[3].seed)
    done = curriculum.export_pack(results, prov=PROV, approved_only=True)
    assert done["grpo"] == 3 and done["approved_count"] == 3 and done["candidate_count"] == 8
    att = json.loads((done["dir"] / "provenance_attestation.json").read_text(encoding="utf-8"))
    assert att["qa_filter"] == "approved_only" and att["approved_count"] == 3
    rows = (done["dir"] / "grpo_rlvr_verl.jsonl").read_text(encoding="utf-8").splitlines()
    kept = {json.loads(r)["extra_info"]["bundle_id"] for r in rows}
    assert kept == {i.bundle_id for i in items[:3]}
    # the default (filter off) is covered by tests/posttrain/test_curriculum.py: all 8 items, no
    # qa_filter key, so packs written before Human QA existed are unchanged
