"""Frontier classification, seed disjointness and the post-training pack on a fixed session."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from knotwork.arena import storage
from knotwork.posttrain import curriculum
from posttrain_fixtures import session

PROV = {"git_sha": "f" * 40, "kernel_sha256": "a" * 64}


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    root = tmp_path_factory.mktemp("posttrain")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(storage.ROOT_ENV, str(root))
        cfg = session(root)
        yield storage.resolve(cfg.results_dir)


def test_regime_boundaries() -> None:
    assert curriculum.regime(0.76) == "saturated" and curriculum.regime(0.75) == "frontier"
    assert curriculum.regime(0.20) == "frontier" and curriculum.regime(0.19) == "cold_start"
    assert curriculum.regime(None) is None
    assert curriculum.advantage_proxy(0.5) == 0.25 and curriculum.advantage_proxy(1.0) == 0.02


def test_frontier_classification_and_weights(results: Path) -> None:
    spec = curriculum.analyze(results)
    got = {(c["block_id"], c["model"]): (c["accuracy"], c["regime"], c["badge"], c["mode"])
           for c in spec["cells"]}  # fmt: skip
    assert got[("b1", "m/easy")] == (1.0, "saturated", "Saturated", "GRPO")
    assert got[("b1", "m/mid")] == (0.5, "frontier", "GRPO Frontier", "GRPO")
    assert got[("b2", "m/easy")] == (0.0, "cold_start", "SFT Cold-Start", "SFT_COLD_START")
    assert got[("b2", "m/mid")] == (0.25, "frontier", "GRPO Frontier", "GRPO")
    weights = {(c["block_id"], c["model"]): c["weight"] for c in spec["cells"]}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights[("b1", "m/mid")] > weights[("b2", "m/mid")] > weights[("b1", "m/easy")]


def test_training_and_ood_seeds_are_strictly_disjoint(results: Path) -> None:
    spec = curriculum.analyze(results)
    s = spec["seeds"]
    assert s["max_eval_seed"] == 103 and s["train_seed_start"] == 10_000
    assert (
        s["train_seed_start"] > s["max_eval_seed"]
        and s["ood_eval_seed_start"] > s["train_seed_end"]
    )
    assert s["disjoint"] is True
    ranges = [b["train_seed_range"] for b in spec["blocks"]]
    assert ranges[0][1] < ranges[1][0]
    assert sum(b["n_train"] for b in spec["blocks"]) == pytest.approx(spec["train_budget"], abs=2)
    bad = [{**spec["blocks"][0], "train_seed_range": [100, 120]}]
    assert curriculum.seeds_disjoint(list(range(100, 104)), bad) is False
    jobs = curriculum.generation_jobs(spec, Path("x"), "train", limit=2)
    assert len(jobs) == 4 and all(j["seed"] >= s["train_seed_start"] for j in jobs)
    assert "posttrain-generate" in spec["cli"]["train"] and "--split ood" in spec["cli"]["ood_eval"]


def test_export_pack_writes_every_file_and_a_zip(results: Path) -> None:
    done = curriculum.export_pack(results, prov=PROV)
    pack = results / "posttrain_pack"
    assert done["dir"] == pack and done["zip"] == results / "posttrain_pack.zip"
    assert done["grpo"] == 8 and done["skipped"] == 0 and done["train_seed_disjoint"]
    sft = [json.loads(x) for x in (pack / "sft_grounded_vcot.jsonl").read_text().splitlines()]
    # b1 items 1 and 3 (m/mid failed) + all four b2 items (failures, and a cold-start block)
    assert len(sft) == done["sft"] == 6
    assert all(r["messages"][1]["content"].startswith("<think>") for r in sft)
    assert all((pack / r["image_path"]).is_file() for r in sft)
    dpo = [json.loads(x) for x in (pack / "dpo_preferences.jsonl").read_text().splitlines()]
    assert len(dpo) == done["dpo"] == 2 + 7
    assert all(r["chosen_reward"] > r["rejected_reward"] for r in dpo)
    assert {r["chosen_source"] for r in dpo} == {"model:m/easy", "model:m/mid",
                                                  "deterministic_vcot"}  # fmt: skip
    table = pq.read_table(pack / "grpo_rlvr_verl.parquet").to_pylist()
    assert len(table) == 8
    row = table[0]
    assert row["data_source"] == "knotwork/rack_patch" and row["reward_model"]["style"] == "rule"
    assert json.loads(row["reward_model"]["ground_truth"])["type"] == "label"
    assert row["extra_info"]["kernel_sha256"] == PROV["kernel_sha256"]
    assert all(0 <= v <= 1000 for b in row["extra_info"]["vcot_boxes_1000"] for v in b)
    spec = json.loads((pack / "curriculum_spec.json").read_text())
    assert spec["seeds"]["disjoint"] is True
    att = json.loads((pack / "provenance_attestation.json").read_text())
    assert att["train_seed_disjoint"] is True and att["git_sha"] == PROV["git_sha"]
    assert "qa_filter" not in att and "qa_filter" not in done  # Human QA filter is opt-in
    assert att["clean_ip_procedural"] and att["zero_web_scrape"] and att["zero_pii"]
    assert [a["path"] for a in att["external_assets"]] == ["assets/fonts/DejaVuSans.ttf"]
    assert "curriculum_spec.json" in att["files_sha256"]
    with zipfile.ZipFile(done["zip"]) as z:
        names = set(z.namelist())
    for name in ("sft_grounded_vcot.jsonl", "dpo_preferences.jsonl", "grpo_rlvr_verl.parquet",
                 "grpo_rlvr_verl.jsonl", "curriculum_spec.json", "provenance_attestation.json",
                 "rlvr_kit/knotwork_verl_reward.py", "rlvr_kit/run_verl_grpo.sh",
                 "rlvr_kit/run_trl_grpo.py"):  # fmt: skip
        assert f"posttrain_pack/{name}" in names, name
    again = curriculum.zip_dir(pack, results / "again.zip").read_bytes()
    assert again == done["zip"].read_bytes()  # deterministic: sorted names, fixed timestamps
