"""Task 4.3: HF parquet columns + round trip, dataset card, RLVR grading, sealed answers."""

from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from PIL import Image

from knotwork.export.hf import COLUMNS, write_hf
from knotwork.export.records import load_items
from knotwork.export.rlvr import REWARD_FN, write_rlvr
from knotwork.export.run import run_export
from knotwork.freshness.emission import answer_commitment, load_salt
from knotwork.knotcore.grading import grade_prediction

sys.path.insert(0, str(Path(__file__).parent))
from test_harbor_oracle import wrong  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def test_parquet_columns_and_round_trip(exported: Path, export_data: Path) -> None:
    files = sorted((exported / "hf" / "data").glob("*.parquet"))
    assert [f.name for f in files] == ["dev-00000-of-00001.parquet"]
    t = pq.read_table(files[0])
    assert tuple(t.column_names) == COLUMNS
    rows = t.to_pylist()
    assert len(rows) == 33
    tasks = {}
    for ds in export_data.iterdir():
        for line in (ds / "index.jsonl").read_text().splitlines():
            r = json.loads(line)
            tasks[r["bundle_id"]] = json.loads((ds / r["dir"] / "task.json").read_text())
    for r in rows:
        task = tasks[r["bundle_id"]]
        assert json.loads(r["answer_json"]) == task["answer"]
        assert r["answer"] == task["answer"]["canonical"]
        assert json.loads(r["certificate_json"]) == task["certificate"]
        assert r["canary"] == task["canary"] and r["twin_role"] == task["twins"]["role"]
        assert Image.open(io.BytesIO(r["image"]["bytes"])).size == (768, 768)
        assert r["image"]["path"] == f"images/{r['bundle_id']}.png"
        assert r["answer_commitment"] is None and isinstance(r["options"], list)
        assert json.loads(r["steps_json"]) and "score" in json.loads(r["difficulty_json"])


def test_dataset_card(exported: Path) -> None:
    text = (exported / "hf" / "README.md").read_text(encoding="utf-8")
    front = yaml.safe_load(text.split("---")[1])
    assert front["configs"][0]["config_name"] == "default"
    assert front["configs"][0]["data_files"] == [{"split": "dev", "path": "data/dev-*.parquet"}]
    assert front["license"] == "apache-2.0"
    assert "KNOTWORK-CANARY-" in text and "Intended use" in text


def test_rlvr_oracle_and_wrong(exported: Path) -> None:
    lines = [json.loads(x) for x in (exported / "rlvr" / "tasks.jsonl").read_text().splitlines()]
    assert len(lines) == 33
    for ln in lines:
        assert ln["reward_fn"] == REWARD_FN and ln["canary"].startswith("KNOTWORK-CANARY-")
        assert (exported / "rlvr" / ln["image_path"]).is_file()
        assert ln["steps"] and {"op", "inputs", "output"} == set(ln["steps"][0])
        a = ln["answer"]
        assert grade_prediction(a["canonical"], a) == 1.0
        assert grade_prediction(f"reasoning...\nANSWER: {a['canonical']}", a) == 1.0
        assert grade_prediction(wrong(a), a) == 0.0


def test_sealed_answers_become_commitments(export_data: Path, tmp_path: Path) -> None:
    ds = tmp_path / "sealed"
    shutil.copytree(export_data / "gauge", ds)
    for line in (ds / "index.jsonl").read_text().splitlines():
        p = ds / json.loads(line)["dir"] / "task.json"
        t = json.loads(p.read_text())
        t["split"] = "test"
        p.write_text(json.dumps(t))
    prof = yaml.safe_load((REPO / "configs/default_profile.yaml").read_text(encoding="utf-8"))
    prof["freshness"]["sealed_fraction"] = 1.0
    items = load_items(ds, prof)
    salt = load_salt(prof["freshness"]["emission_salt_path"])
    assert items and all(i.sealed for i in items)
    for i in items:
        assert i.commitment == answer_commitment(salt, i.task["answer"]["canonical"])
    write_hf(items, tmp_path / "out")
    write_rlvr(items, tmp_path / "out")
    hf = tmp_path / "out" / "hf" / "data" / "test-00000-of-00001.parquet"
    for r in pq.read_table(hf).to_pylist():
        assert r["answer"] is None and r["answer_json"] is None
        assert len(r["answer_commitment"]) == 64
    for x in (tmp_path / "out" / "rlvr" / "tasks.jsonl").read_text().splitlines():
        ln = json.loads(x)
        assert "answer" not in ln and len(ln["answer_commitment"]) == 64


def test_limit_and_formats(export_data: Path, tmp_path: Path) -> None:
    assert run_export(export_data, "rlvr", tmp_path / "o", 5) == 0
    assert len((tmp_path / "o" / "rlvr" / "tasks.jsonl").read_text().splitlines()) == 5
    assert not (tmp_path / "o" / "harbor").exists() and not (tmp_path / "o" / "hf").exists()
