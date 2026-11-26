"""Human calibration: blind trials, timing, a JSONL log and human-versus-model comparisons."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from knotwork.knotcore.grading import extract, grade_prediction
from knotwork.knotcore.seeds import SeedTree
from knotwork.pipeline.io import append_jsonl, read_json, read_jsonl
from knotwork.render.canvas import REPO_ROOT

LOG = REPO_ROOT / "artifacts" / "calibration" / "human_calibration.jsonl"
VLM_TRIALS = REPO_ROOT / "samples" / "live_run" / "trials.jsonl"
BUNDLE_RE = re.compile(r"kw-[0-9a-f]{20}")


def queue(dataset: Path, session: str) -> list[dict[str, Any]]:
    """Base items of a dataset in a session-specific shuffled order. The returned items hold only
    what a participant may see: id, family, directory, question and options -- never the answer."""
    rows = [r for r in read_jsonl(dataset / "index.jsonl") if r.get("role", "base") == "base"]
    order = SeedTree(0).child("calibration", session).rng().permutation(len(rows))
    out = []
    for i in order:
        row = rows[int(i)]
        bundle = dataset / row["dir"]
        question = read_json(bundle / "task.json")["question"]
        out.append(
            {"bundle_id": row["bundle_id"], "family": row["family"], "plugin": row["plugin"],
             "dir": str(bundle), "question": question["text"], "options": question.get("options")}
        )  # fmt: skip
    return out


def record(
    item: dict[str, Any], response: str, elapsed_s: float, session: str, log: Path = LOG
) -> dict[str, Any]:
    """Grade one blind response against the certified answer and append it to the log."""
    answer = read_json(Path(item["dir"]) / "task.json")["answer"]
    row = {
        "session": session,
        "bundle_id": item["bundle_id"],
        "plugin": item["plugin"],
        "family": item["family"],
        "response": response,
        "truth": answer["canonical"],
        "correct": grade_prediction(f"ANSWER: {response}", answer) == 1.0,
        "elapsed_s": round(float(elapsed_s), 2),
    }
    append_jsonl(log, [row])
    return row


def done_ids(session: str, log: Path = LOG) -> set[str]:
    return {r["bundle_id"] for r in read_jsonl(log) if r["session"] == session}


def _accuracy(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(key))].append(r)
    out = []
    for name, members in sorted(groups.items()):
        timed = sorted(m["elapsed_s"] for m in members if "elapsed_s" in m)
        row: dict[str, Any] = {
            key: name,
            "n": len(members),
            "accuracy": round(sum(bool(m["correct"]) for m in members) / len(members), 4),
        }
        if timed:
            row["median_s"] = timed[len(timed) // 2]
        out.append(row)
    return out


def confusion(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wrong answers as (truth, response) pairs, most frequent first."""
    pairs = Counter(
        (r["family"], r["truth"], extract(f"ANSWER: {r['response']}"))
        for r in rows
        if not r["correct"]
    )
    return [
        {"family": f, "truth": t, "response": a, "count": c}
        for (f, t, a), c in sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def summary(log: Path = LOG, trials: Path = VLM_TRIALS) -> dict[str, Any]:
    """Human accuracy by family, model accuracy by block and slot, and head-to-head results on the
    bundles both the human log and the model trials contain."""
    human = read_jsonl(log)
    models = [t for t in read_jsonl(trials) if not t.get("missing") and not t.get("error")]
    by_bundle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in models:
        match = BUNDLE_RE.search(str(t.get("key", "")))
        if match:
            by_bundle[match.group(0)].append(t)
    matched = []
    for r in human:
        for t in by_bundle.get(r["bundle_id"], []):
            matched.append({"who": t["slot"], "correct": t["correct"], "bundle_id": r["bundle_id"]})
        if r["bundle_id"] in by_bundle:
            matched.append({"who": "human", "correct": r["correct"], "bundle_id": r["bundle_id"]})
    return {
        "n_human": len(human),
        "human_overall": round(sum(r["correct"] for r in human) / len(human), 4) if human else None,
        "human_by_family": _accuracy(human, "family"),
        "model_by_slot": _accuracy(models, "slot"),
        "model_by_block": _accuracy(models, "block"),
        "matched": _accuracy(matched, "who"),
        "confusion": confusion(human),
    }
