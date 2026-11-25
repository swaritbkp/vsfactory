"""Benchmark summary (Phase 6): per-model metrics from ``bench_report.json`` + ``trials.jsonl``.

Source preference: the live run (``artifacts/phase5/bench_live``), else its offline replay, else the
mock run; the chosen source is recorded in the summary so every figure can label it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from knotwork.render.canvas import REPO_ROOT

SOURCES = (
    ("live", "phase5/bench_live"),
    ("replay", "phase5/bench_replayed"),
    ("mock", "phase5/bench_mock"),
)
PLUGIN_OF_BLOCK = {
    "staircase:tangle": "tangle", "ablation:unique": "tangle", "ablation:mono": "tangle",
    "twins": "tangle", "staircase:panel": "panel", "gaslight": "panel", "composition": "panel",
}  # fmt: skip


def find_source(artifacts: Path) -> tuple[str, Path] | None:
    for name, rel in SOURCES:
        d = artifacts / rel
        if (d / "bench_report.json").is_file() and (d / "trials.jsonl").is_file():
            return name, d
    return None


def _acc(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in rows if not r["missing"]]
    if not rows:
        return {"acc": None, "n": 0}
    return {"acc": round(sum(r["correct"] for r in rows) / len(rows), 4), "n": len(rows)}


def _mean(xs: list[bool]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def image_of(key: str, artifacts: Path, bench_dir: Path) -> str | None:
    """Repo-relative beauty.png of a trial key (pool bundle or derived render)."""
    kind, bid = key.split(":", 1)
    root = artifacts.parent
    if kind == "unique":
        p = bench_dir / "derived" / "ablation" / bid / "beauty.png"
    elif kind in ("causal", "nuisance"):
        hits = sorted((bench_dir / "derived" / "twins" / bid).glob(f"{kind}*/beauty.png"))
        p = hits[0] if hits else bench_dir / "missing"
    else:
        hits = sorted((artifacts / "phase3" / "bench_split").glob(f"*/bundles/{bid}/beauty.png"))
        p = hits[0] if hits else bench_dir / "missing"
    return p.relative_to(root).as_posix() if p.is_file() else None


def model_row(
    slot: str, slug: str, m: dict[str, Any], trials: list[dict[str, Any]]
) -> dict[str, Any]:
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in trials:
        by_block[r["block"]].append(r)
    tw = by_block["twins"]
    triples = [tw[i : i + 3] for i in range(0, len(tw) - 2, 3)]
    base = [t[0]["correct"] for t in triples]
    caus = [t[1]["correct"] for t in triples]
    nuis = [t[2]["correct"] for t in triples]
    mono = by_block["ablation:mono"]
    return {
        "slot": slot,
        "slug": slug,
        "trials": len(trials),
        "errors": int(m.get("n_errors", sum(1 for r in trials if r.get("error")))),
        "missing": sum(1 for r in trials if r["missing"]),
        "parse_failures": sum(1 for r in trials if not r["missing"] and not r["parsed"]),
        "cost_usd": m.get("cost_usd", 0.0),
        "seconds": m.get("seconds"),
        "accuracy": {
            "overall": _acc(trials),
            "tangle": _acc([r for r in trials if PLUGIN_OF_BLOCK.get(r["block"]) == "tangle"]),
            "panel": _acc([r for r in trials if PLUGIN_OF_BLOCK.get(r["block"]) == "panel"]),
            "gauge_read": _acc(
                [r for r in by_block["composition"] if r["key"].startswith("read:")]
            ),
        },
        "blocks": {k: _acc(v) for k, v in sorted(by_block.items())},
        "twins": {
            "n": len(triples),
            "original": _mean(base),
            "causal": _mean(caus),
            "nuisance": _mean(nuis),
            "all_three": _mean([all(r["correct"] for r in t) for t in triples]),
            "answer_changed_on_causal": _mean([t[0]["reply"] != t[1]["reply"] for t in triples]),
            "same_answer_on_nuisance": m.get("twins", {}).get("nuisance_consistency"),
        },
        "color_ablation": dict(
            m.get("color_ablation", {}), chance_mono=_mean_inv([r["m"] for r in mono])
        ),  # fmt: skip
        "gaslight": m.get("gaslight", {}),
        "staircase": m.get("staircase", {}),
        "per_crossing": m.get("per_crossing", {}),
        "composition": m.get("composition", {}),
        "track_switch": m.get("track_switch", {}),
        "parse_fail_rate": m.get("parse_fail_rate"),
    }


def _mean_inv(ms: list[float]) -> float | None:
    ms = [x for x in ms if x]
    return round(sum(1.0 / x for x in ms) / len(ms), 4) if ms else None


def question_of(key: str, artifacts: Path) -> str:
    """The question actually asked for a pool trial key (first line of the prompt)."""
    kind, bid = key.split(":", 1)
    hits = sorted((artifacts / "phase3" / "bench_split").glob(f"*/bundles/{bid}/task.json"))
    if not hits:
        return ""
    task = json.loads(hits[0].read_text(encoding="utf-8"))
    if kind == "read":
        trace = next((st for st in task["certificate"]["steps"] if st["op"] == "TRACE"), None)
        return f"What value does gauge {trace['output']} read?" if trace else ""
    if kind == "trace":
        start = str(task["question"]["steps"][0]["args"][0]).split("start_", 1)[-1]
        return f"Which gauge is the cable from socket {start} plugged into?"
    text = str(task["question"]["text"]).split("\n")[0]
    return text.replace(
        "The image shows cables running from lettered terminals to numbered terminals. ", ""
    )


def failure_examples(trials: list[dict[str, Any]], slugs: dict[str, str], artifacts: Path,
                     bench_dir: Path) -> list[dict[str, Any]]:  # fmt: skip
    """Items every model got wrong: the easiest tangle trace, a gauge read, and the false premise
    with the most divergent replies."""
    items: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in trials:
        items[(r["block"], r["key"])][r["slot"]] = r
    n_models = len(slugs)
    wrong = [(blk, key, d) for (blk, key), d in sorted(items.items())
             if len(d) == n_models and not any(x["correct"] for x in d.values())]  # fmt: skip
    picks: list[tuple[str, str, dict[str, dict[str, Any]]]] = []
    tr = sorted(
        (x for x in wrong if x[0] == "staircase:tangle"),
        key=lambda x: (next(iter(x[2].values()))["k"], x[1]),
    )
    rd = [x for x in wrong if x[0] == "composition" and x[1].startswith("read:")]
    gas = sorted((x for x in wrong if x[0] == "gaslight"),
                 key=lambda x: (-sum(str(v["reply"] or "").lower().startswith("yes") for v in x[2].values()), x[1]))  # fmt: skip
    for group in (tr, rd, gas):
        if group:
            picks.append(group[0])
    out = []
    for blk, key, d in picks:
        any_row = next(iter(d.values()))
        out.append({"block": blk, "key": key, "image": image_of(key, artifacts, bench_dir), "truth": any_row["answer"],
                    "question": question_of(key, artifacts), "crossings_on_path": any_row["k"], "replies": {slugs[s]: d[s]["reply"] for s in sorted(d)}})  # fmt: skip
    return out


def build_summary(artifacts: Path) -> dict[str, Any] | None:
    artifacts = Path(artifacts).resolve()
    found = find_source(artifacts)
    if found is None:
        return None
    source, d = found
    rep = json.loads((d / "bench_report.json").read_text(encoding="utf-8"))
    trials = [
        json.loads(x)
        for x in (d / "trials.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    slugs = _slugs(REPO_ROOT, list(rep["models"]))
    models = {
        s: model_row(s, slugs[s], m, [r for r in trials if r["slot"] == s])
        for s, m in rep["models"].items()
    }
    return {
        "source": source,
        "dir": d.relative_to(artifacts.parent).as_posix(),
        "mode": rep["mode"],
        "cost_usd": rep["cost_usd"],
        "network_calls": rep["network_calls"],
        "trials": len(trials),
        "stopped": rep["stopped"],
        "max_trials": rep.get("max_trials"),
        "models": models,
        "failure_examples": failure_examples(trials, slugs, artifacts, d),
    }


def _slugs(repo: Path, slots: list[str]) -> dict[str, str]:
    import yaml

    cfg = yaml.safe_load((repo / "configs" / "models.yaml").read_text(encoding="utf-8"))
    known = {s["slot"]: s["slug"] for s in cfg.get("slots", [])}
    return {s: known.get(s, s) for s in slots}


def write_summary(artifacts: Path) -> Path | None:
    artifacts = Path(artifacts).resolve()
    summ = build_summary(artifacts)
    if summ is None:
        return None
    out = artifacts / "phase5" / "bench_summary.json"
    out.write_text(json.dumps(summ, indent=1, sort_keys=True), encoding="utf-8")
    return out


__all__ = ["build_summary", "find_source", "write_summary"]
