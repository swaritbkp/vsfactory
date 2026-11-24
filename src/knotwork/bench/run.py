"""`knot bench` (SPEC 12): adaptive staircases + diagnostic blocks for every model slot.

Modes: ``mock`` (simulated VLMs from ``mock_slots``), ``replay`` (cached responses only, a miss is
recorded as missing), ``openrouter`` (live; triple-locked by the CLI) and ``dry_run`` (writes
``OUT/preflight.json`` only). Per model:

* staircases (SPEC 12.2): tangle T1 by crossings on the queried strand; panel = the TRACE-only
  question on P1 images by crossings on the traced cable (D-33);
* colour ablation: matched T1 pairs (unique-colour re-render vs the mono original);
* twins: T1 base + causal + nuisance triples (re-built and re-certified);
* gaslight: T_GASLIGHT items; composition: TRACE-only, READ-only and READ(TRACE) on the same P1 image.

Block sizes come from ``models.yaml: plan`` scaled by ``max_trials / 50``. Derived renders are shared
by all models, so every model sees exactly the same items.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from knotwork.bench import stats
from knotwork.bench.budget import Budget, preflight
from knotwork.bench.client_base import BudgetExhausted, ReplayMiss, Reply, VLMClient
from knotwork.bench.client_mock import MockClient
from knotwork.bench.client_replay import ReplayClient, ReplayRecorder, replay_key
from knotwork.bench.prompts import image_png, image_sha256, read_only, trace_only
from knotwork.bench.staircase import Staircase, expected_level
from knotwork.datasets import dataset_dirs
from knotwork.knotcore.grading import ANSWER_RE, YESNO_RE, extract, grade_prediction
from knotwork.knotcore.seeds import SeedTree
from knotwork.render.canvas import REPO_ROOT

PROFILE = REPO_ROOT / "configs" / "default_profile.yaml"
LEVELS = {"tangle": (1, 12), "panel": (1, 8)}
BASE_TRIALS = 50


@dataclass
class Item:
    key: str
    beauty: Path
    prompt: str
    answer: dict[str, Any]
    meta: dict[str, Any]
    bundle_id: str
    scene_id: str
    k: int = 0
    m: float = 0.0
    ds: Path | None = None
    task: dict[str, Any] | None = None


@dataclass
class P1:
    trace: Item
    read: Item
    comp: Item
    c_trace: float
    c_read: float


@dataclass
class Pool:
    t1: list[Item] = field(default_factory=list)
    p1: list[P1] = field(default_factory=list)
    gaslight: list[Item] = field(default_factory=list)


# ------------------------------------------------------------------ pool


def _read(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _cross(claims: dict[str, Any], start: str) -> int:
    for c in claims["claims"]:
        if c["op"] == "CROSS" and c["args"][0] == start and c["args"][1] == "*":
            return int(c["expected"]["value"])
    return 0


def t1_item(d: Path, task: dict[str, Any], ds: Path | None, regime: str = "mono") -> Item:
    q = task["question"]
    start = str(q["steps"][0]["args"][0])
    k = _cross(_read(d / "claims.json"), start)
    opts = list(q.get("options") or [])
    return Item(
        key=f"t1:{task['bundle_id']}", beauty=d / "beauty.png", prompt=q["text"], answer=task["answer"],
        meta={"kind": "trace", "answer": task["answer"]["canonical"], "options": opts, "k": k, "regime": regime},
        bundle_id=task["bundle_id"], scene_id=task["scene_id"], k=k, m=float(len(opts)), ds=ds, task=task,
    )  # fmt: skip


def p1_record(d: Path, task: dict[str, Any]) -> P1 | None:
    claims, man = _read(d / "claims.json"), _read(d / "manifest.json")
    q, a = task["question"], task["answer"]
    start = str(q["steps"][0]["args"][0])
    steps = {s["op"]: s for s in task["certificate"]["steps"]}
    if "TRACE" not in steps or "GAUGE_OF" not in steps:
        return None
    gauge_label, gauge_id = str(steps["TRACE"]["output"]), str(steps["GAUGE_OF"]["output"])
    values = {
        c["args"][0]: float(c["expected"]["value"]) for c in claims["claims"] if c["op"] == "READ"
    }
    k_g = len(man["gauges"])
    span, tol = float(q["difficulty"].get("span", 80.0)), float(a.get("tolerance") or 0.0)
    level = _cross(claims, start)
    socket = start.split("start_", 1)[-1]
    opts = [f"G{i + 1}" for i in range(k_g)]
    others = [v for g, v in sorted(values.items()) if g != gauge_id]
    value = values.get(gauge_id, float(a["value"]))
    base = {
        "bundle_id": task["bundle_id"],
        "scene_id": task["scene_id"],
        "beauty": d / "beauty.png",
        "k": level,
    }
    trace = Item(key=f"trace:{task['bundle_id']}", prompt=trace_only(socket, k_g),
                 answer={"type": "label", "value": gauge_label, "tolerance": 0.0, "canonical": gauge_label},
                 meta={"kind": "trace", "answer": gauge_label, "options": opts, "k": level}, m=float(k_g), **base)  # fmt: skip
    read = Item(key=f"read:{task['bundle_id']}", prompt=read_only(gauge_label),
                answer={"type": "float", "value": value, "tolerance": tol, "canonical": format(value, "g")},
                meta={"kind": "read", "value": value, "span": span}, **base)  # fmt: skip
    comp = Item(key=f"comp:{task['bundle_id']}", prompt=q["text"], answer=a,
                meta={"kind": "comp", "value": value, "others": others, "k": level, "span": span}, **base)  # fmt: skip
    c_num = min(1.0, 2.0 * tol / span) if span > 0 else 0.0
    return P1(trace, read, comp, 1.0 / k_g, c_num)


def load_pool(data: Path) -> Pool:
    pool = Pool()
    for ds in dataset_dirs(Path(data)):
        for line in (ds / "index.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["role"] != "base":
                continue
            d = ds / row["dir"]
            task = _read(d / "task.json")
            fam = task["family"]
            if fam == "T1_TRACE":
                pool.t1.append(t1_item(d, task, ds))
            elif fam == "P1_READ_TRACE":
                rec = p1_record(d, task)
                if rec is not None:
                    pool.p1.append(rec)
            elif fam == "T_GASLIGHT":
                a = task["answer"]
                pool.gaslight.append(Item(
                    key=f"gas:{task['bundle_id']}", beauty=d / "beauty.png", prompt=task["question"]["text"],
                    answer=a, bundle_id=task["bundle_id"], scene_id=task["scene_id"],
                    meta={"kind": "premise", "holds": bool(a["value"]["premise_holds"]),
                          "value": float(a["value"]["value"]), "span": float(task["question"]["difficulty"].get("span", 80.0))},
                ))  # fmt: skip
    return pool


# ------------------------------------------------------------------ plan
def block_sizes(cfg: dict[str, Any], max_trials: int) -> dict[str, int]:
    plan, scale = cfg["plan"], max_trials / BASE_TRIALS
    return {k: max(1, math.ceil(int(plan[f"{k}_trials"]) * scale))
            for k in ("ablation", "twin", "gaslight", "composition")}  # fmt: skip


def tracks_of(track: str) -> list[str]:
    return ["tangle", "panel"] if track == "all" else [track]


def calls_per_model(sizes: dict[str, int], tracks: list[str], max_trials: int) -> int:
    n = max_trials * len(tracks)
    if "tangle" in tracks:
        n += 2 * sizes["ablation"] + 3 * sizes["twin"]
    if "panel" in tracks:
        n += sizes["gaslight"] + 3 * sizes["composition"]
    return n


def select(pool: Pool, sizes: dict[str, int], seed: int) -> dict[str, list[Any]]:
    rng = SeedTree(seed).child("bench", "blocks").rng()
    t1 = [pool.t1[int(i)] for i in rng.permutation(len(pool.t1))]
    p1 = [pool.p1[int(i)] for i in rng.permutation(len(pool.p1))]
    gas = [pool.gaslight[int(i)] for i in rng.permutation(len(pool.gaslight))]
    a, t = sizes["ablation"], sizes["twin"]
    return {"ablation": t1[:a], "twin": t1[a : a + t], "gaslight": gas[: sizes["gaslight"]],
            "composition": p1[: sizes["composition"]]}  # fmt: skip


def _map(fn: Any, jobs: list[dict[str, Any]], workers: int) -> list[Any]:
    from knotwork.pipeline.generate import run_jobs

    return run_jobs(jobs, fn, workers) if jobs else []


def derive(
    chosen: dict[str, list[Any]], out: Path, profile: dict[str, Any], workers: int
) -> dict[str, Any]:
    """Unique-colour variants (ablation) and twin triples, re-certified (shared by all models)."""
    from knotwork.bench.derive import twin_set, unique_variant

    abl = [{"ds": str(it.ds), "dir": str(it.beauty.parent), "out": str(out / "ablation" / it.bundle_id),
            "profile": profile} for it in chosen["ablation"]]  # fmt: skip
    tw = [{"ds": str(it.ds), "dir": str(it.beauty.parent), "out": str(out / "twins" / it.bundle_id),
           "profile": profile} for it in chosen["twin"]]  # fmt: skip
    pairs = []
    for it, res in zip(chosen["ablation"], _map(unique_variant, abl, workers), strict=True):
        if res["ok"]:
            d = Path(res["dir"])
            uniq = t1_item(
                d, dict(it.task or {}, bundle_id=f"{it.bundle_id}-u"), None, regime="unique"
            )
            uniq.key = f"unique:{it.bundle_id}"
            pairs.append((uniq, it))
    triples = []
    for it, res in zip(chosen["twin"], _map(twin_set, tw, workers), strict=True):
        if res["ok"]:
            tw_items = []
            for role in ("causal", "nuisance"):
                r = res[role]
                task = dict(
                    it.task or {},
                    bundle_id=r["bundle_id"],
                    answer=r["answer"],
                    question=r["question"],
                )
                x = t1_item(Path(r["dir"]), task, None)
                x.key = f"{role}:{it.bundle_id}"
                tw_items.append(x)
            triples.append((it, tw_items[0], tw_items[1]))
    return {"pairs": pairs, "triples": triples}


# ------------------------------------------------------------------ trials
@dataclass
class Outcome:
    correct: bool
    parsed: bool
    text: str
    missing: bool = False


class Runner:
    def __init__(self, client: VLMClient, slot: str, seed: int, ledger_rows: list[dict[str, Any]],
                 slug: str) -> None:  # fmt: skip
        self.client, self.slot, self.seed, self.slug = client, slot, seed, slug
        self.png: dict[Path, bytes] = {}
        self.log: list[dict[str, Any]] = []
        self.ledger_rows = ledger_rows

    def image(self, p: Path) -> bytes:
        if p not in self.png:
            self.png[p] = image_png(p)
        return self.png[p]

    def _outcome(self, item: Item, reply: Reply | None, block: str) -> Outcome:
        if reply is None:
            out = Outcome(False, False, "", missing=True)
        else:
            parsed = bool(ANSWER_RE.search(reply.text))
            out = Outcome(
                parsed and grade_prediction(reply.text, item.answer) == 1.0, parsed, reply.text
            )
        self.log.append({"slot": self.slot, "block": block, "key": item.key, "k": item.k, "m": item.m,
                         "correct": out.correct, "parsed": out.parsed, "missing": out.missing,
                         "answer": item.answer["canonical"], "reply": extract(out.text)[:200] if out.text else None,
                         "cost_usd": reply.cost_usd if reply else 0.0})  # fmt: skip
        if reply is not None and self.client.mode == "live":
            png = self.image(item.beauty)
            self.ledger_rows.append({"bundle_id": item.bundle_id, "scene_id": item.scene_id, "slot": self.slot,
                                     "model": self.slug, "request_sha256": replay_key(
                                         self.slot, self.slug, item.prompt, image_sha256(png), self.seed)})  # fmt: skip
        return out

    def ask(self, item: Item, block: str) -> Outcome:
        try:
            reply = self.client.ask(
                self.slot, self.image(item.beauty), item.prompt, self.seed, item.meta
            )
        except ReplayMiss:
            reply = None
        return self._outcome(item, reply, block)

    def ask_many(self, items: list[Item], block: str) -> list[Outcome]:
        many = getattr(self.client, "ask_many", None)
        if many is None:
            return [self.ask(it, block) for it in items]
        res = many([(self.slot, self.image(it.beauty), it.prompt, self.seed) for it in items])
        outs = []
        for it, r in zip(items, res, strict=True):
            if isinstance(r, BudgetExhausted):
                raise r
            outs.append(self._outcome(it, r if isinstance(r, Reply) else None, block))
        return outs


def run_staircase(runner: Runner, items: list[Item], track: str, max_trials: int, seed: int,
                  p_true: float | None) -> tuple[dict[str, Any], list[tuple[Item, Outcome]]]:  # fmt: skip
    by_level: dict[int, list[Item]] = defaultdict(list)
    lo, cap = LEVELS[track]
    for it in items:
        if lo <= it.k <= cap:
            by_level[it.k].append(it)
    if not by_level:
        return {"threshold_level": None, "expected_level": None, "reversals": [], "trials": 0}, []
    hi = max(by_level)
    rng = SeedTree(seed).child("bench", runner.slot, track).rng()
    queues: dict[int, list[Item]] = {}

    def draw(level: int) -> Item:
        lv = min(by_level, key=lambda x: (abs(x - level), x))
        if not queues.get(lv):
            queues[lv] = [by_level[lv][int(i)] for i in rng.permutation(len(by_level[lv]))]
        return queues[lv].pop()

    st = Staircase(lo, hi, max_trials=max_trials)
    done: list[tuple[Item, Outcome]] = []
    attempts = 0
    while not st.done and attempts < 3 * max_trials:
        attempts += 1
        it = draw(st.level)
        out = runner.ask(it, f"staircase:{track}")
        if out.missing:
            continue
        st.update(out.correct)
        done.append((it, out))
    exp = None
    if p_true is not None:

        def m_bar(k: float) -> float:
            lv = min(by_level, key=lambda x: (abs(x - k), x))
            return float(np.mean([i.m for i in by_level[lv]]))

        exp = round(expected_level(p_true, m_bar, lo, hi), 4)
    return {"threshold_level": round(st.threshold(), 4), "expected_level": exp, "reversals": st.reversals,
            "trials": len(st.history), "levels": [lo, hi]}, done  # fmt: skip


def single_switch(item: Item, profile: dict[str, Any], cache: dict[str, set[str]]) -> set[str]:
    """End labels reachable from the queried start by switching track exactly once."""
    if item.key in cache:
        return cache[item.key]
    from knotwork.diversity import rebuild_scene
    from knotwork.plugins.tangle import braid

    reach: set[str] = set()
    if item.ds is not None and item.task is not None:
        pay = rebuild_scene(item.ds, item.task, profile).payload
        n = int(pay["n"])
        word = [None if x is None else tuple(int(v) for v in x) for x in pay["word"]]
        start = str(item.task["question"]["steps"][0]["args"][0]).split("start_", 1)[-1]
        s = list(pay["start_labels"]).index(start)
        ends = braid.end_position(n, word)
        for c in braid.compose(n, word)[1]:
            if s in (c.a, c.b):
                reach.add(str(pay["end_labels"][ends[c.b if c.a == s else c.a]]))
    cache[item.key] = reach
    return reach


def _premise_token(text: str) -> str | None:
    m = YESNO_RE.search(extract(text)) if text else None
    return m.group(1).lower() if m else None


def run_model(runner: Runner, pool: Pool, chosen: dict[str, list[Any]], derived: dict[str, Any],
              tracks: list[str], max_trials: int, seed: int, p_true: float | None,
              profile: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    row: dict[str, Any] = {"staircase": {}}
    t1_trials: list[tuple[Item, Outcome]] = []
    rng = SeedTree(seed).child("bench", runner.slot, "stats").rng()
    if "tangle" in tracks:
        row["staircase"]["tangle"], done = run_staircase(
            runner, pool.t1, "tangle", max_trials, seed, p_true
        )
        t1_trials += done
        pairs = []
        uniq = runner.ask_many([u for u, _ in derived["pairs"]], "ablation:unique")
        mono = runner.ask_many([m for _, m in derived["pairs"]], "ablation:mono")
        for (_, mi), ou, om in zip(derived["pairs"], uniq, mono, strict=True):
            if not (ou.missing or om.missing):
                pairs.append((ou.correct, om.correct))
                t1_trials.append((mi, om))
        row["color_ablation"] = stats.color_ablation(pairs)
        trip = []
        flat = [x for t in derived["triples"] for x in t]
        outs = runner.ask_many(flat, "twins")
        for i, (b, c, n) in enumerate(derived["triples"]):
            ob, oc, on = outs[3 * i : 3 * i + 3]
            if not (ob.missing or oc.missing or on.missing):
                trip.append(
                    (
                        ob.correct,
                        oc.correct,
                        extract(ob.text).strip().upper(),
                        extract(on.text).strip().upper(),
                    )
                )
                t1_trials += [(b, ob), (c, oc), (n, on)]
        row["twins"] = stats.twins(trip)
        valid = [(it, o) for it, o in t1_trials if it.m > 0]
        row["per_crossing"] = stats.per_crossing(
            [{"k": it.k, "m": it.m, "correct": o.correct} for it, o in valid], rng
        )
        cache: dict[str, set[str]] = {}
        errors = [(extract(o.text).strip().upper(), {x.upper() for x in single_switch(it, profile, cache)})
                  for it, o in valid if o.parsed and not o.correct and it.ds is not None]  # fmt: skip
        row["track_switch"] = stats.track_switch(errors)
    if "panel" in tracks:
        row["staircase"]["panel"], _ = run_staircase(
            runner, [r.trace for r in pool.p1], "panel", max_trials, seed, p_true
        )
        gas = runner.ask_many(chosen["gaslight"], "gaslight")
        row["gaslight"] = stats.gaslight([_premise_token(o.text) for o in gas if not o.missing])
        comp_rows = []
        recs: list[P1] = chosen["composition"]
        outs = runner.ask_many([x for r in recs for x in (r.trace, r.read, r.comp)], "composition")
        for i, r in enumerate(recs):
            ot, orr, oc = outs[3 * i : 3 * i + 3]
            if not (ot.missing or orr.missing or oc.missing):
                comp_rows.append({"trace": ot.correct, "read": orr.correct, "comp": oc.correct,
                                  "c_trace": r.c_trace, "c_read": r.c_read, "c_comp": r.c_read})  # fmt: skip
        row["composition"] = stats.composition(comp_rows, rng)
    answered = [e for e in runner.log if not e["missing"]]
    row["parse_fail_rate"] = (
        round(sum(not e["parsed"] for e in answered) / len(answered), 6) if answered else 0.0
    )
    row["n_trials"], row["n_missing"] = len(answered), len(runner.log) - len(answered)
    row["cost_usd"] = round(sum(e["cost_usd"] for e in runner.log), 6)
    return row


# ------------------------------------------------------------------ entry point
def make_client(
    client: str, cfg: dict[str, Any], replay_dir: Path, budget_usd: float | None
) -> VLMClient:
    if client == "mock":
        return MockClient({m["slot"]: m for m in cfg["mock_slots"]})
    slugs = {s["slot"]: s["slug"] for s in cfg["slots"]}
    if client == "replay":
        return ReplayClient(replay_dir, slugs)
    if client == "record-mock":  # offline fixture builder: mock answers stored in replay format
        params = {
            s["slot"]: dict(m, slot=s["slot"])
            for s, m in zip(cfg["slots"], cfg["mock_slots"], strict=False)
        }
        return ReplayRecorder(MockClient(params), replay_dir, slugs)
    from knotwork.bench.client_openrouter import OpenRouterClient

    b = cfg["budget"]
    budget = Budget(float(budget_usd or 0.0), float(b["stop_when_remaining_usd_below"]))
    return OpenRouterClient(cfg, replay_dir, os.environ.get("OPENROUTER_API_KEY", ""), budget)


def run_bench(data: Path, out: Path, models: Path, live: bool, client: str, replay_dir: Path,
              budget_usd: float | None, track: str, max_trials: int, seed: int, dry_run: bool,
              workers: int = 0) -> int:  # fmt: skip
    cfg = yaml.safe_load(Path(models).read_text(encoding="utf-8"))
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tracks = tracks_of(track)
    sizes = block_sizes(cfg, max_trials)
    if dry_run:
        per = calls_per_model(sizes, tracks, max_trials)
        est = preflight(cfg, {s["slot"]: per for s in cfg["slots"]}, budget_usd)
        est.update(
            {"max_trials": max_trials, "tracks": tracks, "block_sizes": sizes, "network_calls": 0}
        )
        (out / "preflight.json").write_text(
            json.dumps(est, indent=1, sort_keys=True), encoding="utf-8"
        )
        print(f"bench preflight: ${est['estimated_usd']:.2f} for {est['calls']} calls (no network)")
        return 0
    pool = load_pool(Path(data))
    chosen = select(pool, sizes, seed)
    derived = (
        derive(chosen, out / "derived", profile, workers)
        if "tangle" in tracks
        else {"pairs": [], "triples": []}
    )
    vlm = make_client(client, cfg, Path(replay_dir), budget_usd)
    mode = {"mock": "mock", "replay": "replay", "record-mock": "replay"}.get(client, "live")
    slots = cfg["mock_slots"] if client == "mock" else cfg["slots"]
    report: dict[str, Any] = {"mode": mode, "network_calls": 0, "cost_usd": 0.0, "stopped": None, "models": {},
                              "seed": seed, "max_trials": max_trials, "tracks": tracks, "block_sizes": sizes,
                              "pool": {"t1": len(pool.t1), "p1": len(pool.p1), "gaslight": len(pool.gaslight)},
                              "derived": {"ablation_pairs": len(derived["pairs"]), "twin_triples": len(derived["triples"])}}  # fmt: skip
    ledger_rows: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    for s in slots:
        runner = Runner(vlm, s["slot"], seed, ledger_rows, str(s.get("slug", s["slot"])))
        p_true = float(s["p_cross"]) if client == "mock" else None
        try:
            report["models"][s["slot"]] = run_model(
                runner, pool, chosen, derived, tracks, max_trials, seed, p_true, profile
            )
        except BudgetExhausted as exc:
            report["stopped"] = "budget"
            report["stop_detail"] = str(exc)
            trials += runner.log
            break
        trials += runner.log
    report["network_calls"] = int(vlm.network_calls)
    report["cost_usd"] = round(sum(t["cost_usd"] for t in trials), 6)
    if ledger_rows:
        from knotwork.freshness import ledger

        ledger.append(profile, ledger_rows)
    (out / "trials.jsonl").write_text(
        "".join(json.dumps(t, sort_keys=True) + "\n" for t in trials), encoding="utf-8"
    )
    (out / "bench_report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True), encoding="utf-8"
    )
    print(f"bench: mode={mode} models={list(report['models'])} calls={report['network_calls']} "
          f"cost=${report['cost_usd']:.4f} stopped={report['stopped']}")  # fmt: skip
    return 0


__all__ = ["load_pool", "run_bench"]
