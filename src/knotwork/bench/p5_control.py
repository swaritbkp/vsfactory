"""True-premise control for the `T_GASLIGHT` probe (additive block, SPEC D-37).

For every false-premise item the benchmark asked, the same scene is re-certified with one change:
the claimed zone is the zone the traced gauge is really in. The result is a `P5_PREMISE_CONTROL`
question on pixel-identical input, so each pair differs in exactly one word of the prompt. A pair is
used only when the source bundle re-verifies from its files, the control is `CERTIFIED`, the kernel's
own ZONE step returns the claimed zone and the re-rendered pixels equal the source pixels.

``python -m knotwork.bench.p5_control`` replays by default; ``--live`` goes through the same triple
lock as ``knot bench --live`` and refuses a budget above ``MAX_BUDGET_USD``. Nothing in
``samples/live_run`` or the main benchmark output is written, and only a live run may replace a
live control record (each mode has its own default output directory).
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from knotwork.bench import stats
from knotwork.bench.budget import preflight
from knotwork.bench.client_base import BudgetExhausted
from knotwork.bench.client_replay import replay_key, replay_path
from knotwork.bench.prompts import image_id
from knotwork.bench.run import PROFILE, Item, Runner, load_pool, make_client
from knotwork.knotcore.contracts import Family, QuestionProgram
from knotwork.knotcore.grading import YESNO_RE, extract, grade_prediction
from knotwork.render.canvas import REPO_ROOT

BLOCK = "p5_control"
FALSE_BLOCK = "gaslight"
N_ITEMS = 24
SEED = 0
MAX_BUDGET_USD = 0.50
TRIALS_FILE = "p5_control_trials.jsonl"
REPORT_FILE = "p5_control.json"
SOURCE_TRIALS = REPO_ROOT / "samples" / "live_run" / "trials.jsonl"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def source_keys(trials: list[dict[str, Any]]) -> list[str]:
    """Keys of the false-premise items in the order asked; every model must have seen the same."""
    per_slot: dict[str, list[str]] = {}
    for r in trials:
        if r["block"] == FALSE_BLOCK:
            per_slot.setdefault(r["slot"], []).append(r["key"])
    lists = list(per_slot.values())
    if not lists or any(x != lists[0] for x in lists):
        raise ValueError("the false-premise block is missing or differs between models")
    return lists[0]


def control_job(job: dict[str, Any]) -> dict[str, Any]:
    """Certify the true-premise counterpart of one `T_GASLIGHT` bundle (picklable)."""
    from knotwork.diversity import rebuild_scene
    from knotwork.pipeline.certify import run_candidate
    from knotwork.pipeline.verify import verify_bundle
    from knotwork.plugins.gauge.scale import dial_zone
    from knotwork.plugins.panel.plugin import PLUGIN, target_gauge
    from knotwork.plugins.registry import get_plugin

    ds, bdir, profile = Path(job["ds"]), Path(job["dir"]), job["profile"]
    task = json.loads((bdir / "task.json").read_text(encoding="utf-8"))
    if task["family"] != Family.T_GASLIGHT.value:
        raise ValueError(f"{task['bundle_id']} is not a T_GASLIGHT bundle")
    source_ok = (
        task["certificate"]["verdict"] == "CERTIFIED"
        and verify_bundle(bdir, profile["kernel"]) is None
    )
    original = rebuild_scene(ds, task, profile)
    pay = copy.deepcopy(original.payload)
    gauge = target_gauge(pay)
    zone, false_zone = dial_zone(gauge), str(pay["claimed"])
    pay["claimed"] = zone
    scene = original.model_copy(update={"payload": pay})
    base = QuestionProgram.model_validate(task["question"])
    q = PLUGIN.twin_question(scene, base).model_copy(
        update={"program_id": f"{base.program_id}/p5-control"}
    )
    cand = run_candidate(get_plugin("panel"), scene, q, Path(job["out"]), profile)
    cert = cand.certificate
    steps = {s.op.value: s.output for s in cert.steps}
    source_pixels = image_id(bdir / "beauty.png")
    checks = {
        "source_certified": source_ok,
        "control_certified": cand.certified,
        "family_is_p5": q.family is Family.P5_PREMISE_CONTROL,
        "false_premise_was_false": false_zone != zone
        and task["answer"]["value"]["premise_holds"] is False,
        "kernel_zone_is_claimed": str(steps.get("ZONE")) == str(zone),
        "kernel_premise_holds": cert.d3.canonical.startswith("yes;"),
        "same_pixels": image_id(cand.dir / "beauty.png") == source_pixels,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "source_bundle_id": task["bundle_id"],
        "control_bundle_id": cand.bundle_id,
        "scene_id": task["scene_id"],
        "socket": str(pay["target"]),
        "claimed_zone_true": str(zone),
        "claimed_zone_false": false_zone,
        "kernel_zone": str(steps.get("ZONE")),
        "kernel_answer": cert.d3.canonical,
        "kernel_sha256": cert.kernel_sha256,
        "image_id": source_pixels,
        "question": q.text,
        "answer": cand.answer.model_dump(mode="json"),
        "span": float(q.difficulty.get("span", 80.0)),
    }


def build_control(data: Path, work: Path, keys: Sequence[str], profile: dict[str, Any],
                  workers: int = 0) -> tuple[list[Item], list[dict[str, Any]]]:  # fmt: skip
    """The control items (same images as the false-premise ones) and their certification records."""
    from knotwork.pipeline.generate import run_jobs

    pool = {it.key: it for it in load_pool(data).gaslight}
    missing = [k for k in keys if k not in pool]
    if missing:
        raise ValueError(f"{len(missing)} false-premise items are not in {data}: {missing[:3]}")
    sources = [pool[k] for k in keys]
    jobs = [{"ds": str(it.beauty.parent.parent.parent), "dir": str(it.beauty.parent),
             "out": str(work / it.bundle_id), "profile": profile} for it in sources]  # fmt: skip
    records = run_jobs(jobs, control_job, workers)
    items = [
        Item(
            key=f"p5:{src.bundle_id}",
            beauty=src.beauty,
            prompt=rec["question"],
            answer=rec["answer"],
            bundle_id=rec["control_bundle_id"],
            scene_id=rec["scene_id"],
            meta={
                "kind": "premise",
                "holds": True,
                "value": float(rec["answer"]["value"]["value"]),
                "span": rec["span"],
            },
        )  # fmt: skip
        for src, rec in zip(sources, records, strict=True)
    ]
    return items, records


# ------------------------------------------------------------------ analysis
def premise_token(reply: str | None) -> str | None:
    m = YESNO_RE.search(extract(reply)) if reply else None
    return m.group(1).lower() if m else None


def reading_correct(reply: str | None, value: float, tolerance: float) -> bool:
    """The numeric part of a premise reply alone, graded within the item's tolerance."""
    if not reply:
        return False
    text = extract(reply)
    m = YESNO_RE.search(text)
    truth = {"type": "float", "value": value, "tolerance": tolerance}
    return grade_prediction(text[m.end() :] if m else text, truth) == 1.0


def analyze(false_rows: list[dict[str, Any]], true_rows: list[dict[str, Any]],
            items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:  # fmt: skip
    """Per slot, over the matched pairs both arms answered: yes rates, tests, reading accuracy."""
    truth = {i["source_bundle_id"]: i["answer"] for i in items}
    arms: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for arm, rows, block in (("false", false_rows, FALSE_BLOCK), ("true", true_rows, BLOCK)):
        for r in rows:
            if r["block"] == block and not r["missing"]:
                bid = str(r["key"]).split(":", 1)[1]
                arms.setdefault(r["slot"], {}).setdefault(bid, {})[arm] = r
    out = {}
    for slot, by_item in sorted(arms.items()):
        pairs = [(bid, d) for bid, d in sorted(by_item.items()) if len(d) == 2 and bid in truth]
        if not pairs:
            continue
        res = stats.premise_control(
            [(premise_token(d["true"]["reply"]), premise_token(d["false"]["reply"])) for _, d in pairs]
        )  # fmt: skip
        for arm in ("true", "false"):
            reads = [reading_correct(d[arm]["reply"], float(truth[b]["value"]["value"]),
                                     float(truth[b]["tolerance"])) for b, d in pairs]  # fmt: skip
            res[f"{arm}_reading_correct"] = sum(reads)
            res[f"{arm}_fully_correct"] = sum(bool(d[arm]["correct"]) for _, d in pairs)
        out[slot] = res
    return out


# ------------------------------------------------------------------ run
def estimate(cfg: dict[str, Any], budget_usd: float | None,
             source: list[dict[str, Any]]) -> dict[str, Any]:  # fmt: skip
    """The standard preflight formula, plus the cost the same slots actually paid for the
    false-premise block (same images and prompt length)."""
    est = preflight(cfg, {s["slot"]: N_ITEMS for s in cfg["slots"]}, budget_usd)
    observed = {}
    for s in cfg["slots"]:
        paid = [
            r["cost_usd"] for r in source if r["slot"] == s["slot"] and r["block"] == FALSE_BLOCK
        ]
        observed[s["slot"]] = round(sum(paid) / len(paid) * N_ITEMS, 4) if paid else None
    est["observed_per_slot"] = observed
    est["observed_usd"] = round(sum(v or 0.0 for v in observed.values()), 4)
    est["max_budget_usd"] = MAX_BUDGET_USD
    return est


def _recorded(replay_dir: Path, slot: str, slug: str, item: Item) -> dict[str, Any]:
    """Cost and serving model of a cached response (survives resumed runs)."""
    key = replay_key(slot, slug, item.prompt, image_id(item.beauty), SEED)
    path = replay_path(replay_dir, slot, key)
    if not path.is_file():
        return {}
    entry = json.loads(path.read_text(encoding="utf-8"))
    response = entry.get("response") or {}
    return {"cost_usd": float(entry.get("cost_usd") or 0.0), "served_model": response.get("model"),
            "provider": response.get("provider"), "request_sha256": key}  # fmt: skip


def run_control(data: Path, out: Path, models: Path, replay_dir: Path, client: str,
                budget_usd: float | None, dry_run: bool, source_trials: Path = SOURCE_TRIALS,
                workers: int = 0) -> int:  # fmt: skip
    cfg = yaml.safe_load(models.read_text(encoding="utf-8"))
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    source = _jsonl(source_trials)
    keys = source_keys(source)
    previous = out / REPORT_FILE
    keeps_record = client != "openrouter" and not dry_run and previous.is_file()
    if keeps_record and json.loads(previous.read_text(encoding="utf-8")).get("mode") == "live":
        print(f"E_LIVE_RECORD: {previous.as_posix()} holds a live run; choose another --out.")
        return 4
    out.mkdir(parents=True, exist_ok=True)
    items, records = build_control(data, out / "bundles", keys, profile, workers)
    bad = [r for r in records if not r["ok"]]
    est = estimate(cfg, budget_usd, source)
    (out / "preflight.json").write_text(
        json.dumps(dict(est, items=records), indent=1, sort_keys=True), encoding="utf-8"
    )
    print(f"p5 control: {len(records) - len(bad)}/{len(records)} matched pairs certified; preflight "
          f"${est['estimated_usd']:.4f} (formula), ${est['observed_usd']:.4f} (observed) for "
          f"{est['calls']} calls")  # fmt: skip
    if bad or len(records) != N_ITEMS:
        for r in bad:
            print(
                f"  failed {r['source_bundle_id']}: {[k for k, v in r['checks'].items() if not v]}"
            )
        return 1
    if dry_run:
        return 0
    vlm = make_client(client, cfg, replay_dir, budget_usd)
    mode = {"mock": "mock", "replay": "replay"}.get(client, "live")
    slots = cfg["mock_slots"] if client == "mock" else cfg["slots"]
    trials: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    stopped = None
    for s in slots:
        slot, slug = s["slot"], str(s.get("slug", s["slot"]))
        runner = Runner(vlm, slot, SEED, ledger_rows, slug)
        try:
            runner.ask_many(items, BLOCK)
        except BudgetExhausted as exc:
            stopped = f"budget: {exc}"
        for item, row in zip(items, runner.log, strict=False):
            if mode != "mock" and not row["missing"]:
                row.update(_recorded(replay_dir, slot, slug, item))
        trials += runner.log
        if stopped:
            break
    if ledger_rows:
        from knotwork.freshness import ledger

        ledger.append(profile, ledger_rows)
    missing = sum(1 for t in trials if t["missing"])
    report = {
        "block": BLOCK, "mode": mode, "seed": SEED, "stopped": stopped,
        "network_calls": int(vlm.network_calls), "trials": len(trials), "missing": missing,
        "errors": sum(1 for t in trials if t.get("error")),
        "cost_usd": round(sum(t["cost_usd"] for t in trials), 6),
        "slugs": {s["slot"]: str(s.get("slug", s["slot"])) for s in slots},
        "models": analyze(source, trials, records), "items": records,
    }  # fmt: skip
    (out / TRIALS_FILE).write_text(
        "".join(json.dumps(t, sort_keys=True) + "\n" for t in trials),
        encoding="utf-8", newline="\n",
    )  # fmt: skip
    (out / REPORT_FILE).write_text(
        json.dumps(report, indent=1, sort_keys=True), encoding="utf-8", newline="\n"
    )
    print(f"p5 control: mode={mode} trials={len(trials)} missing={missing} "
          f"calls={report['network_calls']} recorded_cost=${report['cost_usd']:.4f} stopped={stopped}")  # fmt: skip
    return 0 if stopped is None and missing == 0 else 1


def main(argv: list[str] | None = None) -> int:
    from knotwork.cli import SafetyRefusal, guard_live

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("artifacts/phase3/bench_split"))
    ap.add_argument("--out", type=Path, default=None, help="default: one directory per mode")
    ap.add_argument("--models", type=Path, default=Path("configs/models.yaml"))
    ap.add_argument("--replay-dir", type=Path, default=Path("bench_replay"))
    ap.add_argument("--client", choices=("replay", "mock"), default="replay")
    ap.add_argument("--live", action="store_true", help="spends real money")
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true", help="certify and estimate only")
    a = ap.parse_args(argv)
    try:
        if a.live and a.budget_usd is not None and a.budget_usd > MAX_BUDGET_USD:
            raise SafetyRefusal(
                f"E_BUDGET_OVER_CAP: --budget-usd {a.budget_usd:.2f} exceeds {MAX_BUDGET_USD:.2f}."
            )
        guard_live(a.live, a.budget_usd, a.models, a.dry_run)
    except SafetyRefusal as exc:
        print(exc.message)
        return exc.exit_code
    suffix = "" if a.live else f"_{a.client}"
    out = a.out or Path("artifacts/phase5") / f"p5_control{suffix}"
    return run_control(a.data, out, a.models, a.replay_dir, "openrouter" if a.live else a.client,
                       a.budget_usd, a.dry_run)  # fmt: skip


if __name__ == "__main__":
    raise SystemExit(main())
