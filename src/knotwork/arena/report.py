"""Model Arena reports: a deterministic statistical engine and an HTML / Markdown / JSON / CSV / ZIP
exporter, plus an optional AI executive brief grounded only in the computed metrics.

Layer 1 (`compute_metrics`) is pure and offline: given the same session files it returns the same
numbers. Every interval is a 95 % Wilson score interval; contrasts between blocks a model saw are a
pooled two-proportion z test and Fisher's exact test (two-sided). Layer 2 (`ai_brief`) is opt-in,
runs only when a person asks for it, and is shown in a labelled callout.
"""

from __future__ import annotations

import base64
import csv
import html
import io
import json
import math
import platform
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from knotwork import net
from knotwork.arena import storage
from knotwork.arena.schemas import (
    ItemRecord,
    RunStatus,
    SessionConfig,
    TrialRecord,
    find_secret,
)

Z95 = 1.959963984540054
FOOTER = (
    "© 2026 BilotaAI · Team Bilota (Swarit Sharma & Pankaj David) · Knotwork Visual Task Factory"
)
COLORS = ("#047857", "#B45309", "#BE123C", "#1D4ED8", "#7C3AED", "#0E7490", "#A16207", "#4D7C0F")
CATEGORIES = (
    "wrong_port_or_strand",
    "numeric_tolerance_miss",
    "format_violation",
    "false_premise_error",
    "refusal_or_empty",
    "api_error",
)
GALLERY_MAX = 6
THUMB_PX = 220
BRIEF_TITLE = "AI-Assisted Executive Synthesis (Grounded in Deterministic Metrics)"
#: pass@k minus pass^k at or above this is reported as a material reliability gap
RELIABILITY_GAP = 0.10
#: Methodology & Literature References printed at the foot of report.html / report.md
REFERENCES: tuple[str, ...] = (
    "Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. "
    "JASA 22(158), 209-212. https://doi.org/10.1080/01621459.1927.10502953. Used for every 95% "
    "binomial score interval in this report.",
    "Roberts, J. et al. ZeroBench: An Impossible Visual Benchmark for Contemporary Large "
    "Multimodal Models. ICML 2026, arXiv:2502.09696 (https://arxiv.org/abs/2502.09696). Source of "
    "the pass@k / pass^k "
    "reliability protocol used when a session asks each item k > 1 times.",
    "Rahmanzadehgervi, P. et al. Vision language models are blind. ACCV 2024, arXiv:2407.06581. "
    "VLMs fail simple 2-D geometric tasks such as line intersections and path following.",
    "Friedman, D. and Dieng, A. B. The Vendi Score: A Diversity Evaluation Metric for Machine "
    "Learning. TMLR 2023, arXiv:2210.02410. Knotwork reports Vendi only over kernel-certified items.",
    "Lee, K. et al. VLind-Bench: Measuring Language Priors in Large Vision-Language Models. "
    "Findings of NAACL 2025, arXiv:2406.08702. Counterfactual tests of language priors.",
    "Sun, J. et al. Seeing What Is Actually There: PriVE-Bench and PriVE-Tools for Counterfactual "
    "Evaluation of Agentic Visual Evidence in VLMs. EMNLP 2026 (2026.emnlp-main.760).",
    "Feng, Y. et al. VisualSphinx: Large-Scale Synthetic Vision Logic Puzzles for RL. "
    "arXiv:2505.23977 (2025). Synthetic puzzles filtered with LLM / VLM scoring.",
)
CSV_FIELDS = (
    "trial_id", "block_id", "domain", "preset", "family", "item_index", "bundle_id", "model",
    "repeat_idx",
    "status", "correct", "truth", "extracted", "error_category", "latency_ms", "prompt_tokens",
    "completion_tokens", "cost_usd", "attempts", "simulated", "ts_utc",
)  # fmt: skip


# ------------------------------------------------------------------ statistics
def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95 % Wilson score interval for k successes in n trials ((0, 1) when n is 0)."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """(z, two-sided p) of the pooled two-proportion z test; (0, 1) when undefined."""
    if n1 <= 0 or n2 <= 0:
        return 0.0, 1.0
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (k1 / n1 - k2 / n2) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_exact(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact p for the 2x2 table [[k1, n1-k1], [k2, n2-k2]] (sum of the
    probabilities of every table no more likely than the observed one)."""
    if n1 <= 0 or n2 <= 0:
        return 1.0
    total, successes = n1 + n2, k1 + k2
    lo, hi = max(0, successes - n2), min(successes, n1)

    def logp(x: int) -> float:
        return _log_comb(n1, x) + _log_comb(n2, successes - x) - _log_comb(total, successes)

    observed = logp(k1)
    p = sum(math.exp(logp(x)) for x in range(lo, hi + 1) if logp(x) <= observed + 1e-7)
    return min(1.0, p)


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (numpy's default), None for no values."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summary(values: list[float]) -> dict[str, float | None]:
    mean = sum(values) / len(values) if values else None
    return {"mean": mean, "p50": percentile(values, 50), "p95": percentile(values, 95)}


# ------------------------------------------------------------------ metrics
def _scored(trials: dict[str, TrialRecord]) -> list[TrialRecord]:
    return [t for t in trials.values() if t.status == "ok"]


def _cell(group: list[TrialRecord], errors: int = 0) -> dict[str, Any]:
    n = len(group)
    k = sum(bool(t.correct) for t in group)
    lo, hi = wilson(k, n)
    cost = sum(t.cost_usd for t in group)
    lat = [t.latency_ms for t in group]
    return {
        "n": n, "correct": k, "accuracy": (k / n) if n else None, "ci_lo": lo, "ci_hi": hi,
        "latency_p50_ms": percentile(lat, 50), "latency_p95_ms": percentile(lat, 95),
        "prompt_tokens": sum(t.prompt_tokens for t in group),
        "completion_tokens": sum(t.completion_tokens for t in group),
        "cost_usd": cost, "cost_per_correct": (cost / k) if k else None, "api_errors": errors,
    }  # fmt: skip


def leaderboard(cfg: SessionConfig, trials: dict[str, TrialRecord]) -> dict[str, Any]:
    """Accuracy, CI, latency, tokens and cost per (block, model) and per model overall."""
    scored = _scored(trials)
    errors = Counter((t.block_id, t.model) for t in trials.values() if t.status == "api_error")
    cells = []
    for block in cfg.blocks:
        for model in cfg.models_for(block):
            group = [t for t in scored if t.block_id == block.block_id and t.model == model]
            row = _cell(group, errors[(block.block_id, model)])
            cells.append({"block_id": block.block_id, "block": block.label, "model": model, **row})
    models = list(dict.fromkeys(m for b in cfg.blocks for m in cfg.models_for(b)))
    overall = []
    for model in models:
        group = [t for t in scored if t.model == model]
        errs = sum(v for (_, m), v in errors.items() if m == model)
        overall.append({"model": model, **_cell(group, errs)})
    overall.sort(key=lambda r: (-(r["accuracy"] or 0.0), r["model"]))
    return {"cells": cells, "models": overall}


def pass_at_k(outcomes: dict[int, list[bool]], k: int) -> dict[str, Any]:
    """pass@1, pass@k and pass^k over items that have all `k` repeats scored (ZeroBench).

    `outcomes` maps an item index to the correctness of each scored repeat. pass@1 is the mean
    single-trial accuracy, pass@k the share of items solved at least once, pass^k the share solved
    on every repeat. Items with fewer than `k` scored repeats are left out (`items_partial`).
    """
    full = [v[:k] for v in outcomes.values() if len(v) >= k]
    n = len(full)
    flat = [x for v in full for x in v]
    at_k = sum(any(v) for v in full)
    hat_k = sum(all(v) for v in full)
    return {
        "k": k, "items": n, "items_partial": len(outcomes) - n,
        "pass_at_1": (sum(flat) / len(flat)) if flat else None,
        "pass_at_k": (at_k / n) if n else None, "pass_hat_k": (hat_k / n) if n else None,
        "solved_any": at_k, "solved_all": hat_k,
    }  # fmt: skip


def reliability(cfg: SessionConfig, trials: dict[str, TrialRecord]) -> dict[str, Any] | None:
    """pass@1 / pass@k / pass^k per (block, model) when the session asks each item k > 1 times,
    plus one deterministic finding per cell whose pass@k - pass^k gap is material."""
    if cfg.repeats <= 1:
        return None
    k = cfg.repeats
    ordered = sorted(trials.values(), key=lambda t: (t.item_index, t.repeat_idx))
    cells, findings = [], []
    for block in cfg.blocks:
        for model in cfg.models_for(block):
            outcomes: dict[int, list[bool]] = {}
            for t in ordered:
                if t.block_id == block.block_id and t.model == model and t.status == "ok":
                    outcomes.setdefault(t.item_index, []).append(bool(t.correct))
            row: dict[str, Any] = {"block_id": block.block_id, "block": block.label,
                                   "model": model, **pass_at_k(outcomes, k)}  # fmt: skip
            gap = None
            if row["pass_at_k"] is not None and row["pass_hat_k"] is not None:
                gap = row["pass_at_k"] - row["pass_hat_k"]
            row["gap"] = gap
            cells.append(row)
            if gap is not None and gap >= RELIABILITY_GAP:
                findings.append(
                    f"{short(model)} on {block.label}: solves {pct(row['pass_at_k'])} of items at "
                    f"least once in {k} tries but only {pct(row['pass_hat_k'])} on all {k} "
                    f"(gap {100 * gap:.1f} pp). Its answers are not consistent across repeats, "
                    f"so single-run accuracy ({pct(row['pass_at_1'])}) overstates its reliability."
                )
    return {"k": k, "gap_threshold": RELIABILITY_GAP, "cells": cells, "findings": findings}


def contrasts(cfg: SessionConfig, trials: dict[str, TrialRecord]) -> list[dict[str, Any]]:
    """Every pair of blocks evaluated by the same model: effect size and both tests."""
    scored = _scored(trials)
    tally: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for t in scored:
        cell = tally[(t.block_id, t.model)]
        cell[0] += int(bool(t.correct))
        cell[1] += 1
    out = []
    for i, a in enumerate(cfg.blocks):
        for b in cfg.blocks[i + 1 :]:
            for model in [m for m in cfg.models_for(a) if m in cfg.models_for(b)]:
                k1, n1 = tally[(a.block_id, model)]
                k2, n2 = tally[(b.block_id, model)]
                if not n1 or not n2:
                    continue
                z, p_z = two_proportion_z(k1, n1, k2, n2)
                out.append({
                    "model": model, "block_a": a.label, "block_b": b.label,
                    "block_a_id": a.block_id, "block_b_id": b.block_id,
                    "acc_a": k1 / n1, "acc_b": k2 / n2, "n_a": n1, "n_b": n2,
                    "diff_pp": 100 * (k1 / n1 - k2 / n2), "z": z, "p_z": p_z,
                    "p_fisher": fisher_exact(k1, n1, k2, n2),
                })  # fmt: skip
    return out


def taxonomy(
    cfg: SessionConfig, items: dict[str, ItemRecord], trials: dict[str, TrialRecord]
) -> dict[str, Any]:
    """Failure counts by category per model, each model's dominant failure mode and its hardest
    (block, question family) bucket."""
    per_model: dict[str, Counter[str]] = defaultdict(Counter)
    buckets: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for t in trials.values():
        if t.error_category:
            per_model[t.model][t.error_category] += 1
        if t.status != "ok":
            continue
        item = items.get(f"{t.block_id}:{t.item_index}")
        family = item.family if item else "?"
        cell = buckets[t.model][f"{cfg.block(t.block_id).label} · {family}"]
        cell[0] += int(bool(t.correct))
        cell[1] += 1
    models = {}
    for model in sorted({t.model for t in trials.values()}):
        counts = {c: per_model[model].get(c, 0) for c in CATEGORIES}
        failures = sum(counts.values())
        dominant = max(CATEGORIES, key=lambda c: counts[c]) if failures else None
        rated = [(k / n, name, n) for name, (k, n) in buckets[model].items() if n]
        hardest = min(rated) if rated else None
        models[model] = {
            "counts": counts, "failures": failures, "dominant": dominant,
            "hardest_bucket": hardest[1] if hardest else None,
            "hardest_accuracy": hardest[0] if hardest else None,
            "hardest_n": hardest[2] if hardest else 0,
        }  # fmt: skip
    return {"categories": list(CATEGORIES), "models": models}


def factory_telemetry(cfg: SessionConfig, items: list[ItemRecord]) -> dict[str, Any]:
    """Generation and kernel-verification timing, yield and throughput per block and overall."""

    def block_stats(group: list[ItemRecord]) -> dict[str, Any]:
        certified = sum(i.certified for i in group)
        busy_ms = sum(i.gen_ms + i.verify_ms for i in group)
        return {
            "items": len(group), "certified": certified, "quarantined": len(group) - certified,
            "yield": (certified / len(group)) if group else None,
            "gen_ms": summary([i.gen_ms for i in group]),
            "verify_ms": summary([i.verify_ms for i in group]),
            "certified_per_min": (certified / (busy_ms / 60000.0)) if busy_ms else None,
        }  # fmt: skip

    blocks = []
    for block in cfg.blocks:
        group = [i for i in items if i.block_id == block.block_id]
        blocks.append({"block_id": block.block_id, "block": block.label, **block_stats(group)})
    return {"blocks": blocks, "overall": block_stats(items)}


def unit_economics(items: list[ItemRecord], vcpu_usd_hr: float | None = None) -> dict[str, Any]:
    """Single-core certified items / hour and projected $ per 1M certified samples from the
    session's own observed gen_ms + verify_ms (every attempted item's time is charged)."""
    from knotwork.posttrain.throughput import DEFAULT_VCPU_USD_HR
    from knotwork.posttrain.throughput import unit_economics as economics

    return economics([i.gen_ms for i in items], [i.verify_ms for i in items],
                     sum(i.certified for i in items), None,
                     DEFAULT_VCPU_USD_HR if vcpu_usd_hr is None else vcpu_usd_hr)  # fmt: skip


def _econ_rows(e: dict[str, Any]) -> list[list[Any]]:
    rate, cost = e["certified_items_per_hour"], e["usd_per_1m_certified"]
    return [
        ["Items attempted / certified", f"{e['attempted']:,} / {e['certified']:,}"],
        ["Busy time (gen + verify)", f"{e['busy_s']:,.1f} s"],
        ["Seconds per certified item", "—" if e["s_per_certified"] is None
         else f"{e['s_per_certified']:.3f} s"],
        ["Certified items / hour (1 vCPU)", "—" if rate is None else f"{rate:,.0f}"],
        [f"USD per 1M certified samples @ ${e['vcpu_usd_per_hour']}/vCPU-hr",
         "—" if cost is None else f"${cost:,.2f}"],
        ["Formula", e["formula"]],
    ]  # fmt: skip


def regression(
    a_cfg: SessionConfig,
    a_trials: dict[str, TrialRecord],
    b_cfg: SessionConfig,
    b_trials: dict[str, TrialRecord],
) -> dict[str, Any]:
    """Session A vs B on every (block label, model) both evaluated: accuracy, latency, cost deltas."""
    a_lb, b_lb = leaderboard(a_cfg, a_trials), leaderboard(b_cfg, b_trials)
    b_cells = {(c["block"], c["model"]): c for c in b_lb["cells"]}
    rows = []
    for c in a_lb["cells"]:
        other = b_cells.get((c["block"], c["model"]))
        if other is None or not c["n"] or not other["n"]:
            continue
        z, p = two_proportion_z(c["correct"], c["n"], other["correct"], other["n"])
        rows.append({
            "block": c["block"], "model": c["model"], "acc_a": c["accuracy"],
            "acc_b": other["accuracy"], "delta_pp": 100 * (c["accuracy"] - other["accuracy"]),
            "p_z": p, "latency_delta_ms": _delta(c["latency_p50_ms"], other["latency_p50_ms"]),
            "cost_delta_usd": c["cost_usd"] - other["cost_usd"],
            "verdict": "improved" if p < 0.05 and z > 0 else "regressed" if p < 0.05 else "no change",
        })  # fmt: skip
    return {"session_a": a_cfg.session_name, "session_b": b_cfg.session_name, "rows": rows}


def _delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def _duration(status: RunStatus) -> float | None:
    try:
        start = datetime.fromisoformat(status.started_utc.replace("Z", "+00:00"))
        end_text = status.finished_utc or status.updated_utc
        end = datetime.fromisoformat(end_text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (end - start).total_seconds())


def provenance() -> dict[str, str]:
    """git commit and the trusted kernel's source hash (best effort, never raises)."""
    out = {"git_sha": "unknown", "kernel_sha256": "unknown"}
    try:
        from knotwork.export.manifest import git_state
        from knotwork.knotcore import engine

        out["git_sha"] = str(git_state()["sha"])
        out["kernel_sha256"] = engine.kernel_sha256()
    except Exception:  # provenance is informative; a missing git checkout must not stop a report
        pass
    return out


def compute_metrics(
    cfg: SessionConfig,
    items: list[ItemRecord],
    trials: dict[str, TrialRecord],
    status: RunStatus,
    compare: tuple[SessionConfig, dict[str, TrialRecord]] | None = None,
    prov: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Every number in the report, as one JSON-ready dict (report_metrics.json)."""
    by_key = {i.key: i for i in items}
    wanted = {b.block_id for b in cfg.blocks}
    mine = [i for i in items if i.block_id in wanted]
    prov = {
        "git_sha": "unknown",
        "kernel_sha256": "unknown",
        **(prov if prov is not None else provenance()),
    }
    scored = _scored(trials)
    header = {
        "title": f"Model Arena Report — {cfg.session_name}",
        "session_name": cfg.session_name, "slug": cfg.slug, "client": cfg.client,
        "evaluator": cfg.evaluator, "notes": cfg.notes, "run_id": cfg.run_id, "mode": cfg.mode,
        "simulated": cfg.mode == "mock", "state": status.state,
        "started_utc": status.started_utc, "finished_utc": status.finished_utc,
        "duration_s": _duration(status), **prov,
        "dataset_sha256": storage.dataset_fingerprint(mine),
        "data_dir": str(storage.resolve(cfg.data_dir)),
        "results_dir": str(storage.resolve(cfg.results_dir)),
        "blocks": [{"block_id": b.block_id, "label": b.label, "domain": b.domain,
                    "preset": b.preset, "n": b.n, "seed_range": storage.seed_range(b),
                    "models": cfg.models_for(b)} for b in cfg.blocks],
        "python": sys.version.split()[0], "platform": platform.platform(),
        "cli": storage.cli_command(storage.resolve(cfg.results_dir)),
        "temperature": cfg.temperature, "repeats": cfg.repeats,
    }  # fmt: skip
    totals = {
        "trials_planned": cfg.total_trials, "trials_scored": len(scored),
        "api_errors": sum(t.status == "api_error" for t in trials.values()),
        "spend_usd": sum(t.cost_usd for t in trials.values()),
        "mean_latency_ms": summary([t.latency_ms for t in scored])["mean"],
    }  # fmt: skip
    out: dict[str, Any] = {
        "schema": "knotwork.arena.report/1",
        "header": header,
        "totals": totals,
        "factory": factory_telemetry(cfg, mine),
        "unit_economics": unit_economics(mine),
        "leaderboard": leaderboard(cfg, trials),
        "contrasts": contrasts(cfg, trials),
        "taxonomy": taxonomy(cfg, by_key, trials),
        "gallery": gallery_rows(cfg, by_key, trials),
        "references": list(REFERENCES),
    }
    rel = reliability(cfg, trials)
    if rel is not None:
        out["reliability"] = rel
    if compare is not None:
        out["regression"] = regression(cfg, trials, compare[0], compare[1])
    return out


def gallery_rows(
    cfg: SessionConfig, items: dict[str, ItemRecord], trials: dict[str, TrialRecord]
) -> list[dict[str, Any]]:
    """Representative failures: one per (model, category) first, up to GALLERY_MAX."""
    seen: set[tuple[str, str]] = set()
    rows = []
    for t in sorted(trials.values(), key=lambda t: t.trial_id):
        if t.status != "ok" or t.correct or not t.error_category:
            continue
        key = (t.model, t.error_category)
        item = items.get(f"{t.block_id}:{t.item_index}")
        if key in seen or item is None or not item.dir:
            continue
        seen.add(key)
        rows.append({
            "trial_id": t.trial_id, "bundle_id": t.bundle_id, "block": cfg.block(t.block_id).label,
            "model": t.model, "truth": t.truth, "prediction": t.extracted or "(none)",
            "category": t.error_category, "image": item.dir + "/beauty.png",
        })  # fmt: skip
        if len(rows) >= GALLERY_MAX:
            break
    return rows


# ------------------------------------------------------------------ formatting helpers
def pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{100 * x:.{digits}f}%"


def usd(x: float | None) -> str:
    if x is None:
        return "—"
    return f"${x:,.4f}" if abs(x) < 1 else f"${x:,.2f}"


def ms(x: float | None) -> str:
    return "—" if x is None else f"{x:,.0f} ms"


def pval(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def short(model: str) -> str:
    return model.split("/", 1)[-1]


def _esc(x: Any) -> str:
    return html.escape(str(x))


# ------------------------------------------------------------------ inline SVG charts
def svg_accuracy(metrics: dict[str, Any]) -> str:
    """Dot-and-whisker chart: accuracy with its 95 % CI per (block, model)."""
    cells = [c for c in metrics["leaderboard"]["cells"] if c["n"]]
    if not cells:
        return ""
    models = list(dict.fromkeys(c["model"] for c in cells))
    color = {m: COLORS[i % len(COLORS)] for i, m in enumerate(models)}
    row_h, left, width, top = 26, 260, 420, 28
    height = top + row_h * len(cells) + 34

    def x(v: float) -> float:
        return left + v * width

    parts = [f'<svg viewBox="0 0 {left + width + 30} {height}" role="img" '
             f'aria-label="Accuracy by model and block with 95% confidence intervals" '
             f'xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui" '
             f'font-size="12">']  # fmt: skip
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{x(tick):.1f}" y1="{top - 8}" x2="{x(tick):.1f}" '
                     f'y2="{height - 26}" stroke="#E2E8F0"/>'
                     f'<text x="{x(tick):.1f}" y="{height - 10}" text-anchor="middle" '
                     f'fill="#64748B">{int(tick * 100)}%</text>')  # fmt: skip
    for i, c in enumerate(cells):
        y = top + i * row_h + row_h / 2
        label = f"{c['block']} · {short(c['model'])}"
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
                     f'fill="#0F172A">{_esc(label[:44])}</text>'
                     f'<line x1="{x(c["ci_lo"]):.1f}" y1="{y:.1f}" x2="{x(c["ci_hi"]):.1f}" '
                     f'y2="{y:.1f}" stroke="{color[c["model"]]}" stroke-width="2.5"/>'
                     f'<circle cx="{x(c["accuracy"]):.1f}" cy="{y:.1f}" r="5" '
                     f'fill="{color[c["model"]]}"><title>{_esc(label)}: {pct(c["accuracy"])} '
                     f'[{pct(c["ci_lo"])}, {pct(c["ci_hi"])}] n={c["n"]}</title></circle>')  # fmt: skip
    parts.append("</svg>")
    return "".join(parts)


def pareto_front(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Non-dominated points (lower cost, higher accuracy), sorted by cost."""
    front, best = [], -1.0
    for cost, acc, name in sorted(points, key=lambda p: (p[0], -p[1])):
        if acc > best:
            front.append((cost, acc, name))
            best = acc
    return front


def svg_pareto(metrics: dict[str, Any]) -> str:
    """Accuracy vs. cost per scored trial, per model, with the Pareto frontier drawn."""
    rows = [r for r in metrics["leaderboard"]["models"] if r["n"]]
    if not rows:
        return ""
    points = [(r["cost_usd"] / r["n"], r["accuracy"], r["model"]) for r in rows]
    top_cost = max(p[0] for p in points) or 1e-6
    left, width, height, top = 56, 520, 260, 20

    def x(v: float) -> float:
        return left + (v / (top_cost * 1.1)) * width

    def y(v: float) -> float:
        return top + (1 - v) * (height - 50)

    parts = [f'<svg viewBox="0 0 {left + width + 150} {height}" role="img" '
             f'aria-label="Accuracy versus cost per trial with Pareto frontier" '
             f'xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui" '
             f'font-size="12">']  # fmt: skip
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{left}" y1="{y(tick):.1f}" x2="{left + width}" '
                     f'y2="{y(tick):.1f}" stroke="#E2E8F0"/><text x="{left - 8}" '
                     f'y="{y(tick) + 4:.1f}" text-anchor="end" fill="#64748B">'
                     f'{int(tick * 100)}%</text>')  # fmt: skip
    parts.append(f'<text x="{left + width / 2}" y="{height - 6}" text-anchor="middle" '
                 f'fill="#64748B">cost per scored trial (USD, max {usd(top_cost)})</text>')  # fmt: skip
    front = pareto_front(points)
    if len(front) > 1:
        path = " ".join(f"{x(c):.1f},{y(a):.1f}" for c, a, _ in front)
        parts.append(f'<polyline points="{path}" fill="none" stroke="#059669" '
                     f'stroke-dasharray="5 4" stroke-width="1.5"/>')  # fmt: skip
    names = {p[2] for p in front}
    for i, (cost, acc, name) in enumerate(points):
        col = COLORS[i % len(COLORS)]
        ring = ' stroke="#059669" stroke-width="2.5"' if name in names else ""
        parts.append(f'<circle cx="{x(cost):.1f}" cy="{y(acc):.1f}" r="6" fill="{col}"{ring}>'
                     f'<title>{_esc(name)}: {pct(acc)} at {usd(cost)} per trial</title></circle>'
                     f'<text x="{x(cost) + 10:.1f}" y="{y(acc) + 4:.1f}" fill="#0F172A">'
                     f'{_esc(short(name)[:24])}</text>')  # fmt: skip
    parts.append("</svg>")
    return "".join(parts)


# ------------------------------------------------------------------ renderers
def thumbnail(path: Path, px: int = THUMB_PX) -> str:
    """A small inline JPEG data URI of an item image ('' when it is missing)."""
    if not path.is_file():
        return ""
    with Image.open(path) as im:
        small = im.convert("RGB").resize((px, px), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


CSS = """
:root{--ink:#0F172A;--mute:#64748B;--rule:#E2E8F0;--paper:#FCFCFA;--accent:#059669;--warn:#B45309}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);
font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1060px;margin:0 auto;padding:40px 36px 24px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}h2{font-size:17px;font-weight:650;
color:var(--ink);border-bottom:2px solid var(--accent);padding-bottom:6px;margin:36px 0 12px}.sub{color:var(--mute);margin:0 0 18px}
table{border-collapse:collapse;width:100%;font-size:12.5px}th{text-align:left;font-weight:600;
color:var(--mute);border-bottom:1px solid var(--rule);padding:6px 8px}td{padding:6px 8px;
border-bottom:1px solid var(--rule);font-variant-numeric:tabular-nums}
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}.kpi{border:1px solid var(--rule);
border-radius:8px;padding:10px 12px;background:#fff}.kpi b{display:block;font-size:20px;
font-variant-numeric:tabular-nums}.kpi span{color:var(--mute);font-size:12px}.meta{display:grid;grid-template-columns:180px 1fr;gap:2px 16px;font-size:12.5px}
.meta div:nth-child(odd){color:var(--mute)}code{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;
word-break:break-all}.callout{border-left:3px solid var(--accent);background:#ECFDF5;padding:12px 16px;
border-radius:4px}.warn{border-left-color:var(--warn);background:#FFFBEB}
.gallery{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.card{border:1px solid var(--rule);
border-radius:8px;padding:8px;background:#fff;font-size:12px}.card img{width:100%;border-radius:4px}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:11px;background:#F1F5F9}
.up{color:#047857}.down{color:#BE123C}footer{margin-top:40px;padding-top:12px;border-top:1px solid
var(--rule);color:var(--mute);font-size:12px}svg{max-width:100%;height:auto}
@media print{main{padding:0}h2{break-after:avoid}.card,.kpi,table{break-inside:avoid}}
"""


def _kpis(m: dict[str, Any]) -> list[tuple[str, str]]:
    best = next((r for r in m["leaderboard"]["models"] if r["n"]), None)
    f = m["factory"]["overall"]
    best_text = (
        f"{short(best['model'])} {pct(best['accuracy'])} [{pct(best['ci_lo'])}–{pct(best['ci_hi'])}]"
        if best
        else "—"
    )
    return [
        (
            "Completed trials",
            f"{m['totals']['trials_scored']:,} / {m['totals']['trials_planned']:,}",
        ),
        ("Certification yield", pct(f["yield"])),
        ("Best model (95% CI)", best_text),
        ("Total spend", usd(m["totals"]["spend_usd"])),
        ("Mean gen / verify", f"{ms(f['gen_ms']['mean'])} / {ms(f['verify_ms']['mean'])}"),
    ]


def render_html(m: dict[str, Any], data_dir: Path, brief: str | None = None) -> str:
    """The standalone report: no scripts, no external assets, prints cleanly to PDF."""
    h = m["header"]
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{_esc(h['title'])}"
           f"</title><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<style>{CSS}</style></head><body><main>"]  # fmt: skip
    client = f" · {_esc(h['client'])}" if h["client"] else ""
    out.append(f"<h1>{_esc(h['title'])}</h1><p class='sub'>Knotwork Visual Task Factory · "
               f"Model Arena{client} · run <code>{_esc(h['run_id'] or h['slug'])}</code></p>")  # fmt: skip
    if h["simulated"]:
        out.append("<p class='callout warn'><b>Simulated run.</b> Offline mock models answered "
                   "these items; costs are planning estimates, not charges.</p>")  # fmt: skip
    out.append("<div class='kpis'>" + "".join(
        f"<div class='kpi'><span>{_esc(k)}</span><b>{_esc(v)}</b></div>" for k, v in _kpis(m)
    ) + "</div>")  # fmt: skip
    if brief:
        out.append(f"<h2>{_esc(BRIEF_TITLE)}</h2><div class='callout'>"
                   + "".join(f"<p>{_esc(p)}</p>" for p in brief.split("\n\n") if p.strip())
                   + "</div>")  # fmt: skip
    out.append("<h2>Statistical leaderboard</h2>" + _html_table(
        ["Block", "Model", "Accuracy", "95% CI", "Correct / N", "p50 / p95 latency", "Tokens in / out",
         "Cost", "$ / correct"],
        [[c["block"], short(c["model"]), pct(c["accuracy"]),
          f"{pct(c['ci_lo'])} – {pct(c['ci_hi'])}", f"{c['correct']} / {c['n']}",
          f"{ms(c['latency_p50_ms'])} / {ms(c['latency_p95_ms'])}",
          f"{c['prompt_tokens']:,} / {c['completion_tokens']:,}", usd(c["cost_usd"]),
          usd(c["cost_per_correct"])] for c in m["leaderboard"]["cells"]]))  # fmt: skip
    if "reliability" in m:
        rel = m["reliability"]
        out.append(
            f"<h2>Reliability over k = {rel['k']} repeats (pass@k / pass^k)</h2>"
            + _html_table(_REL_HEAD, _rel_rows(rel))
            + "".join(f"<p class='callout warn'>{_esc(x)}</p>" for x in rel["findings"])
        )
    out.append(f"<h2>Accuracy by model and block (95% Wilson CI)</h2>{svg_accuracy(m)}")
    out.append(f"<h2>Accuracy vs. cost (Pareto frontier)</h2>{svg_pareto(m)}")
    if m["contrasts"]:
        out.append("<h2>Cross-block statistical contrasts</h2>" + _html_table(
            ["Model", "Block A", "Block B", "Acc A", "Acc B", "Effect (pp)", "z", "p (z)",
             "p (Fisher)"],
            [[short(c["model"]), c["block_a"], c["block_b"], pct(c["acc_a"]), pct(c["acc_b"]),
              f"{c['diff_pp']:+.1f}", f"{c['z']:+.2f}", pval(c["p_z"]), pval(c["p_fisher"])]
             for c in m["contrasts"]]))  # fmt: skip
    tax = m["taxonomy"]
    out.append("<h2>Error taxonomy and regimes</h2>" + _html_table(
        ["Model", *[c.replace("_", " ") for c in tax["categories"]], "Dominant", "Hardest bucket"],
        [[short(model), *[str(row["counts"][c]) for c in tax["categories"]],
          (row["dominant"] or "—").replace("_", " "),
          f"{row['hardest_bucket'] or '—'} ({pct(row['hardest_accuracy'])}, n={row['hardest_n']})"]
         for model, row in tax["models"].items()]))  # fmt: skip
    if m["gallery"]:
        cards = []
        for g in m["gallery"]:
            uri = thumbnail(data_dir / g["image"])
            img = f"<img src='{uri}' alt='item {_esc(g['bundle_id'])}'>" if uri else ""
            cards.append(f"<div class='card'>{img}<div><b>{_esc(short(g['model']))}</b> · "
                         f"<span class='pill'>{_esc(g['category'].replace('_', ' '))}</span></div>"
                         f"<div>{_esc(g['block'])}</div><div>truth <b>{_esc(g['truth'])}</b> · "
                         f"predicted <b>{_esc(g['prediction'])}</b></div>"
                         f"<code>{_esc(g['bundle_id'])}</code></div>")  # fmt: skip
        out.append("<h2>Representative failure gallery</h2><div class='gallery'>"
                   + "".join(cards) + "</div>")  # fmt: skip
    if "regression" in m:
        r = m["regression"]
        out.append(f"<h2>Session regression diff: {_esc(r['session_a'])} vs {_esc(r['session_b'])}"
                   "</h2>" + _html_table(
            ["Block", "Model", "Acc A", "Acc B", "Δ pp", "p (z)", "Δ p50 latency", "Δ cost",
             "Verdict"],
            [[x["block"], short(x["model"]), pct(x["acc_a"]), pct(x["acc_b"]),
              f"{x['delta_pp']:+.1f}", pval(x["p_z"]), ms(x["latency_delta_ms"]),
              usd(x["cost_delta_usd"]), x["verdict"]] for x in r["rows"]]))  # fmt: skip
    f = m["factory"]
    out.append("<h2>Factory timing and certification telemetry</h2>" + _html_table(
        ["Block", "Items", "Certified", "Quarantined", "Yield", "gen ms mean / p50 / p95",
         "verify ms mean / p50 / p95", "Certified / min"],
        [[b["block"], b["items"], b["certified"], b["quarantined"], pct(b["yield"]),
          _trio(b["gen_ms"]), _trio(b["verify_ms"]),
          "—" if b["certified_per_min"] is None else f"{b['certified_per_min']:.1f}"]
         for b in [*f["blocks"], {**f["overall"], "block": "All blocks"}]]))  # fmt: skip
    if "unit_economics" in m:
        out.append("<h2>Throughput and unit economics</h2>"
                   + _html_table(["Measure", "Value"], _econ_rows(m["unit_economics"]))
                   + "<p class='sub'>From this session's observed gen_ms and verify_ms; vCoT "
                   "extraction is timed separately by <code>knot bench-throughput</code>.</p>")  # fmt: skip
    meta = [("Session", h["session_name"]), ("Client / organisation", h["client"] or "—"),
            ("Evaluator", h["evaluator"] or "—"), ("Run id", h["run_id"] or "—"),
            ("Mode", "Offline mock (simulated)" if h["simulated"] else "Live (OpenRouter)"),
            ("Temperature / repeats k", f"{h.get('temperature', 0.0):g} / {h.get('repeats', 1)}"),
            ("State", h["state"]), ("Started (UTC)", h["started_utc"] or "—"),
            ("Finished (UTC)", h["finished_utc"] or "—"),
            ("Wall clock", "—" if h["duration_s"] is None else f"{h['duration_s']:.0f} s"),
            ("git commit", h["git_sha"]), ("kernel sha256", h["kernel_sha256"]),
            ("dataset sha256", h["dataset_sha256"]), ("Dataset folder", h["data_dir"]),
            ("Results folder", h["results_dir"]),
            ("Seeds", "; ".join(f"{b['label']}: {b['seed_range'][0]}–{b['seed_range'][1]}"
                                for b in h["blocks"])),
            ("Python / OS", f"{h['python']} · {h['platform']}"), ("Reproduce", h["cli"])]  # fmt: skip
    out.append("<h2>Publication and audit header</h2><div class='meta'>" + "".join(
        f"<div>{_esc(k)}</div><div><code>{_esc(v)}</code></div>" for k, v in meta) + "</div>")  # fmt: skip
    refs = m.get("references", REFERENCES)
    out.append(
        "<h2>Methodology &amp; Literature References</h2><ol class='refs'>"
        + "".join(f"<li>{_esc(r)}</li>" for r in refs)
        + "</ol>"
    )
    out.append(f"<footer>{_esc(FOOTER)}</footer></main></body></html>")
    return "".join(out)


_REL_HEAD = ["Block", "Model", "Items", "pass@1", "pass@k", "pass^k", "Gap (pp)"]


def _rel_rows(rel: dict[str, Any]) -> list[list[Any]]:
    return [
        [c["block"], short(c["model"]), c["items"], pct(c["pass_at_1"]), pct(c["pass_at_k"]),
         pct(c["pass_hat_k"]), "—" if c["gap"] is None else f"{100 * c['gap']:.1f}"]
        for c in rel["cells"]
    ]  # fmt: skip


def _trio(s: dict[str, float | None]) -> str:
    return " / ".join("—" if s[k] is None else f"{s[k]:,.0f}" for k in ("mean", "p50", "p95"))


def _html_table(head: list[str], rows: list[list[Any]]) -> str:
    th = "".join(f"<th>{_esc(x)}</th>" for x in head)
    body = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def _md_table(head: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in rows]
    return "\n".join(lines)


def render_md(m: dict[str, Any], brief: str | None = None) -> str:
    h = m["header"]
    out = [f"# {h['title']}", ""]
    if h["simulated"]:
        out += ["> **Simulated run.** Offline mock models; costs are planning estimates.", ""]
    out += [f"- **{k}:** {v}" for k, v in _kpis(m)] + [""]
    if brief:
        out += [f"## {BRIEF_TITLE}", "", brief.strip(), ""]
    out += ["## Statistical leaderboard", "", _md_table(
        ["Block", "Model", "Accuracy", "95% CI", "Correct/N", "p50 latency", "Cost", "$/correct"],
        [[c["block"], c["model"], pct(c["accuracy"]), f"{pct(c['ci_lo'])}–{pct(c['ci_hi'])}",
          f"{c['correct']}/{c['n']}", ms(c["latency_p50_ms"]), usd(c["cost_usd"]),
          usd(c["cost_per_correct"])] for c in m["leaderboard"]["cells"]]), ""]  # fmt: skip
    if "reliability" in m:
        rel = m["reliability"]
        out += [f"## Reliability over k = {rel['k']} repeats (pass@k / pass^k)", "",
                _md_table(_REL_HEAD, _rel_rows(rel)), "",
                *[f"> {x}" for x in rel["findings"]], ""]  # fmt: skip
    if m["contrasts"]:
        out += ["## Cross-block contrasts", "", _md_table(
            ["Model", "A", "B", "Effect (pp)", "p (z)", "p (Fisher)"],
            [[c["model"], c["block_a"], c["block_b"], f"{c['diff_pp']:+.1f}", pval(c["p_z"]),
              pval(c["p_fisher"])] for c in m["contrasts"]]), ""]  # fmt: skip
    tax = m["taxonomy"]
    out += ["## Error taxonomy", "", _md_table(
        ["Model", *tax["categories"], "Dominant", "Hardest bucket"],
        [[model, *[row["counts"][c] for c in tax["categories"]], row["dominant"] or "—",
          row["hardest_bucket"] or "—"] for model, row in tax["models"].items()]), ""]  # fmt: skip
    if "regression" in m:
        r = m["regression"]
        out += [f"## Regression diff: {r['session_a']} vs {r['session_b']}", "", _md_table(
            ["Block", "Model", "Acc A", "Acc B", "Δ pp", "p (z)", "Verdict"],
            [[x["block"], x["model"], pct(x["acc_a"]), pct(x["acc_b"]), f"{x['delta_pp']:+.1f}",
              pval(x["p_z"]), x["verdict"]] for x in r["rows"]]), ""]  # fmt: skip
    f = m["factory"]
    out += ["## Factory telemetry", "", _md_table(
        ["Block", "Items", "Certified", "Yield", "gen ms (mean/p50/p95)", "verify ms (mean/p50/p95)"],
        [[b["block"], b["items"], b["certified"], pct(b["yield"]), _trio(b["gen_ms"]),
          _trio(b["verify_ms"])] for b in [*f["blocks"], {**f["overall"], "block": "All"}]]), ""]  # fmt: skip
    if "unit_economics" in m:
        out += ["## Throughput and unit economics", "",
                _md_table(["Measure", "Value"], _econ_rows(m["unit_economics"])), ""]  # fmt: skip
    out += ["## Audit header", "", *[f"- **{k}:** `{h[k]}`" for k in (
        "session_name", "client", "run_id", "mode", "temperature", "started_utc", "finished_utc", "git_sha",
        "kernel_sha256", "dataset_sha256", "data_dir", "results_dir", "python", "platform", "cli")],
        ""]  # fmt: skip
    out += ["## Methodology & Literature References", "",
            *[f"{i}. {r}" for i, r in enumerate(m.get("references", REFERENCES), 1)],
            "", f"_{FOOTER}_", ""]  # fmt: skip
    return "\n".join(out)


def trial_rows(items: list[ItemRecord], trials: dict[str, TrialRecord]) -> list[dict[str, Any]]:
    by_key = {i.key: i for i in items}
    rows = []
    for t in sorted(trials.values(), key=lambda t: t.trial_id):
        item = by_key.get(f"{t.block_id}:{t.item_index}")
        row = t.model_dump(mode="json")
        row |= {"domain": item.domain if item else "", "preset": item.preset if item else "",
                "family": item.family if item else ""}  # fmt: skip
        rows.append({k: row.get(k, "") for k in CSV_FIELDS})
    return rows


def render_csv(items: list[ItemRecord], trials: dict[str, TrialRecord]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(CSV_FIELDS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(trial_rows(items, trials))
    return buf.getvalue()


# ------------------------------------------------------------------ export
def export(
    cfg: SessionConfig,
    compare: tuple[SessionConfig, dict[str, TrialRecord]] | None = None,
    brief: str | None = None,
    prov: dict[str, str] | None = None,
) -> dict[str, Path]:
    """Write report.html, report.md, report_metrics.json, trials.csv and the audit ZIP into
    `<results_dir>/reports/`. Refuses to write anything that carries a secret-shaped value."""
    results, data = storage.resolve(cfg.results_dir), storage.resolve(cfg.data_dir)
    items = storage.dataset_items(data)
    trials = storage.load_trials(results)
    status = storage.read_status(results)
    metrics = compute_metrics(cfg, items, trials, status, compare, prov)
    if brief:
        metrics["ai_brief"] = {"title": BRIEF_TITLE, "text": brief}
    texts = {
        "report.html": render_html(metrics, data, brief),
        "report.md": render_md(metrics, brief),
        "report_metrics.json": json.dumps(metrics, indent=2, default=str) + "\n",
        "trials.csv": render_csv(items, trials),
    }
    spec = results / storage.SPEC
    if spec.is_file():
        texts["campaign_spec.json"] = spec.read_text(encoding="utf-8")
    for name, text in texts.items():
        if find_secret(text):
            raise ValueError(f"refusing to export {name}: it contains a secret-shaped value")
    out = results / "reports"
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, text in texts.items():
        if name == "campaign_spec.json":
            continue
        (out / name).write_text(text, encoding="utf-8")
        paths[name] = out / name
    bundle = out / f"{cfg.slug}_audit_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, text in sorted(texts.items()):
            zf.writestr(name, text)
    paths["zip"] = bundle
    return paths


# ------------------------------------------------------------------ AI brief (opt-in)
def brief_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    """The only facts the AI brief may use: the deterministic metrics, trimmed."""
    keep = ("accuracy", "ci_lo", "ci_hi", "n", "cost_usd", "cost_per_correct", "latency_p50_ms")
    return {
        "session": metrics["header"]["session_name"],
        "simulated": metrics["header"]["simulated"],
        "models": [{"model": r["model"], **{k: r[k] for k in keep}}
                   for r in metrics["leaderboard"]["models"]],
        "cells": [{"block": c["block"], "model": c["model"], **{k: c[k] for k in keep}}
                  for c in metrics["leaderboard"]["cells"]],
        "contrasts": metrics["contrasts"],
        "taxonomy": {m: {"dominant": r["dominant"], "hardest_bucket": r["hardest_bucket"]}
                     for m, r in metrics["taxonomy"]["models"].items()},
        "yield": metrics["factory"]["overall"]["yield"],
    }  # fmt: skip


def ai_brief(metrics: dict[str, Any], model: str, api_key: str, transport: Any = None) -> str:
    """A three-paragraph executive recommendation written strictly from the metrics JSON. Runs
    only when a person clicks Synthesize with a live key; the key is never stored."""
    facts = json.dumps(brief_payload(metrics), default=str)
    body = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 700,
        "messages": [
            {"role": "system", "content": (
                "You write executive summaries of model evaluation results. Use ONLY the numbers "
                "in the JSON you are given; never invent figures, models or causes. Quote "
                "accuracies with their 95% confidence intervals and say when a difference is not "
                "statistically significant (p >= 0.05). If 'simulated' is true, say the run used "
                "simulated models. Write exactly three short paragraphs, plain text, no headings."
            )},
            {"role": "user", "content": f"Evaluation metrics JSON:\n{facts}"},
        ],
    }  # fmt: skip
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "X-Title": "knotwork-model-arena"}  # fmt: skip
    data = net.post_json(
        "https://openrouter.ai/api/v1/chat/completions", body, headers, 60.0, transport,
        policy_override="openrouter",
    )  # fmt: skip
    text = str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    return text.replace(api_key, "***").strip() if api_key else text.strip()


__all__ = [
    "BRIEF_TITLE", "ai_brief", "compute_metrics", "contrasts", "export", "fisher_exact",
    "leaderboard", "pareto_front", "pass_at_k", "percentile", "reliability", "render_html", "render_md", "svg_accuracy",
    "svg_pareto", "taxonomy", "two_proportion_z", "wilson",
]  # fmt: skip
