"""Throughput and unit economics of the certified-data factory (`knot bench-throughput`).

Every number is measured, and the cost model is printed in full:

    busy_s            = sum over every attempted item of (gen_ms + verify_ms + vcot_ms) / 1000
    s_per_certified   = busy_s / certified          (quarantined items' time is charged too)
    certified_per_hour = 3600 / s_per_certified     (one vCPU core)
    usd_per_1M        = 1e6 * s_per_certified / 3600 * vcpu_usd_per_hour

The default price, $0.04 per vCPU-hour, is a spot-cloud baseline; pass your own with --vcpu-usd-hr.
Generation is embarrassingly parallel, so N cores divide wall time by about N at the same cost.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

DEFAULT_VCPU_USD_HR = 0.04
FORMULA = (
    "usd_per_1M_certified = 1e6 * (sum(gen_ms + verify_ms + vcot_ms) / 1000 / certified) / 3600 "
    "* vcpu_usd_per_hour"
)
#: seed base of throughput runs: far from Arena defaults so a benchmark never reuses eval seeds
SEED_BASE = 7_000_000


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "max": None}
    ordered = sorted(values)
    return {"mean": sum(values) / len(values), "p50": ordered[len(ordered) // 2],
            "max": ordered[-1]}  # fmt: skip


def unit_economics(
    gen_ms: list[float],
    verify_ms: list[float],
    certified: int,
    vcot_ms: list[float] | None = None,
    vcpu_usd_hr: float = DEFAULT_VCPU_USD_HR,
) -> dict[str, Any]:
    """Single-core rate and $ per 1M certified samples from per-item timings (all attempted items)."""
    vcot = list(vcot_ms or [])
    attempted = len(gen_ms)
    busy_s = (sum(gen_ms) + sum(verify_ms) + sum(vcot)) / 1000.0
    per = busy_s / certified if certified else None
    return {
        "attempted": attempted,
        "certified": certified,
        "yield": certified / attempted if attempted else None,
        "busy_s": round(busy_s, 4),
        "s_per_certified": None if per is None else round(per, 6),
        "certified_items_per_hour": None if not per else round(3600.0 / per, 2),
        "vcpu_usd_per_hour": vcpu_usd_hr,
        "usd_per_1m_certified": None if per is None else round(1e6 * per / 3600.0 * vcpu_usd_hr, 2),
        "includes_vcot": bool(vcot),
        "formula": FORMULA,
    }


def bench(
    domain: str,
    n: int,
    out: Path | None = None,
    preset: str = "Clean",
    vcpu_usd_hr: float = DEFAULT_VCPU_USD_HR,
    seed: int = SEED_BASE,
) -> dict[str, Any]:
    """Generate + certify `n` items of `domain` on this core, time vCoT on each certified one."""
    from knotwork.arena.generation import make_item
    from knotwork.posttrain import vcot

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="knot-throughput-") as tmp:
        root = Path(out) if out else Path(tmp)
        for k in range(n):
            item = make_item("throughput", domain, preset, k, seed + k, root)
            v_ms = None
            if item.certified:
                t0 = time.perf_counter()
                trace = vcot.try_extract(root / item.dir)
                v_ms = (time.perf_counter() - t0) * 1000.0 if trace is not None else None
            rows.append({"index": k, "seed": seed + k, "family": item.family,
                         "certified": item.certified, "attempts": item.attempts,
                         "gen_ms": item.gen_ms, "verify_ms": item.verify_ms,
                         "vcot_ms": None if v_ms is None else round(v_ms, 2)})  # fmt: skip
    gen = [float(r["gen_ms"]) for r in rows]
    ver = [float(r["verify_ms"]) for r in rows]
    vc = [float(r["vcot_ms"]) for r in rows if r["vcot_ms"] is not None]
    certified = sum(bool(r["certified"]) for r in rows)
    return {
        "schema": "knotwork.posttrain.throughput/1",
        "domain": domain,
        "preset": preset,
        "n": n,
        "seed_range": [seed, seed + n - 1],
        "items": rows,
        "gen_ms": _stats(gen),
        "verify_ms": _stats(ver),
        "vcot_ms": _stats(vc),
        "economics": unit_economics(gen, ver, certified, vc, vcpu_usd_hr),
    }


def summary_lines(result: dict[str, Any]) -> list[str]:
    e = result["economics"]

    def ms(s: dict[str, Any]) -> str:
        return "-" if s["mean"] is None else f"{s['mean']:.0f} ms"

    rate = e["certified_items_per_hour"]
    cost = e["usd_per_1m_certified"]
    return [
        f"domain {result['domain']}, n {result['n']}, certified {e['certified']} "
        f"(yield {0 if e['yield'] is None else 100 * e['yield']:.0f}%)",
        f"mean gen {ms(result['gen_ms'])}, verify {ms(result['verify_ms'])}, "
        f"vCoT {ms(result['vcot_ms'])}",
        f"single-core certified items/hour: {'-' if rate is None else f'{rate:,.1f}'}",
        f"USD per 1M certified samples @ ${e['vcpu_usd_per_hour']}/vCPU-hr: "
        f"{'-' if cost is None else f'${cost:,.2f}'}",
        f"formula: {e['formula']}",
    ]


def run(domain: str, n: int, out: Path | None, json_out: Path | None, vcpu_usd_hr: float,
        preset: str) -> int:  # fmt: skip
    result = bench(domain, n, out, preset, vcpu_usd_hr)
    for line in summary_lines(result):
        print(line)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {json_out}")
    return 0 if result["economics"]["certified"] else 1


__all__ = ["DEFAULT_VCPU_USD_HR", "FORMULA", "bench", "run", "summary_lines", "unit_economics"]
