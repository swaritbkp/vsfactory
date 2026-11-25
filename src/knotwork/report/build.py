"""`knot report` (Task 6.3): one self-contained ``report.html`` from ``artifacts/``.

Embeds every figure as base64 PNG and renders the gate table, the M1-M10 fault matrix, the pooled
shortcut-probe table and the benchmark summary. Missing artifacts become a visible "not available"
note rather than an error, so the report always builds.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

from knotwork.report.figures import SOURCE_LABEL, run_figures
from knotwork.report.summary import write_summary

CSS = """
body{font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;color:#0b0b0b;background:#fcfcfb;
max-width:1100px;margin:32px auto;padding:0 20px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:36px 0 8px;border-bottom:1px solid #e4e2dc}
.sub{color:#52514e}table{border-collapse:collapse;width:100%;margin:8px 0 16px;font-size:13.5px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid #e4e2dc;vertical-align:top}
th{color:#52514e;font-weight:600}td.num{text-align:right;font-variant-numeric:tabular-nums}
.ok{color:#0a7a32;font-weight:600}.bad{color:#b42318;font-weight:600}
img{max-width:100%;border:1px solid #e4e2dc;margin:8px 0}.note{color:#52514e;font-style:italic}
"""


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _e(x: Any) -> str:
    return html.escape(str(x))


def _table(head: list[str], rows: list[list[Any]], num: set[int] | None = None) -> str:
    num = num or set()
    th = "".join(f"<th>{_e(h)}</th>" for h in head)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td class='num'>{c}</td>" if i in num else f"<td>{c}</td>" for i, c in enumerate(r)
        )
        + "</tr>"  # fmt: skip
        for r in rows
    )
    return f"<table><tr>{th}</tr>{body}</table>"


def _flag(ok: bool) -> str:
    return "<span class='ok'>PASS</span>" if ok else "<span class='bad'>FAIL</span>"


def _pct(v: Any) -> str:
    return "n/a" if v is None else f"{100 * float(v):.1f}%"


def gates_section(art: Path) -> str:
    rows = []
    for ph in range(7):
        rep = _json(art / "gates" / f"phase{ph}.json")
        if rep is None:
            continue
        failed = [r["gate"] for r in rep["results"] if not r["ok"]]
        rows.append(
            [f"Phase {ph}", _flag(rep["ok"]), len(rep["results"]), _e(", ".join(failed) or "-")]
        )
    if not rows:
        return "<p class='note'>No gate reports found.</p>"
    return _table(["phase", "gate", "checks", "failed checks"], rows, {2})


def faults_section(art: Path) -> str:
    rep = _json(art / "phase2" / "mutate_report.json")
    if rep is None:
        return "<p class='note'>No mutate report found.</p>"
    rows = [[f, _e(r["plugin"]), _e(r["expected"]), r["injected"], r["caught_exact"],
             _flag(r["caught_exact"] == r["injected"])] for f, r in rep["faults"].items()]  # fmt: skip
    clean = rep.get("clean", {})
    return _table(
        ["fault", "plugin", "expected code", "injected", "caught exactly", ""], rows, {3, 4}
    ) + (
        f"<p class='sub'>Exact catch rate {_pct(rep.get('exact_catch_rate'))}; clean controls "
        f"false-quarantined {_pct(clean.get('false_quarantine_rate'))}.</p>"
    )


def probes_section(art: Path) -> str:
    rep = _json(art / "phase3" / "probe_report.json")
    if rep is None:
        return "<p class='note'>No probe report found.</p>"
    rows = []
    for plugin, row in sorted(rep["plugins"].items()):
        ch = row["pooled"]["chance"]
        for name, p in sorted(row["pooled"]["probes"].items()):
            ok = p["wilson_upper"] <= ch + 0.05
            rows.append(
                [
                    _e(plugin),
                    _e(name),
                    p["n"],
                    _pct(p["acc"]),
                    _pct(p["wilson_upper"]),
                    _pct(ch),
                    _flag(ok),
                ]
            )
    dd = ", ".join(f"{k} {v['dup_rate']:.3f}" for k, v in sorted(rep.get("dedup", {}).items()))
    return _table(["plugin", "probe", "n", "accuracy", "Wilson-95 upper", "chance", "≤ chance + 5%"], rows,
                  {2, 3, 4, 5}) + f"<p class='sub'>pHash near-duplicate rate per item: {_e(dd)}.</p>"  # fmt: skip


def bench_section(art: Path) -> str:
    s = _json(art / "phase5" / "bench_summary.json")
    if s is None:
        return "<p class='note'>No benchmark run found.</p>"
    rows = []
    for m in s["models"].values():
        a, t = m["accuracy"], m["twins"]
        st = m.get("staircase", {})
        rows.append([
            _e(m["slug"]), m["trials"], m["errors"], f"${m['cost_usd']:.4f}", _pct(a["overall"]["acc"]),
            _pct(a["tangle"]["acc"]), _pct(a["panel"]["acc"]), _pct(a["gauge_read"]["acc"]),
            f"{_pct(m['color_ablation'].get('acc_unique'))} → {_pct(m['color_ablation'].get('acc_mono'))}",
            f"{_pct(t['original'])} / {_pct(t['causal'])} / {_pct(t['nuisance'])} / {_pct(t['all_three'])}",
            _pct(m["gaslight"].get("failure_rate")),
            " / ".join(f"{k} {v.get('threshold_level')}" for k, v in sorted(st.items())),
        ])  # fmt: skip
    head = ["model", "trials", "errors", "cost", "overall", "tangle", "panel", "gauge read",
            "colour: unique → mono", "twins: orig / causal / nuisance / all 3", "false premise accepted",
            "staircase threshold"]  # fmt: skip
    note = (
        f"<p class='sub'>Source: {_e(SOURCE_LABEL.get(s['source'], s['source']))} ({_e(s['dir'])}); "
        f"{s['trials']} trials, {s['network_calls']} network calls, ${s['cost_usd']:.4f}. Staircases and the "
        "per-crossing / composition fits sit at the floor (every model fails at 1-2 crossings), so they "
        "bound rather than measure the threshold. The false-premise prompt forces yes/no: abstention is "
        "not measured.</p>"
    )
    return _table(head, rows, {1, 2, 3}) + note


def run_report(artifacts: Path, out: Path) -> int:
    artifacts, out = Path(artifacts).resolve(), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_summary(artifacts)
    figs = run_figures(artifacts, out.parent / "figures")
    imgs = "".join(
        f"<h3>{_e(p.stem)}</h3><img alt='{_e(p.stem)}' src='data:image/png;base64,"
        f"{base64.b64encode(p.read_bytes()).decode('ascii')}'>"
        for p in figs
    )
    doc = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Knotwork report</title>"
        f"<style>{CSS}</style></head><body><h1>Knotwork</h1><p class='sub'>Proof-carrying synthetic "
        "visual task factory: every bundle's answer is re-derived from pixels by a trusted kernel.</p>"
        f"<h2>Phase gates</h2>{gates_section(artifacts)}"
        f"<h2>Benchmark (3 VLMs)</h2>{bench_section(artifacts)}"
        f"<h2>Fault conformance matrix (M1-M10)</h2>{faults_section(artifacts)}"
        f"<h2>Shortcut probes (pooled)</h2>{probes_section(artifacts)}"
        f"<h2>Figures</h2>{imgs or '<p class=note>No figures.</p>'}</body></html>"
    )
    out.write_text(doc, encoding="utf-8")
    print(f"report: {out.as_posix()} ({out.stat().st_size // 1024} KB, {len(figs)} figures)")
    return 0


__all__ = ["run_report"]
