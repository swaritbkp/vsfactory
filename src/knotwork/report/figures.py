"""Publication figures (Task 6.1): ``python -m knotwork.report.figures --artifacts artifacts --out docs/figures``.

Static 200-dpi PNGs, >= 1600 px wide, light surface. Models keep one colour across every figure
(categorical slots 1-3 of the validated reference palette; every bar carries a direct value label,
which the palette's contrast relief rule requires for the aqua slot). Each figure's subtitle names
its data source (live run, offline replay or mock) so nothing simulated passes as measured.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402

from knotwork.report.summary import build_summary  # noqa: E402

DPI = 200
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated all-pairs (CVD 9.2, normal 24.0)
ACCENT, BASELINE = "#6250d6", "#9b9a95"
SHORT = {
    "openai/gpt-5.4": "GPT-5.4",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "qwen/qwen3-vl-235b-a22b-instruct": "Qwen3-VL-235B",
}
SOURCE_LABEL = {
    "live": "live OpenRouter run",
    "replay": "offline replay of the live run",
    "mock": "SIMULATED mock VLMs",
}


def _name(slug: str) -> str:
    return SHORT.get(slug, slug.split("/")[-1])


def _fig(w: float = 9.0, h: float = 5.0) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(w, h), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _titles(fig: Figure, title: str, sub: str) -> None:
    fig.text(0.02, 0.965, title, ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.02, 0.900, textwrap.fill(sub, 128), ha="left", va="top", fontsize=9.5, color=INK2,
             linespacing=1.4)  # fmt: skip


def _save(fig: Figure, out: Path, name: str, left: float = 0.08) -> Path:
    path = out / name
    fig.subplots_adjust(top=0.78, bottom=0.14, left=left, right=0.98)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return path


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.0f}%"


def _grouped(ax: Axes, groups: list[str], series: list[tuple[str, list[float | None]]]) -> None:
    n = len(series)
    width = 0.8 / n
    for i, (label, vals) in enumerate(series):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        ys = [0.0 if v is None else v for v in vals]
        ax.bar(
            xs, ys, width=width, color=SERIES[i % 3], edgecolor=SURFACE, linewidth=1.5, label=label
        )
        for x, v in zip(xs, vals, strict=True):
            ax.text(
                x, (v or 0.0) + 0.012, _pct(v), ha="center", va="bottom", fontsize=8.5, color=INK
            )
    ax.set_xticks(range(len(groups)), groups, fontsize=10.5, color=INK)
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")


def _legend(ax: Axes, n: int) -> None:
    ax.legend(
        loc="upper right", ncols=n, frameon=False, fontsize=9.5, labelcolor=INK, handlelength=1.2
    )


def fig_colour_ablation(s: dict[str, Any], out: Path) -> Path:
    fig, ax = _fig()
    models = list(s["models"].values())
    ys = list(range(len(models)))[::-1]
    for y, (i, m) in zip(ys, enumerate(models), strict=True):
        ca = m["color_ablation"]
        u, mo = ca.get("acc_unique") or 0.0, ca.get("acc_mono") or 0.0
        ax.plot([mo, u], [y, y], color=SERIES[i], linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter([u], [y], s=90, color=SERIES[i], zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.scatter([mo], [y], s=90, facecolor=SURFACE, edgecolor=SERIES[i], linewidth=2, zorder=3)
        ax.text(
            u + 0.02,
            y,
            f"unique colours {_pct(u)}",
            fontsize=9.5,
            color=INK,
            ha="left",
            va="center",
        )
        ax.text(
            mo, y - 0.26, f"monochrome {_pct(mo)}", fontsize=9, color=INK2, ha="center", va="top"
        )
    chance = next((m["color_ablation"].get("chance_mono") for m in models), None)
    if chance:
        ax.axvline(chance, color=BASELINE, linestyle=(0, (4, 3)), linewidth=1)
        ax.text(
            chance,
            len(models) - 0.45,
            f"chance ≈ {_pct(chance)}",
            fontsize=8.5,
            color=INK2,
            ha="center",
        )
    ax.set_yticks(ys, [_name(m["slug"]) for m in models], fontsize=11, color=INK)
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(-0.6, len(models) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.set_xlabel("cable-tracing accuracy on the same scenes (filled = unique colours, open = monochrome)",
                  fontsize=9.5, color=INK2)  # fmt: skip
    n = models[0]["color_ablation"].get("n")
    _titles(fig, "Colour as a shortcut: tracing with unique vs identical cable colours",
            f"Matched pairs (same braid, same geometry, only cable colours changed), n = {n} per model; "
            "exact McNemar per model in README · "
            f"{SOURCE_LABEL[s['source']]}")  # fmt: skip
    return _save(fig, out, "fig01_colour_ablation.png", left=0.17)


def fig_false_premise(s: dict[str, Any], out: Path) -> Path:
    fig, ax = _fig(9.0, 4.6)
    models = list(s["models"].values())
    vals = [m["gaslight"].get("failure_rate") for m in models]
    ys = list(range(len(models)))[::-1]
    for y, (i, v) in zip(ys, enumerate(vals), strict=True):
        ax.barh(y, v or 0.0, height=0.55, color=SERIES[i], edgecolor=SURFACE, linewidth=1.5)
        ax.text(
            (v or 0.0) + 0.015, y, _pct(v), va="center", fontsize=11, color=INK, fontweight="bold"
        )
    ax.set_yticks(ys, [_name(m["slug"]) for m in models], fontsize=11, color=INK)
    ax.set_xlim(0, 1.08)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.set_xlabel(
        "share of false-premise questions answered “yes” (premise accepted)",
        fontsize=9.5,
        color=INK2,
    )
    n = models[0]["gaslight"].get("n")
    _titles(fig, f"Preliminary false-premise probe (all {n} prompts false)",
            f"n = {n} per model; forced yes/no with no true-premise control — response bias not ruled out "
            f"· {SOURCE_LABEL[s['source']]}")  # fmt: skip
    return _save(fig, out, "fig02_false_premise.png", left=0.17)


def fig_twins(s: dict[str, Any], out: Path) -> Path:
    fig, ax = _fig()
    models = list(s["models"].values())
    groups = [
        "original",
        "causal twin\n(answer changes)",
        "nuisance twin\n(answer kept)",
        "all three\nsolved",
    ]
    series = [(_name(m["slug"]), [m["twins"][k] for k in ("original", "causal", "nuisance", "all_three")])
              for m in models]  # fmt: skip
    _grouped(ax, groups, series)
    _legend(ax, len(models))
    ax.set_ylim(0, max(0.45, max((v or 0) for _, vs in series for v in vs) + 0.12))
    ax.set_ylabel("accuracy", fontsize=9.5, color=INK2)
    n = models[0]["twins"]["n"]
    _titles(fig, "Twin consistency on cable tracing (directional: small n)",
            f"Cable-tracing triples: original, causal twin (two end labels swapped, cables unchanged), nuisance "
            f"twin (restyle only); n = {n} triples per model, so per-model drops are not significant · "
            f"{SOURCE_LABEL[s['source']]}")  # fmt: skip
    return _save(fig, out, "fig03_twin_consistency.png")


def fig_accuracy(s: dict[str, Any], out: Path) -> Path:
    fig, ax = _fig()
    models = list(s["models"].values())
    keys = [("overall", "overall"), ("tangle", "tangle\n(cable tracing)"), ("panel", "panel\n(trace + gauge)"),
            ("gauge_read", "gauge reading\n(on panels)")]  # fmt: skip
    series = [(_name(m["slug"]), [m["accuracy"][k]["acc"] for k, _ in keys]) for m in models]
    _grouped(ax, [g for _, g in keys], series)
    _legend(ax, len(models))
    ax.set_ylim(0, max(0.45, max((v or 0) for _, vs in series for v in vs) + 0.12))
    ax.set_ylabel("accuracy", fontsize=9.5, color=INK2)
    trials = sum(m["trials"] for m in models)
    _titles(fig, "Accuracy by task: all three models sit near the floor",
            f"{trials} trials across adaptive staircases and diagnostic blocks; gauge reading = ±2% of scale · "
            f"{SOURCE_LABEL[s['source']]}, ${s['cost_usd']:.2f}")  # fmt: skip
    return _save(fig, out, "fig04_accuracy_by_task.png")


def fig_failures(s: dict[str, Any], repo: Path, out: Path) -> Path | None:
    ex = [e for e in s["failure_examples"] if e.get("image")]
    if not ex:
        return None
    fig = plt.figure(figsize=(4.0 * len(ex), 6.8), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    what = {
        "staircase:tangle": "Trace the cable",
        "composition": "Read the gauge",
        "gaslight": "False premise",
    }
    for i, e in enumerate(ex):
        ax = fig.add_axes((i / len(ex) + 0.01, 0.31, 1 / len(ex) - 0.02, 0.54))
        with Image.open(repo / e["image"]) as im:
            ax.imshow(im.convert("RGB").resize((768, 768), Image.Resampling.LANCZOS))
        ax.set_axis_off()
        head = what.get(e["block"], e["block"])
        if e["block"] == "staircase:tangle":
            head += (
                f" ({e['crossings_on_path']} crossing{'s' if e['crossings_on_path'] != 1 else ''})"
            )
        fig.text(
            i / len(ex) + 0.02, 0.875, head, fontsize=12, fontweight="bold", color=INK, ha="left"
        )
        q = textwrap.fill(f"Q: {e.get('question', '')}", 52)
        fig.text(
            i / len(ex) + 0.02,
            0.29,
            q,
            fontsize=9,
            color=INK2,
            ha="left",
            va="top",
            linespacing=1.35,
        )
        lines = [f"truth: {e['truth']}"] + [f"{_name(k)}: {v}" for k, v in e["replies"].items()]
        fig.text(i / len(ex) + 0.02, 0.18, "\n".join(textwrap.shorten(x, 46) for x in lines), fontsize=10,
                 color=INK, ha="left", va="top", linespacing=1.5, family="monospace")  # fmt: skip
    fig.text(
        0.01, 0.985, "Where every model fails", fontsize=15, fontweight="bold", color=INK, va="top"
    )
    fig.text(0.01, 0.945, f"Items all three models answered wrong · {SOURCE_LABEL[s['source']]}",
             fontsize=9.5, color=INK2, va="top")  # fmt: skip
    path = out / "fig05_failure_examples.png"
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_diversity(artifacts: Path, out: Path) -> Path | None:
    p = artifacts / "phase3" / "diversity_report.json"
    if not p.is_file():
        return None
    rep = json.loads(p.read_text(encoding="utf-8"))["plugins"]
    names = sorted(rep)
    fig, ax = _fig()
    width = 0.36
    for j, (key, label, col) in enumerate((("vendi_ours", "Knotwork dataset", ACCENT),
                                           ("vendi_recolor", "recolour-only baseline", BASELINE))):  # fmt: skip
        xs = [g + (j - 0.5) * width for g in range(len(names))]
        vals = [float(rep[n][key]) for n in names]
        ax.bar(xs, vals, width=width, color=col, edgecolor=SURFACE, linewidth=1.5, label=label)
        for x, v in zip(xs, vals, strict=True):
            ax.text(x, v + 0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(names)), [f"{n}\n({float(rep[n]['ratio']):.1f}× baseline)" for n in names],
                  fontsize=10.5, color=INK)  # fmt: skip
    _legend(ax, 2)
    ax.set_ylim(0, max(float(rep[n]["vendi_ours"]) for n in names) * 1.3)
    ax.set_ylabel("Vendi score (effective number of distinct items)", fontsize=9.5, color=INK2)
    _titles(fig, "Diversity: generated scenes vary far beyond a recolouring",
            f"Vendi score on n = {rep[names[0]]['n']} items per plugin (32×32 image + task tokens), "
            "vs one scene recoloured n times; the gate requires ≥ 2×")  # fmt: skip
    return _save(fig, out, "fig06_diversity_vendi.png")


def run_figures(artifacts: Path, out: Path) -> list[Path]:
    artifacts, out = Path(artifacts).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    made: list[Path | None] = [fig_diversity(artifacts, out)]
    s = build_summary(artifacts)
    if s is not None:
        made += [fig_colour_ablation(s, out), fig_false_premise(s, out), fig_twins(s, out),
                 fig_accuracy(s, out), fig_failures(s, artifacts.parent, out)]  # fmt: skip
    return sorted(p for p in made if p is not None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    a = ap.parse_args(argv)
    for p in run_figures(a.artifacts, a.out):
        with Image.open(p) as im:
            print(f"{p.as_posix()}  {im.size[0]}x{im.size[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
