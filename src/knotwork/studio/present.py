"""View models for the Studio: repository status, scenario presets, difficulty scores, KPIs, twin
diagnosis, the calibration scoreboard and export summaries. Everything shown is computed from files
or from the live kernel; missing evidence is reported as missing."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, TypedDict

import yaml

from knotwork.devtools.kernel_audit import tcb_report
from knotwork.export.manifest import git_state
from knotwork.knotcore import engine
from knotwork.knotcore.contracts import FaultCode
from knotwork.knotcore.grading import grade_prediction
from knotwork.pipeline import telemetry
from knotwork.pipeline.io import read_json, read_jsonl
from knotwork.render.canvas import REPO_ROOT

AUTO = "auto"
Colour = Literal["red", "orange", "gray"]
#: knob values each one-click scenario sets; "auto" keeps the profile's own sampling ranges
PRESETS: dict[str, dict[str, Any]] = {
    "Clean baseline": {
        "reasoning_load": 0.2, "visual_stress": 0.0, "occlusion_level": 0.0, "realism": 0.0,
        "color_regime": "unique",
    },
    "Hard reasoning": {
        "reasoning_load": 0.9, "visual_stress": 0.2, "occlusion_level": 0.6, "realism": AUTO,
        "color_regime": AUTO,
    },
    "High visual stress": {
        "reasoning_load": 0.4, "visual_stress": 0.8, "occlusion_level": 0.4, "realism": 0.9,
        "color_regime": "mono",
    },
    "CVD check": {
        "reasoning_load": 0.4, "visual_stress": AUTO, "occlusion_level": AUTO, "realism": AUTO,
        "color_regime": "unique",
    },
}  # fmt: skip
KNOB_DEFAULTS: dict[str, Any] = {
    "reasoning_load": AUTO, "visual_stress": AUTO, "occlusion_level": AUTO, "realism": AUTO,
    "color_regime": AUTO,
}  # fmt: skip
DIAGNOSTIC_KINDS = ("perceptual_clean", "palette_swap", "reasoning_step_down", "text_ir_twin")


# ------------------------------------------------------------------ header
class RepoStatus(TypedDict):
    branch: str
    sha: str
    dirty: bool
    kernel: str
    tcb_ok: bool


def repo_status() -> RepoStatus:
    """Git branch and commit, the live kernel hash and whether the TCB audit passes."""
    git = git_state()
    head = REPO_ROOT / ".git" / "HEAD"
    ref = head.read_text(encoding="utf-8").strip() if head.is_file() else ""
    return {
        "branch": ref.rsplit("refs/heads/", 1)[-1] if "refs/heads/" in ref else "detached",
        "sha": str(git["sha"])[:7],
        "dirty": bool(git["dirty"]),
        "kernel": engine.kernel_sha256()[:12],
        "tcb_ok": tcb_report().ok,
    }


# ------------------------------------------------------------------ task studio
def _clamp(x: float) -> int:
    return int(round(100 * min(max(x, 0.0), 1.0)))


def difficulty_scores(
    plugin: str, params: dict[str, Any], payload: dict[str, Any]
) -> dict[str, tuple[int, str]]:
    """(0-100 score, how it was obtained) for reasoning load and visual stress. A score is the dial
    when one was set, otherwise an estimate from the sampled scene."""
    spec = payload.get("difficulty_spec") or {}
    if plugin == "tangle":
        load = (float(payload.get("n", 3)) - 3) / 7
        how = f"{payload.get('n')} cables"
    elif plugin == "panel":
        load = (float(payload.get("k", 2)) - 2) / 2
        how = f"{payload.get('k')} gauges, {payload.get('n')} cables"
    else:
        scale = str(params.get("scale", ""))
        load = 0.5 if scale == "offset" else 0.25
        how = f"{scale or 'dial'} scale"
    surface = payload.get("surface") or {}
    parts = [float(surface.get("grain", 0.0)) / 6.0, float(surface.get("noise", 0.0))]
    parts.append((1.0 - float(surface.get("dpi_scale", 1.0))) * 2.0)
    stress = 0.5 * min(parts[0], 1.0) + parts[1] * 0.5 + parts[2] * 0.5
    if plugin == "tangle":
        stress += (11.0 - float(payload.get("stroke", 9.0))) / 8.0 * 0.5
    reasoning = spec.get("reasoning_load")
    visual = spec.get("visual_stress")
    return {
        "reasoning": (_clamp(reasoning), "set by dial")
        if reasoning is not None
        else (_clamp(load), f"estimated from {how}"),
        "visual": (_clamp(visual), "set by dial")
        if visual is not None
        else (_clamp(stress), "estimated from grain, noise, resolution and stroke"),
    }


# ------------------------------------------------------------------ forensics
def run_kpis(dataset: Path) -> dict[str, Any]:
    """Yield and quarantine rate of a dataset, from its telemetry or else its stats."""
    report = dataset / telemetry.FILENAME
    if report.is_file():
        a = read_json(report)["aggregate"]
        return {
            "yield_rate": a["yield_rate"], "quarantine_rate": a["false_quarantine_rate"],
            "candidates": a["candidates"], "source": "telemetry.json",
        }  # fmt: skip
    stats = dataset / "stats.json"
    if stats.is_file():
        s = read_json(stats)
        rate = float(s["false_quarantine_rate"])
        return {
            "yield_rate": round(1.0 - rate, 6), "quarantine_rate": rate,
            "candidates": s["n_candidates"], "source": "stats.json",
        }  # fmt: skip
    return {"yield_rate": None, "quarantine_rate": None, "candidates": None, "source": "none"}


def fault_coverage(root: Path = REPO_ROOT / "artifacts") -> dict[str, Any]:
    """Faults caught with their exact code in the newest `knot mutate` report under `root`."""
    total = len(FaultCode)
    found: list[tuple[float, Path, dict[str, Any]]] = []
    for pattern in ("*mutate*.json", "*/*mutate*.json", "mutate/*.json", "*/mutate/*.json"):
        for path in root.glob(pattern):
            try:
                data = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and "faults" in data and "exact_catch_rate" in data:
                found.append((path.stat().st_mtime, path, data))
    if not found:
        return {"caught": None, "total": total, "report": None}
    _, path, data = max(found, key=lambda f: (f[0], str(f[1])))
    caught = sum(
        1
        for row in data["faults"].values()
        if row.get("injected") and row.get("caught_exact") == row.get("injected")
    )
    return {"caught": caught, "total": total, "report": path}


def grade_twins(suite: dict[str, Any], replies: dict[str, str]) -> dict[str, bool]:
    """Which twins a model solved, for the twins that have a reply and a certified answer."""
    out = {}
    for kind in DIAGNOSTIC_KINDS:
        twin = suite["twins"].get(kind, {})
        reply = replies.get(kind, "").strip()
        if reply and twin.get("status") in ("certified", "text"):
            text = reply if "answer:" in reply.lower() else f"ANSWER: {reply}"
            out[kind] = grade_prediction(text, twin["answer"]) == 1.0
    return out


def diagnose(solved: dict[str, bool]) -> tuple[str, str, Colour]:
    """(verdict, one-line reason, colour) from which twins the failing model solved."""
    if not solved:
        return "Awaiting twin results", "Enter the model's reply to at least one twin.", "gray"
    if solved.get("perceptual_clean"):
        return "Vision failure", "Solved once clutter and surface noise were removed.", "orange"
    if solved.get("palette_swap"):
        return "Vision failure", "Solved under a different palette: colour perception.", "orange"
    if solved.get("text_ir_twin"):
        return "Vision failure", "Solved from text but not from the picture.", "orange"
    if solved.get("reasoning_step_down"):
        return "Reasoning failure", "Solved only with one reasoning step removed.", "red"
    if "text_ir_twin" in solved:
        return "Reasoning failure", "Failed even with the facts given as text.", "red"
    return "Inconclusive", "Add a reply to the text twin to separate seeing from reasoning.", "gray"


def load_twin_results(report_dir: Path) -> dict[str, dict[str, str]]:
    """Optional `results.jsonl` beside a report: {bundle_id, kind, reply} rows from a model run."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for row in read_jsonl(report_dir / "results.jsonl"):
        out[str(row.get("bundle_id"))][str(row.get("kind"))] = str(row.get("reply", ""))
    return dict(out)


# ------------------------------------------------------------------ human eval
def model_names(config: Path = REPO_ROOT / "configs" / "models.yaml") -> dict[str, str]:
    """slot -> model slug from the benchmark configuration."""
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    return {str(s["slot"]): str(s.get("slug", s["slot"])) for s in data.get("slots", [])}


def short_model_name(slug: str) -> str:
    known = (
        ("gpt-5.4", "GPT-5.4"),
        ("gemini-2.5-flash", "Gemini 2.5 Flash"),
        ("qwen3-vl", "Qwen3-VL"),
    )
    return next((name for key, name in known if key in slug.lower()), slug)


def scoreboard(summary: dict[str, Any], names: dict[str, str]) -> list[dict[str, Any]]:
    """Rows for the participant and each model: overall accuracy, and accuracy on shared items."""
    shared = {r["who"]: r for r in summary["matched"]}
    rows = [
        {"who": "You", "accuracy": summary["human_overall"], "n": summary["n_human"],
         "shared": shared.get("human", {}).get("accuracy")}
    ]  # fmt: skip
    for r in summary["model_by_slot"]:
        rows.append(
            {"who": short_model_name(names.get(r["slot"], r["slot"])), "accuracy": r["accuracy"],
             "n": r["n"], "shared": shared.get(r["slot"], {}).get("accuracy")}
        )  # fmt: skip
    return rows


# ------------------------------------------------------------------ export
FORMAT_LABELS = {"hf": "Parquet / HF", "harbor": "Harbor", "rlvr": "RLVR", "manifest": "Manifest"}


def export_summary(out: Path) -> list[dict[str, Any]]:
    """Files and bytes per format, from the manifest that hashed them."""
    path = out / "dataset_manifest.json"
    if not path.is_file():
        return []
    files: dict[str, list[int]] = defaultdict(list)
    for name, info in read_json(path)["artifacts"].items():
        files[name.split("/", 1)[0] if "/" in name else "other"].append(int(info["bytes"]))
    files["manifest"] = [path.stat().st_size]
    return [
        {"format": FORMAT_LABELS.get(k, k), "files": len(v), "bytes": sum(v)}
        for k, v in sorted(files.items())
    ]


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"
