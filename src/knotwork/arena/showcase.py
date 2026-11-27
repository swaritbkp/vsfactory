"""The committed v2 live audit (`docs/evidence/v2_live_audit/`) as a read-only Model Arena session.

A cold visitor sees real numbers before running anything: 531 live calls (530 scored, 1 API error)
to three vision models on 60 rendered items, with spend, yield, the leaderboard with Wilson 95%
intervals, pass@k vs pass^k reliability and the post-training frontier. Everything is read from
the committed files, never recomputed from a network call, and missing files mean no showcase.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knotwork.render.canvas import REPO_ROOT

EVIDENCE = REPO_ROOT / "docs" / "evidence" / "v2_live_audit"
FILES = ("report_metrics.json", "posttrain_summary.json", "report.md", "trials_summary.csv")
SLUG = "__showcase_v2_live_audit__"
NAMES = {
    "openai/gpt-5.4": "GPT-5.4",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "qwen/qwen3-vl-235b-a22b-instruct": "Qwen3-VL-235B",
}


def short(model: str) -> str:
    return NAMES.get(model, model.split("/", 1)[-1])


@dataclass(frozen=True)
class Showcase:
    metrics: dict[str, Any]
    posttrain: dict[str, Any]
    folder: Path

    @property
    def calls(self) -> int:
        t = self.metrics["totals"]
        return int(t["trials_scored"]) + int(t["api_errors"])

    @property
    def title(self) -> str:
        return f"Showcase: v2 4-Domain Live Audit ({self.calls} Calls · Read-Only)"

    def kpis(self) -> dict[str, Any]:
        t, f = self.metrics["totals"], self.metrics["factory"]["overall"]
        models = self.metrics["leaderboard"]["models"]
        best = max(models, key=lambda m: m["accuracy"])
        return {"spend_usd": float(t["spend_usd"]), "calls": self.calls,
                "scored": int(t["trials_scored"]), "api_errors": int(t["api_errors"]),
                "planned": int(t["trials_planned"]), "certified": int(f["certified"]),
                "items": int(f["items"]), "yield": float(f["yield"]), "best": short(best["model"]),
                "best_acc": float(best["accuracy"]), "best_lo": float(best["ci_lo"]),
                "best_hi": float(best["ci_hi"]), "models": len(models),
                "blocks": len(self.metrics["header"]["blocks"]),
                "k": int(self.metrics["header"].get("repeats", 1)),
                "temperature": float(self.metrics["header"].get("temperature", 0.0))}  # fmt: skip

    def leaderboard(self) -> list[dict[str, Any]]:
        rows = sorted(self.metrics["leaderboard"]["models"], key=lambda m: -m["accuracy"])
        return [{"Model": short(m["model"]), "Accuracy": 100 * m["accuracy"],
                 "95% CI": f"{100 * m['ci_lo']:.1f}% – {100 * m['ci_hi']:.1f}%",
                 "Correct / N": f"{m['correct']} / {m['n']}",
                 "p50 latency": f"{m['latency_p50_ms'] / 1000:.1f} s",
                 "Cost": m["cost_usd"], "$ / correct": m["cost_per_correct"]}
                for m in rows]  # fmt: skip

    def ties(self) -> list[tuple[str, str]]:
        """Pairs of models whose Wilson 95% intervals overlap: not separable at this sample size."""
        models = sorted(self.metrics["leaderboard"]["models"], key=lambda m: -m["accuracy"])
        out = []
        for i, a in enumerate(models):
            for b in models[i + 1 :]:
                if a["ci_lo"] <= b["ci_hi"] and b["ci_lo"] <= a["ci_hi"]:
                    out.append((short(a["model"]), short(b["model"])))
        return out

    def reliability(self) -> list[dict[str, Any]]:
        cells = sorted(self.metrics["reliability"]["cells"], key=lambda c: -c["gap"])
        return [{"Block": c["block"], "Model": short(c["model"]), "Items": c["items"],
                 "pass@1": 100 * c["pass_at_1"], "pass@k": 100 * c["pass_at_k"],
                 "pass^k": 100 * c["pass_hat_k"], "Gap (pp)": 100 * c["gap"]}
                for c in cells]  # fmt: skip

    def widest_gap(self) -> dict[str, Any]:
        return max(self.metrics["reliability"]["cells"], key=lambda c: c["gap"])

    def frontier(self) -> dict[str, int]:
        cells = [c for c in self.posttrain.get("frontier_table", []) if c.get("n")]
        counts = {k: sum(c["regime"] == k for c in cells)
                  for k in ("frontier", "cold_start", "saturated")}  # fmt: skip
        pack = self.posttrain.get("counts", {})
        return {**counts, "sft": int(pack.get("sft", 0)), "dpo": int(pack.get("dpo", 0)),
                "grpo": int(pack.get("grpo", 0))}  # fmt: skip

    def downloads(self) -> list[Path]:
        return [self.folder / name for name in FILES if (self.folder / name).is_file()]


def load(folder: Path = EVIDENCE) -> Showcase | None:
    """The committed audit, or None when its metrics file is missing or unreadable."""
    try:
        metrics = json.loads((folder / FILES[0]).read_text(encoding="utf-8"))
        post_path = folder / FILES[1]
        posttrain = json.loads(post_path.read_text(encoding="utf-8")) if post_path.is_file() else {}
    except (OSError, ValueError):
        return None
    needed = ("totals", "factory", "leaderboard", "reliability", "header")
    if not isinstance(metrics, dict) or any(k not in metrics for k in needed):
        return None
    return Showcase(metrics, posttrain, folder)


__all__ = ["EVIDENCE", "SLUG", "Showcase", "load", "short"]
