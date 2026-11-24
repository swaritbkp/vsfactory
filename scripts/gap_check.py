#!/usr/bin/env python3
"""Optional [HUMAN-CHECKPOINT-1] live VLM gap check: <= $0.50, 20 items, one model.

HUMAN-ONLY. Claude Code must never run this (denied in .claude/settings.json; CLAUDE.md law 5).

    echo ok > .checkpoints/GAPCHECK_APPROVED
    export OPENROUTER_API_KEY=sk-or-...
    uv run python scripts/gap_check.py --data artifacts/phase2 --n 20 --slot vlm_b --budget-usd 0.50

It sends the first N certified *base* bundles (round-robin over artifacts/phase2/data_<plugin>/) to one
OpenRouter model slot from configs/models.yaml, grades the replies with knotcore.grading when available,
and writes artifacts/phase2/gap_check.json (accuracy by family, parse failures, cost). Every item sent
is appended to freshness/burn_ledger.jsonl (SPEC.md section 11.1).

Locks: --budget-usd must be <= 0.50; .checkpoints/GAPCHECK_APPROVED must exist; OPENROUTER_API_KEY must
be set (except with --dry-run, which never touches the network).

Exit codes: 0 ok | 2 usage | 4 safety refusal.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

GAP_CAP_USD = 0.50
APPROVAL = Path(".checkpoints/GAPCHECK_APPROVED")
LEDGER = Path("freshness/burn_ledger.jsonl")
PLUGIN_ORDER = ("tangle", "gauge", "panel")
SYSTEM = (
    "You are taking a visual reasoning test. Answer from the image only. "
    "End your reply with a single line 'ANSWER: <answer>'."
)
FORMAT_LINE = {
    "label": "ANSWER: <label>",
    "int": "ANSWER: <integer>",
    "float": "ANSWER: <number>",
    "zone": "ANSWER: <green|amber|red>",
    "premise": "ANSWER: <yes|no>; <number>",
}


class Refusal(SystemExit):
    def __init__(self, msg: str) -> None:
        print(msg, file=sys.stderr)
        super().__init__(4)


def _fallback_grade(pred: str, answer: dict[str, Any]) -> float:
    """Used only if knotwork.knotcore.grading is not built yet: canonical-string comparison."""
    lines = [m.group(1) for m in re.finditer(r"(?im)^\s*ANSWER:\s*(.+)$", pred)]
    text = lines[-1] if lines else pred
    norm = lambda s: re.sub(r"\s+", "", unicodedata.normalize("NFKC", s)).upper()  # noqa: E731
    return 1.0 if norm(text) == norm(str(answer["canonical"])) else 0.0


def _grader() -> Callable[[Any, dict[str, Any]], float]:
    try:
        from knotwork.knotcore.grading import grade_prediction  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return _fallback_grade  # type: ignore[return-value]
    return grade_prediction  # type: ignore[no-any-return]


def _parsed(text: str) -> bool:
    return re.search(r"(?im)^\s*ANSWER:\s*\S", text) is not None


def collect(data: Path, n: int) -> list[dict[str, Any]]:
    """First n base bundles, round-robin across data_<plugin> datasets (deterministic order)."""
    queues: list[list[dict[str, Any]]] = []
    for plugin in PLUGIN_ORDER:
        ds = data / f"data_{plugin}"
        index = ds / "index.jsonl"
        if not index.is_file():
            continue
        rows = [json.loads(x) for x in index.read_text(encoding="utf-8").splitlines() if x.strip()]
        queues.append([{**r, "_root": ds} for r in rows if r.get("role") == "base"])
    items: list[dict[str, Any]] = []
    while len(items) < n and any(queues):
        for q in queues:
            if q and len(items) < n:
                items.append(q.pop(0))
    return items


def image_data_url(path: Path, long_side: int) -> tuple[str, str]:
    img = Image.open(path).convert("RGB")
    scale = long_side / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()
    return "data:image/png;base64," + base64.b64encode(raw).decode(), hashlib.sha256(
        raw
    ).hexdigest()


def build_body(slot: dict[str, Any], cfg: dict[str, Any], prompt: str, url: str) -> dict[str, Any]:
    d = cfg["request_defaults"]
    body: dict[str, Any] = {
        "model": slot["slug"],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url, "detail": d["image_detail"]}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "temperature": d["temperature"],
        "max_tokens": int(slot.get("max_tokens") or d["max_tokens"]),
        "provider": dict(cfg["routing"]["provider"]),
        "usage": dict(cfg["routing"]["usage"]),
    }
    effort = slot.get("reasoning_effort", d.get("reasoning_effort"))
    if effort:
        body["reasoning"] = {"effort": effort}
    return body


def estimate_call_usd(slot: dict[str, Any], cfg: dict[str, Any], body: dict[str, Any]) -> float:
    plan = cfg["plan"]
    usd = (
        plan["est_tokens_in_per_call"] * slot["est_usd_per_mtok_in"]
        + body["max_tokens"] * slot["est_usd_per_mtok_out"]
    ) / 1e6
    return float(usd * cfg["budget"]["per_call_safety_margin"])


def main(argv: list[str] | None = None, transport: httpx.BaseTransport | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", type=Path, default=Path("artifacts/phase2"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--slot", default="vlm_b")
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--models", type=Path, default=REPO / "configs" / "models.yaml")
    ap.add_argument("--out", type=Path, default=Path("artifacts/phase2/gap_check.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not 0 < args.budget_usd <= GAP_CAP_USD:
        raise Refusal(f"E_BUDGET_OVER_CAP: gap check budget must be in (0, {GAP_CAP_USD:.2f}]")
    if not 1 <= args.n <= 20:
        raise Refusal("E_TOO_MANY_ITEMS: gap check sends at most 20 items")
    cfg = yaml.safe_load(args.models.read_text(encoding="utf-8"))
    slots = {s["slot"]: s for s in cfg["slots"]}
    if args.slot not in slots:
        print(f"unknown slot {args.slot}; choose from {sorted(slots)}", file=sys.stderr)
        return 2
    slot = slots[args.slot]
    items = collect(args.data, args.n)
    if not items:
        print(f"no base bundles under {args.data}/data_<plugin>/", file=sys.stderr)
        return 2

    prepared = []
    for it in items:
        task = json.loads((it["_root"] / it["dir"] / "task.json").read_text(encoding="utf-8"))
        q = task["question"]
        prompt = f"{q['text']}\n\nEnd with: {FORMAT_LINE[task['answer']['type']]}"
        url, img_sha = image_data_url(
            it["_root"] / it["dir"] / "beauty.png", cfg["request_defaults"]["image_long_side_px"]
        )
        body = build_body(slot, cfg, prompt, url)
        prepared.append((it, task, body, img_sha))
    est_total = sum(estimate_call_usd(slot, cfg, b) for _, _, b, _ in prepared)
    print(
        f"gap check: {len(prepared)} items -> {slot['slug']} (slot {args.slot}); "
        f"estimated <= ${est_total:.3f}, budget ${args.budget_usd:.2f}"
    )
    if args.dry_run:
        return 0
    if not APPROVAL.is_file():
        raise Refusal(f"E_CHECKPOINT_REQUIRED: create {APPROVAL} to approve the gap check")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise Refusal("E_API_KEY_MISSING: export OPENROUTER_API_KEY")

    grade = _grader()
    spent, results, worst_seen = 0.0, [], 0.0
    headers = {"Authorization": f"Bearer {key}", "X-Title": "knotwork-gap-check"}
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=cfg["request_defaults"]["timeout_s"], transport=transport) as client:
        for _it, task, body, img_sha in prepared:
            # pessimistic: never assume the next call is cheaper than the dearest one observed
            est = max(estimate_call_usd(slot, cfg, body), worst_seen)
            if spent + est > args.budget_usd:
                print(f"budget stop after {len(results)} items (spent ${spent:.4f})")
                break
            req_sha = hashlib.sha256(
                json.dumps(
                    {**body, "messages": "<omitted>", "image_sha256": img_sha}, sort_keys=True
                ).encode()
            ).hexdigest()
            with LEDGER.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "bundle_id": task["bundle_id"],
                            "scene_id": task["scene_id"],
                            "slot": args.slot,
                            "model": slot["slug"],
                            "request_sha256": req_sha,
                        }
                    )
                    + "\n"
                )
            r = client.post(cfg["endpoint"], json=body, headers=headers)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"].get("content") or ""
            usage = data.get("usage") or {}
            cost = float(usage.get("cost") or est)
            spent += cost
            worst_seen = max(worst_seen, cost)
            score = grade(text, task["answer"])
            results.append(
                {
                    "bundle_id": task["bundle_id"],
                    "family": task["family"],
                    "expected": task["answer"]["canonical"],
                    "reply_tail": text[-160:],
                    "parsed": _parsed(text),
                    "correct": score == 1.0,
                    "cost_usd": cost,
                }
            )
    by_family: dict[str, dict[str, float]] = {}
    for res in results:
        row = by_family.setdefault(res["family"], {"n": 0, "correct": 0})
        row["n"] += 1
        row["correct"] += int(res["correct"])
    for row in by_family.values():
        row["accuracy"] = row["correct"] / row["n"]
    n_done = len(results)
    report = {
        "slot": args.slot,
        "slug": slot["slug"],
        "n_requested": args.n,
        "n_done": n_done,
        "accuracy": (sum(r["correct"] for r in results) / n_done) if n_done else None,
        "parse_fail_rate": (sum(not r["parsed"] for r in results) / n_done) if n_done else None,
        "by_family": by_family,
        "cost_usd": round(spent, 6),
        "budget_usd": args.budget_usd,
        "items": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: report[k] for k in ("slug", "n_done", "accuracy", "parse_fail_rate", "cost_usd")}
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
