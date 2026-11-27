"""Composite anti-reward-hacking RLVR reward with the native `verl` custom-reward signature.

STDLIB ONLY apart from `grade_prediction` (itself stdlib only): `export_kit` ships this file and the
kernel's grading module verbatim as `knotwork_verl_reward.py`, so a trainer needs no Knotwork install.

score = w_o * outcome + w_g * grounding_iou + w_f * (format + abstention) / 2, with defaults
w_o = 0.60, w_g = 0.25, w_f = 0.15. Rules that keep the reward honest:

* outcome is graded strictly by `knotcore.grading.grade_prediction` on the `\\boxed{}` content (else
  the whole reply, whose last `ANSWER:` line wins); two different committed answers score 0;
* grounding is the mean IoU of a one-to-one greedy matching between predicted `<box>` regions and
  the certified vCoT boxes, divided by max(#predicted, #certified): spraying boxes, or one box
  over the whole image, cannot buy IoU;
* when the outcome is wrong, grounding and format earn only `WRONG_PROCESS_SCALE` of their weight,
  so a wrong answer always scores below 0.25 with the default weights;
* a correct answer with zero grounding when grounding was asked for raises `reward_hack_flag`
  (a lucky or memorised guess, not a grounded read);
* on a false-premise task, accepting the premise is a confident hallucination (abstention 0).

Without reference boxes the grounding weight is dropped and the other two weights renormalised.
"""

from __future__ import annotations

import json
import re
from typing import Any

from knotwork.knotcore.grading import ANSWER_RE, committed_answers, grade_prediction

DEFAULT_WEIGHTS: dict[str, float] = {"outcome": 0.60, "grounding": 0.25, "format": 0.15}
#: share of the grounding + format weight a wrong answer can still earn
WRONG_PROCESS_SCALE = 0.5
BOX_RE = re.compile(
    r"<box>\s*\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*\]?\s*</box>",
    re.IGNORECASE,
)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
BOXED = "\\boxed{"
ABSTAIN_RE = re.compile(
    r"\b(cannot (?:be )?determin\w*|can't determine|unanswerable|not answerable|insufficient "
    r"(?:information|evidence)|false premise|premise is (?:false|incorrect|wrong)|"
    r"does not hold|is not (?:in|true))\b",
    re.IGNORECASE,
)
YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
NO_RE = re.compile(r"\bno\b", re.IGNORECASE)

Box = tuple[float, float, float, float]


# ------------------------------------------------------------------ parsing
def boxed_answers(text: str) -> list[str]:
    """Contents of every `\\boxed{...}`, braces balanced."""
    out, i = [], text.find(BOXED)
    while i >= 0:
        j, depth = i + len(BOXED), 1
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        if depth == 0:
            out.append(text[i + len(BOXED) : j - 1].strip())
        i = text.find(BOXED, j)
    return out


def parse_boxes(text: str) -> list[Box]:
    """Every `<box>[x1, y1, x2, y2]</box>` on the 0-1000 grid, clipped and corner-sorted."""
    boxes = []
    for m in BOX_RE.finditer(text):
        a, b, c, d = (min(1000.0, max(0.0, float(v))) for v in m.groups())
        x0, x1 = sorted((a, c))
        y0, y1 = sorted((b, d))
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1, y1))
    return boxes


def parse_answer(ground_truth: Any) -> dict[str, Any]:
    """The Answer dict: a JSON string / dict, or a bare canonical label."""
    if isinstance(ground_truth, dict):
        return dict(ground_truth)
    text = str(ground_truth)
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    if isinstance(value, dict) and "type" in value:
        return value
    return {"type": "label", "value": text, "canonical": text, "tolerance": 0.0}


# ------------------------------------------------------------------ signals
def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def grounding(pred: list[Box], ref: list[Box], hit: float = 0.5) -> tuple[float, int]:
    """(greedy one-to-one mean IoU over max(#pred, #ref), pairs with IoU >= hit)."""
    if not pred or not ref:
        return 0.0, 0
    pairs = sorted(
        ((iou(p, r), i, j) for i, p in enumerate(pred) for j, r in enumerate(ref)),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    used_p: set[int] = set()
    used_r: set[int] = set()
    total, matched = 0.0, 0
    for value, i, j in pairs:
        if value <= 0.0 or i in used_p or j in used_r:
            continue
        used_p.add(i)
        used_r.add(j)
        total += value
        matched += value >= hit
    return total / max(len(pred), len(ref)), matched


def format_score(text: str) -> float:
    """1.0 for one `<think>...</think>` followed by exactly one `\\boxed{}`; 0.5 for a bare
    `\\boxed{}` or `ANSWER:` line; 0.0 otherwise."""
    boxed = boxed_answers(text)
    think = THINK_RE.search(text)
    if think and len(boxed) == 1 and text.rfind(BOXED) > think.end() - 1:
        return 1.0
    if len(boxed) == 1 or (not boxed and ANSWER_RE.search(text)):
        return 0.5
    return 0.0


def _premise(answer: dict[str, Any]) -> bool | None:
    value = answer.get("value")
    if answer.get("type") == "premise" and isinstance(value, dict):
        return bool(value.get("premise_holds"))
    return None


def abstention_score(final: str, answer: dict[str, Any], extra: dict[str, Any]) -> float:
    """Calibration: on a false premise, rejecting it (or abstaining) is right and accepting it is a
    confident hallucination; on an unanswerable item, only abstaining is right; on an ordinary item,
    committing to an answer is right."""
    holds = _premise(answer)
    abstains = bool(ABSTAIN_RE.search(final))
    if extra.get("unanswerable"):
        return 1.0 if abstains else 0.0
    if holds is False:
        if YES_RE.search(final) and not NO_RE.search(final):
            return 0.0
        return 1.0 if (NO_RE.search(final) or abstains) else 0.0
    return 0.0 if abstains and not committed_answers(final) and not boxed_answers(final) else 1.0


def _normalise(pred: str, answer: dict[str, Any], domain: str) -> str:
    """Rack port answers are graded on the number: `PP-07`, `port 7` and `07` all mean 07."""
    if domain != "rack_patch" or answer.get("type") != "label":
        return pred
    digits = re.findall(r"\d+", pred)
    return f"{int(digits[-1]):02d}" if digits else pred


# ------------------------------------------------------------------ the reward
def score_rollout(
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
    data_source: str = "",
) -> dict[str, Any]:
    """Composite reward and its transparent breakdown for one rollout."""
    extra = dict(extra_info or {})
    text = str(solution_str or "")
    answer = parse_answer(ground_truth)
    w = {**DEFAULT_WEIGHTS, **(weights or extra.get("reward_weights") or {})}
    domain = str(extra.get("domain") or data_source.split("/")[-1])
    boxed = boxed_answers(text)
    if len(set(boxed)) > 1:  # two different boxed answers: hedging
        final, outcome = boxed[-1], 0.0
    else:
        final = boxed[-1] if boxed else text
        outcome = grade_prediction(_normalise(final, answer, domain), answer)
    ref = [tuple(float(v) for v in b) for b in extra.get("vcot_boxes_1000") or [] if len(b) == 4]
    pred = parse_boxes(text)
    asked = bool(ref)
    g_iou, matched = grounding(pred, ref) if asked else (0.0, 0)  # type: ignore[arg-type]
    fmt = format_score(text)
    abst = abstention_score(final, answer, extra)
    if not asked:
        total_w = w["outcome"] + w["format"]
        w = {"outcome": w["outcome"] / total_w, "grounding": 0.0, "format": w["format"] / total_w}
    process = w["grounding"] * g_iou + w["format"] * 0.5 * (fmt + abst)
    if outcome < 1.0:
        process *= WRONG_PROCESS_SCALE
    score = w["outcome"] * outcome + process
    return {
        "score": round(min(1.0, max(0.0, score)), 6),
        "outcome_score": float(outcome),
        "grounding_iou": round(g_iou, 6),
        "format_score": fmt,
        "abstention_score": abst,
        "matched_boxes": int(matched),
        "pred_boxes": len(pred),
        "ref_boxes": len(ref),
        "reward_hack_flag": bool(outcome == 1.0 and asked and g_iou == 0.0),
        "extracted": final.strip()[:120],
        "weights": {k: round(v, 6) for k, v in w.items()},
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """`verl` custom reward entry point (`reward.custom_reward_function.name=compute_score`)."""
    return score_rollout(solution_str, ground_truth, extra_info, data_source=data_source)


def trl_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    """Hugging Face TRL `GRPOTrainer` reward function: dataset columns arrive as keyword lists
    (`ground_truth`, `extra_info`, `data_source`); conversational completions are unwrapped."""
    n = len(completions)
    truths = kwargs.get("ground_truth") or [""] * n
    extras = kwargs.get("extra_info") or [None] * n
    sources = kwargs.get("data_source") or [""] * n
    out = []
    for c, gt, ex, src in zip(completions, truths, extras, sources, strict=False):
        if isinstance(c, list):  # [{"role": "assistant", "content": ...}]
            c = c[-1].get("content", "") if c and isinstance(c[-1], dict) else ""
        info = json.loads(ex) if isinstance(ex, str) else ex
        out.append(float(compute_score(str(src), str(c), gt, info)["score"]))
    return out


__all__ = [
    "DEFAULT_WEIGHTS",
    "boxed_answers",
    "compute_score",
    "grounding",
    "iou",
    "parse_boxes",
    "score_rollout",
    "trl_reward",
]
