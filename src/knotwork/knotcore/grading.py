"""Answer grading `grade_prediction(pred, answer) -> 1.0 | 0.0` (SPEC 13.3).

STDLIB ONLY (json, math, re, unicodedata, typing): this file is copied verbatim into every exported
Harbor task as its verifier. `answer` is an Answer model or its JSON dict; `pred` is free text (the
last `ANSWER:` line wins, else the whole text) or a native value. Never raises.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Any

ANSWER_RE = re.compile(r"^\s*ANSWER:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
INT_RE = re.compile(r"[-+]?\d+")
ZONE_RE = re.compile(r"\b(green|amber|red)\b", re.IGNORECASE)
YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def extract(pred: str) -> str:
    hits = ANSWER_RE.findall(pred)
    return str(hits[-1]).strip() if hits else pred


def _num(x: Any, pattern: re.Pattern[str]) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = pattern.search(str(x))
    return float(m.group(0)) if m else None


def _close(x: float | None, v: Any, tol: float) -> bool:
    return x is not None and math.isfinite(x) and abs(x - float(v)) <= tol


def _grade(pred: Any, a: dict[str, Any]) -> bool:
    kind, v, tol = a["type"], a["value"], float(a.get("tolerance") or 0.0)
    if isinstance(pred, str):
        pred = extract(pred)
    if kind == "label":
        norm = [unicodedata.normalize("NFKC", str(s)).strip().upper() for s in (pred, v)]
        return norm[0] == norm[1]
    if kind == "int":
        x = _num(pred, INT_RE)
        return x is not None and x == float(v) and x == int(x)
    if kind == "float":
        return _close(_num(pred, NUM_RE), v, tol)
    if kind == "zone":
        words = {w.lower() for w in ZONE_RE.findall(str(pred))}
        return words == {str(v).lower()}
    if kind == "premise":
        if isinstance(pred, dict):
            holds, x = pred.get("premise_holds"), _num(pred.get("value"), NUM_RE)
        else:
            m = YESNO_RE.search(str(pred))
            holds = None if m is None else m.group(1).lower() == "yes"
            x = _num(str(pred)[m.end() :] if m else str(pred), NUM_RE)
        return holds is v["premise_holds"] and _close(x, v["value"], tol)
    return False


def grade_prediction(pred: Any, answer: Any) -> float:
    try:
        a = answer.model_dump(mode="json") if hasattr(answer, "model_dump") else answer
        a = json.loads(a) if isinstance(a, str) else dict(a)
        return 1.0 if _grade(pred, a) else 0.0
    except Exception:
        return 0.0
