"""Human QA review states: an append-only `review_log.jsonl` of Approve / Flag decisions.

Each line is one decision:

    {"item_id": "kw-...", "domain": "tangle", "seed": 7, "status": "approved",
     "note": "", "reviewer": "operator", "timestamp_utc": "2026-11-27T10:00:00Z"}

The log is never rewritten: a later line for the same `item_id` supersedes an earlier one, so the
full audit trail survives and `load_review_states` resolves the latest state in one pass (O(N)).
Setting a status back to `unreviewed` is itself a recorded decision. Unparseable lines are skipped,
so a torn final line (a crash mid-write) never hides the decisions before it.

Downstream, `approved_ids` feeds the *Approved items only* filter of Export and of the Model
Arena's Post-Training Pack.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from knotwork.render.canvas import REPO_ROOT

ReviewStatus = Literal["approved", "flagged", "unreviewed"]
STATUSES: tuple[str, ...] = ("unreviewed", "approved", "flagged")
#: the QA tags the Studio offers; free-form tags are accepted too (lower-case, short)
TAGS: tuple[str, ...] = ("hard-case", "monochrome", "stress", "regression", "needs-review")
MAX_TAGS = 8
LOG_NAME = "review_log.jsonl"
#: the Studio's own review log (Human Eval and Export); Arena sessions keep one per results folder
STUDIO_LOG = REPO_ROOT / "artifacts" / "review" / LOG_NAME
BADGES: dict[str, tuple[str, str, str]] = {
    "unreviewed": ("Unreviewed", "gray", ":material/radio_button_unchecked:"),
    "approved": ("Approved", "green", ":material/verified:"),
    "flagged": ("Flagged", "red", ":material/flag:"),
}
QA_FILTER = "approved_only"


class ReviewRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1, max_length=120)
    domain: str = ""
    seed: int | None = None
    status: ReviewStatus
    note: str = Field(default="", max_length=280)
    reviewer: str = Field(default="operator", max_length=80)
    timestamp_utc: str = ""
    #: optional QA tags (added in v2.1; absent in older logs, which still load unchanged)
    tags: list[str] = Field(default_factory=list, max_length=MAX_TAGS)


def clean_tags(tags: list[str] | tuple[str, ...] | None) -> list[str]:
    """Lower-case, de-duplicated tags of at most 24 characters each, in first-seen order."""
    out: list[str] = []
    for tag in tags or ():
        text = "-".join(str(tag).lower().split())[:24]
        if text and text not in out:
            out.append(text)
    return out[:MAX_TAGS]


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def record_review(
    log_path: Path,
    item_id: str,
    status: str,
    domain: str = "",
    seed: int | None = None,
    note: str = "",
    reviewer: str = "operator",
    timestamp_utc: str | None = None,
    tags: list[str] | tuple[str, ...] | None = None,
) -> ReviewRecord:
    """Append one decision and flush it to disk. Raises ValueError for an unknown status."""
    try:
        rec = ReviewRecord(item_id=item_id, domain=domain, seed=seed, status=status,  # type: ignore[arg-type]
                           note=" ".join(note.split())[:280], reviewer=reviewer.strip() or "operator",
                           timestamp_utc=timestamp_utc or _now(), tags=clean_tags(tags))  # fmt: skip
    except ValidationError as exc:
        raise ValueError(exc.errors()[0]["msg"]) from None
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(rec.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return rec


def load_review_states(log_path: Path) -> dict[str, ReviewRecord]:
    """item_id -> its latest decision (later lines win). Missing log -> {}."""
    path = Path(log_path)
    if not path.is_file():
        return {}
    latest: dict[str, ReviewRecord] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = ReviewRecord.model_validate(json.loads(line))
        except (ValueError, ValidationError):
            continue
        latest[rec.item_id] = rec
    return latest


def status_of(states: dict[str, ReviewRecord], item_id: str) -> str:
    rec = states.get(item_id)
    return rec.status if rec else "unreviewed"


def approved_ids(log_path: Path) -> set[str]:
    return {k for k, r in load_review_states(log_path).items() if r.status == "approved"}


def counts(states: dict[str, ReviewRecord], item_ids: list[str]) -> dict[str, int]:
    """How many of `item_ids` are approved / flagged / unreviewed."""
    out = dict.fromkeys(STATUSES, 0)
    for i in item_ids:
        out[status_of(states, i)] += 1
    return out


def filter_note(approved: int, total: int) -> str:
    return f"{approved} of {total} items approved"


def tags_of(states: dict[str, ReviewRecord], item_id: str) -> list[str]:
    rec = states.get(item_id)
    return list(rec.tags) if rec else []


__all__ = [
    "BADGES", "LOG_NAME", "MAX_TAGS", "QA_FILTER", "STATUSES", "STUDIO_LOG", "TAGS",
    "ReviewRecord", "approved_ids", "clean_tags", "counts", "filter_note", "load_review_states",
    "record_review", "status_of", "tags_of",
]  # fmt: skip
