"""The Working Set: one shared, writable dataset that ties the five Studio pages together.

Studio adds certified items to it, Model Arena can evaluate it, Forensics and Human Eval open on it
and Export packs it. It is an ordinary Knotwork dataset folder (`index.jsonl` + `bundles/<id>/`),
so every existing reader (exporters, `knot verify`, the Arena's Studio-batch loader) works on it
unchanged. Two small files ride along:

* `review_log.jsonl`: the append-only Human QA log of this set (`knotwork.studio.review`);
* `working_set.json`: how many candidates were rendered and how many certified, for the KPIs.

Where it lives, first match wins:

1. `KNOTWORK_WORKING_SET` (an explicit folder);
2. under pytest, a per-process temporary folder, so parallel xdist workers never share files;
3. `/tmp/knotwork_demo/studio/working_set/` in the public demo;
4. `artifacts/studio/working_set/` otherwise.

Invariants: every write is atomic (a temporary sibling, then `os.replace`), so a reader never sees a
half-written index or bundle; items are de-duplicated by bundle id; only certified bundles (a
`task.json` whose certificate verdict is CERTIFIED) are ever added. On first use an empty set is
seeded from the committed, kernel-certified starter set in `samples/starter_working_set/` (one
item per built-in domain), which is a file copy, not a render, so a cold start stays fast.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import threading
import uuid
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from knotwork.render.canvas import REPO_ROOT
from knotwork.studio import demo_guard, review

ENV = "KNOTWORK_WORKING_SET"
STARTER = REPO_ROOT / "samples" / "starter_working_set"
INDEX, META, BUNDLES = "index.jsonl", "working_set.json", "bundles"
LABEL = "Working Set"
#: files a bundle download carries (beauty, witness overlay and manifest come from the bundle)
BUNDLE_FILES = ("beauty.png", "manifest.json", "task.json")
_LOCK = threading.Lock()  # one writer at a time inside a Streamlit server process


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_root() -> Path:
    """The folder of the active Working Set (see the module docstring for the order)."""
    override = os.environ.get(ENV, "").strip()
    if override:
        return demo_guard.confine(override) if demo_guard.is_public_demo() else Path(override)
    if demo_guard.is_public_demo():
        return demo_guard.sandbox_root() / "studio" / "working_set"
    if "PYTEST_CURRENT_TEST" in os.environ or "PYTEST_XDIST_WORKER" in os.environ:
        return Path(tempfile.gettempdir()) / f"knotwork-ws-{os.getpid()}"
    return REPO_ROOT / "artifacts" / "studio" / "working_set"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line) if line.strip() else None
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _task(bundle: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((bundle / "task.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def certified(task: dict[str, Any] | None) -> bool:
    return bool(task) and (task or {}).get("certificate", {}).get("verdict") == "CERTIFIED"


def index_row(task: dict[str, Any], source: str, seed: int | None = None) -> dict[str, Any]:
    """The `index.jsonl` row of a certified bundle (the pipeline's own fields, plus provenance)."""
    bid = str(task["bundle_id"])
    return {
        "answer_canonical": str(task["answer"]["canonical"]), "base_id": None, "bundle_id": bid,
        "dir": f"{BUNDLES}/{bid}", "family": str(task["family"]), "plugin": str(task["plugin"]),
        "role": "base", "scene_id": str(task.get("scene_id", "")),
        "split": str(task.get("split", "dev")), "seed": seed, "source": source,
        "added_utc": _now(),
    }  # fmt: skip


@dataclass(frozen=True)
class AddResult:
    added: bool
    bundle_id: str
    count: int
    reason: str = ""


class WorkingSet:
    """A Working Set folder. Cheap to construct; touches the disk only when asked."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_root()

    @property
    def index(self) -> Path:
        return self.root / INDEX

    @property
    def review_log(self) -> Path:
        return self.root / review.LOG_NAME

    # -------------------------------------------------------------- reading
    def valid(self) -> bool:
        return self.index.is_file() and (self.root / BUNDLES).is_dir() and bool(self.rows())

    def rows(self) -> list[dict[str, Any]]:
        return [r for r in _read_jsonl(self.index) if r.get("bundle_id") and r.get("dir")]

    def ids(self) -> list[str]:
        return [str(r["bundle_id"]) for r in self.rows()]

    def meta(self) -> dict[str, Any]:
        try:
            data = json.loads((self.root / META).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def kpis(self) -> dict[str, Any]:
        """Yield and quarantine rate over every candidate rendered into this set."""
        meta = self.meta()
        candidates = int(meta.get("candidates") or 0)
        good = int(meta.get("certified") or 0)
        if not candidates:
            return {"yield_rate": None, "quarantine_rate": None, "candidates": None,
                    "source": "none"}  # fmt: skip
        rate = good / candidates
        return {"yield_rate": round(rate, 6), "quarantine_rate": round(1 - rate, 6),
                "candidates": candidates, "source": META}  # fmt: skip

    def qa(self) -> dict[str, int]:
        """approved / flagged / unreviewed counts over the items of this set."""
        return review.counts(review.load_review_states(self.review_log), self.ids())

    # -------------------------------------------------------------- writing
    def ensure(self, starter: Path = STARTER) -> bool:
        """Seed an empty or invalid set from the committed starter. True when it seeded."""
        if self.valid():
            return False
        with _LOCK:
            if self.valid():
                return False
            source = starter if (starter / INDEX).is_file() else None
            if source is None:
                return False
            self.root.mkdir(parents=True, exist_ok=True)
            for row in _read_jsonl(source / INDEX):
                self._copy_bundle(source / str(row["dir"]), str(row["bundle_id"]))
            meta = json.loads((source / META).read_text(encoding="utf-8"))
            _write_atomic(self.root / META, json.dumps(meta, indent=1) + "\n")
            rows = [{**r, "added_utc": r.get("added_utc") or _now()}
                    for r in _read_jsonl(source / INDEX)]  # fmt: skip
            self._write_index(rows)
        return True

    def _copy_bundle(self, src: Path, bundle_id: str) -> None:
        target = self.root / BUNDLES / bundle_id
        if (target / "task.json").is_file():
            return
        if target.exists():  # a torn earlier copy: replace it
            shutil.rmtree(target, ignore_errors=True)
        tmp = target.with_name(f".{bundle_id}.{uuid.uuid4().hex[:8]}.tmp")
        shutil.copytree(src, tmp)
        try:
            os.replace(tmp, target)
        except OSError:  # another writer won the race: keep theirs
            shutil.rmtree(tmp, ignore_errors=True)

    def _write_index(self, rows: list[dict[str, Any]]) -> None:
        text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        _write_atomic(self.index, text)

    def add(self, bundle: Path, source: str = "studio", seed: int | None = None,
            attempts: int = 1) -> AddResult:  # fmt: skip
        """Copy one certified bundle in (de-duplicated by bundle id) and count its attempts."""
        task = _task(Path(bundle))
        if task is None or not certified(task):
            return AddResult(False, str((task or {}).get("bundle_id", "")), len(self.rows()),
                             "Only certified items can join the Working Set.")  # fmt: skip
        bid = str(task["bundle_id"])
        with _LOCK:
            rows = self.rows()
            if any(r["bundle_id"] == bid for r in rows):
                return AddResult(False, bid, len(rows), "Already in the Working Set.")
            self._copy_bundle(Path(bundle), bid)
            rows.append(index_row(task, source, seed))
            self._write_index(rows)
            meta = self.meta()
            meta["candidates"] = int(meta.get("candidates") or 0) + max(1, attempts)
            meta["certified"] = int(meta.get("certified") or 0) + 1
            meta["updated_utc"] = _now()
            _write_atomic(self.root / META, json.dumps(meta, indent=1) + "\n")
        return AddResult(True, bid, len(rows))

    def review(self, item_id: str, status: str, tags: list[str] | None = None,
               note: str = "") -> review.ReviewRecord:  # fmt: skip
        row = next((r for r in self.rows() if r["bundle_id"] == item_id), {})
        return review.record_review(self.review_log, item_id, status, str(row.get("plugin", "")),
                                    row.get("seed"), note=note, tags=tags)  # fmt: skip

    def table(self) -> list[dict[str, Any]]:
        """One row per item for previews: domain, family, answer, QA status and tags."""
        states = review.load_review_states(self.review_log)
        return [
            {"bundle_id": r["bundle_id"], "domain": r["plugin"], "family": r["family"],
             "answer": r["answer_canonical"], "status": review.status_of(states, r["bundle_id"]),
             "tags": review.tags_of(states, r["bundle_id"]), "source": r.get("source", "")}
            for r in self.rows()
        ]  # fmt: skip


# ------------------------------------------------------------------ downloads
def bundle_zip(
    bundle: Path, witness_png: bytes | None = None, vcot_line: str | None = None
) -> bytes:
    """An in-memory ZIP of one bundle: beauty, witness overlay, manifest, task and SFT trace."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in BUNDLE_FILES:
            path = Path(bundle) / name
            if path.is_file():
                zf.write(path, name)
        if witness_png:
            zf.writestr("witness_overlay.png", witness_png)
        if vcot_line:
            zf.writestr("vcot_sft.jsonl", vcot_line.rstrip("\n") + "\n")
    return buf.getvalue()


def folder_zip(folder: Path, prefix: str = "") -> bytes:
    """An in-memory ZIP of every file under `folder` (sorted, so the archive is reproducible)."""
    buf = io.BytesIO()
    root = Path(folder)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            zf.write(path, (Path(prefix) / path.relative_to(root)).as_posix())
    return buf.getvalue()


def vcot_line(bundle: Path) -> str | None:
    """The bundle's grounded vCoT trace as one SFT JSONL line, or None without a trace."""
    from knotwork.posttrain import vcot

    trace = vcot.try_extract(Path(bundle))
    if trace is None:
        return None
    task = _task(Path(bundle)) or {}
    question = str(task.get("question", {}).get("text", ""))
    record = {"item_id": trace.item_id, "domain": trace.domain, "family": trace.family,
              "prompt": question, "response": trace.formatted_cot,
              "final_answer": trace.final_answer}  # fmt: skip
    return json.dumps(record, ensure_ascii=False)


__all__ = [
    "BUNDLES", "ENV", "INDEX", "LABEL", "META", "STARTER", "AddResult", "WorkingSet",
    "bundle_zip", "certified", "default_root", "folder_zip", "index_row", "vcot_line",
]  # fmt: skip
