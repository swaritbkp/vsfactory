"""Model Arena storage: the sessions index, session folders, atomic JSON and dataset fingerprints.

Layout of one session (both folders may live anywhere on the local disk):

    <data_dir>/dataset_manifest.json            certified items (ItemRecord per line in "items")
    <data_dir>/<block_id>/<index>/...            beauty.png, task.json, certificate.json, ...
    <results_dir>/config.json                    SessionConfig (no secrets)
    <results_dir>/campaign_spec.json             the blocks and models, for reproduction
    <results_dir>/status.json, control.json      runner heartbeat and cooperative control
    <results_dir>/results.jsonl                  one TrialRecord per line, flushed per trial
    <results_dir>/reports/                       exported reports

The index (`sessions_index.json`) only records where sessions live; archiving removes a row and
never deletes a folder.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from knotwork.arena.schemas import (
    BlockSpec,
    ControlCommand,
    ItemRecord,
    RunStatus,
    SessionConfig,
    SessionIndexEntry,
    TrialRecord,
    find_secret,
)
from knotwork.render.canvas import REPO_ROOT

DEFAULT_ROOT = REPO_ROOT / "artifacts" / "arena_workspaces"
ROOT_ENV = "KNOTWORK_ARENA_ROOT"  # tests and multi-tenant installs point the index elsewhere


def workspace_root() -> Path:
    from knotwork.studio import demo_guard

    if demo_guard.is_public_demo():
        if demo_guard.get_demo_session_id() is not None:
            return demo_guard.sandbox_root() / "arena"
        override = os.environ.get(ROOT_ENV, "").strip()
        return demo_guard.confine(override) if override else demo_guard.sandbox_root() / "arena"
    override = os.environ.get(ROOT_ENV, "").strip()
    return Path(override) if override else DEFAULT_ROOT


INDEX_NAME = "sessions_index.json"
MANIFEST = "dataset_manifest.json"
CONFIG, SPEC, STATUS, CONTROL, RESULTS = (
    "config.json",
    "campaign_spec.json",
    "status.json",
    "control.json",
    "results.jsonl",
)


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(name: str, date: str | None = None) -> str:
    """`Rack monochrome audit` -> `rack-monochrome-audit-20261127` (date defaults to today, UTC)."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").lower()
    base = re.sub(r"[^a-z0-9]+", "-", text).strip("-")[:60] or "session"
    day = date if date is not None else datetime.now(UTC).strftime("%Y%m%d")
    return f"{base}-{day}" if day and not base.endswith(day) else base


def resolve(text: str | Path) -> Path:
    """A session folder as an absolute path (relative = repository-relative). In the public demo
    it must lie inside the sandbox (relative = sandbox-relative), else SandboxViolation."""
    from knotwork.studio import demo_guard

    if demo_guard.is_public_demo():
        return demo_guard.confine(text)
    path = Path(str(text).strip()).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


# ------------------------------------------------------------------ atomic JSON
def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON through a temporary file and `os.replace`, so readers never see half a file.
    Refuses data that carries a secret-shaped value."""
    if find_secret(data):
        raise ValueError(f"refusing to write a secret-shaped value to {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    for attempt in range(5):  # Windows: a reader holding the file open makes replace fail briefly
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            import time

            time.sleep(0.05 * (attempt + 1))


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def append_trial(results_dir: Path, trial: TrialRecord) -> None:
    """Append one line and flush it to disk before returning (crash-safe progress)."""
    line = trial.model_dump_json()
    if find_secret(line):
        raise ValueError("refusing to write a secret-shaped value to results.jsonl")
    with (results_dir / RESULTS).open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# ------------------------------------------------------------------ session files
def session_paths(results_dir: Path) -> dict[str, Path]:
    return {
        name: results_dir / f for name, f in (("config", CONFIG), ("spec", SPEC), ("status", STATUS),
                                               ("control", CONTROL), ("results", RESULTS))
    }  # fmt: skip


def campaign_spec(cfg: SessionConfig) -> dict[str, Any]:
    """The reproducible part of a session: blocks, models and run settings."""
    return {
        "session": cfg.slug,
        "run_id": cfg.run_id,
        "mode": cfg.mode,
        "data_mode": cfg.data_mode,
        "blocks": [
            {**b.model_dump(mode="json"), "models": cfg.models_for(b), "seed_range": seed_range(b)}
            for b in cfg.blocks
        ],
        "default_models": cfg.default_models,
        "spend_cap_usd": cfg.spend_cap_usd,
        "max_concurrency": cfg.max_concurrency,
        "rpm": cfg.rpm,
        "temperature": cfg.temperature,
        "total_items": cfg.total_items,
        "total_trials": cfg.total_trials,
        "cli": cli_command(Path(cfg.results_dir)),
    }


def seed_range(block: BlockSpec) -> list[int]:
    return [block.seed, block.seed + block.n - 1]


def cli_command(results_dir: Path, retry_failed: bool = False) -> str:
    """The headless command that runs or resumes this session (no secrets: the key is read from
    the OPENROUTER_API_KEY environment variable)."""
    flag = " --retry-failed-only" if retry_failed else ""
    target = str(results_dir).replace("\\", "/")
    return f'uv run python -m knotwork.arena.runner --results-dir "{target}" --resume{flag}'


def write_session(cfg: SessionConfig) -> None:
    results = resolve(cfg.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    write_json_atomic(results / CONFIG, cfg.model_dump(mode="json"))
    write_json_atomic(results / SPEC, campaign_spec(cfg))


def load_config(results_dir: Path) -> SessionConfig | None:
    data = read_json(results_dir / CONFIG)
    if not isinstance(data, dict):
        return None
    try:
        return SessionConfig.model_validate(data)
    except ValidationError:
        return None


def read_status(results_dir: Path) -> RunStatus:
    data = read_json(results_dir / STATUS)
    try:
        return RunStatus.model_validate(data) if isinstance(data, dict) else RunStatus()
    except ValidationError:
        return RunStatus()


def write_status(results_dir: Path, status: RunStatus) -> None:
    write_json_atomic(results_dir / STATUS, status.model_dump(mode="json"))


def read_control(results_dir: Path) -> ControlCommand:
    data = read_json(results_dir / CONTROL)
    try:
        return ControlCommand.model_validate(data) if isinstance(data, dict) else ControlCommand()
    except ValidationError:
        return ControlCommand()


def send_control(results_dir: Path, action: str) -> None:
    command = ControlCommand.model_validate({"action": action, "issued_utc": utc_now()})
    write_json_atomic(results_dir / CONTROL, command.model_dump(mode="json"))


def load_trials(results_dir: Path) -> dict[str, TrialRecord]:
    """The latest record of every trial id (a retried trial supersedes its failed attempt)."""
    out: dict[str, TrialRecord] = {}
    path = results_dir / RESULTS
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                trial = TrialRecord.model_validate_json(line)
            except ValidationError:
                continue  # a torn last line after a crash is skipped, never fatal
            out[trial.trial_id] = trial
    return out


# ------------------------------------------------------------------ datasets
def load_manifest(data_dir: Path) -> list[ItemRecord]:
    data = read_json(data_dir / MANIFEST, {})
    items = data.get("items", []) if isinstance(data, dict) else []
    out = []
    for row in items:
        try:
            out.append(ItemRecord.model_validate(row))
        except ValidationError:
            continue
    return out


def write_manifest(data_dir: Path, items: list[ItemRecord], meta: dict[str, Any]) -> None:
    ordered = sorted(items, key=lambda i: (i.block_id, i.item_index))
    body = {**meta, "items": [i.model_dump(mode="json") for i in ordered]}
    body["dataset_sha256"] = dataset_fingerprint(ordered)
    write_json_atomic(data_dir / MANIFEST, body)


def dataset_fingerprint(items: list[ItemRecord]) -> str:
    """SHA-256 over the certified items' (block, index, bundle id); bundle ids are content
    addressed, so equal fingerprints mean byte-identical images, questions and answers."""
    keys = sorted(f"{i.block_id}:{i.item_index}:{i.bundle_id}" for i in items if i.certified)
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def studio_batch_items(data_dir: Path, block_id: str = "batch") -> list[ItemRecord]:
    """Base items of a Studio / `knot generate` dataset (index.jsonl + task.json per bundle)."""
    index = data_dir / "index.jsonl"
    if not index.is_file():
        return []
    out = []
    rows = [json.loads(x) for x in index.read_text(encoding="utf-8").splitlines() if x.strip()]
    for k, row in enumerate(r for r in rows if r.get("role", "base") == "base"):
        task = read_json(data_dir / str(row.get("dir", "")) / "task.json", {})
        if not task:
            continue
        answer = task.get("answer", {})
        question = task.get("question", {})
        out.append(
            ItemRecord(
                block_id=block_id, item_index=k, seed=k, domain=str(task.get("plugin", "")),
                preset="Studio batch", family=str(task.get("family", "")), certified=True,
                bundle_id=str(task.get("bundle_id", "")), dir=str(row.get("dir", "")),
                question=str(question.get("text", "")), options=question.get("options"),
                answer=answer, truth=str(answer.get("canonical", "")),
            )
        )  # fmt: skip
    return out


def dataset_items(data_dir: Path) -> list[ItemRecord]:
    """Items of an Arena dataset, else of a Studio batch folder, else none."""
    return load_manifest(data_dir) or studio_batch_items(data_dir)


def dataset_blocks(items: list[ItemRecord]) -> list[dict[str, Any]]:
    """Block summaries of a frozen dataset: id, domain, preset, item count and first seed."""
    blocks: dict[str, dict[str, Any]] = {}
    for item in items:
        b = blocks.setdefault(
            item.block_id,
            {"block_id": item.block_id, "domain": item.domain, "preset": item.preset, "n": 0,
             "certified": 0, "seed": item.seed},
        )  # fmt: skip
        b["n"] += 1
        b["certified"] += int(item.certified)
        b["seed"] = min(b["seed"], item.seed)
    return list(blocks.values())


def check_folders(cfg: SessionConfig, resume: bool) -> list[str]:
    """Problems that block a launch (empty list = fine). Creates the folders it needs."""
    problems = []
    data, results = resolve(cfg.data_dir), resolve(cfg.results_dir)
    if data.resolve() == results.resolve():
        problems.append("The dataset folder and the results folder must be different folders.")
    if cfg.data_mode == "reuse":
        if not dataset_items(data):
            problems.append(f"No certified dataset in {data}: it has no {MANIFEST} or index.jsonl.")
    else:
        data.mkdir(parents=True, exist_ok=True)
    if (results / RESULTS).is_file() and not resume:
        problems.append(
            f"{results} already holds results. Resume that session or choose an empty folder."
        )
    results.mkdir(parents=True, exist_ok=True)
    return problems


# ------------------------------------------------------------------ the sessions index
class Workspace:
    """The sessions index: one row per session, wherever its folders live."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = Path(root) if root is not None else None

    @property
    def root(self) -> Path:
        return self._root if self._root is not None else workspace_root()

    @property
    def index(self) -> Path:
        return self.root / INDEX_NAME

    def entries(self, include_archived: bool = False) -> list[SessionIndexEntry]:
        data = read_json(self.index, {})
        rows = data.get("sessions", []) if isinstance(data, dict) else []
        out = []
        for row in rows:
            try:
                entry = SessionIndexEntry.model_validate(row)
            except ValidationError:
                continue
            if include_archived or not entry.archived:
                out.append(entry)
        return sorted(out, key=lambda e: e.updated_utc or e.created_utc, reverse=True)

    def _save(self, entries: list[SessionIndexEntry]) -> None:
        rows = [e.model_dump(mode="json") for e in entries]
        write_json_atomic(self.index, {"version": 1, "sessions": rows})

    def get(self, slug: str) -> SessionIndexEntry | None:
        return next((e for e in self.entries(include_archived=True) if e.slug == slug), None)

    def upsert(self, entry: SessionIndexEntry) -> None:
        rows = [e for e in self.entries(include_archived=True) if e.slug != entry.slug]
        self._save([*rows, entry.model_copy(update={"updated_utc": utc_now()})])

    def register(self, cfg: SessionConfig) -> SessionIndexEntry:
        old = self.get(cfg.slug)
        entry = SessionIndexEntry(
            slug=cfg.slug, name=cfg.session_name, client=cfg.client, tags=list(cfg.tags),
            results_dir=str(resolve(cfg.results_dir)), data_dir=str(resolve(cfg.data_dir)),
            created_utc=old.created_utc if old else (cfg.created_utc or utc_now()),
        )  # fmt: skip
        self.upsert(entry)
        return entry

    def register_existing(self, results_dir: Path) -> SessionIndexEntry | None:
        """Mount a results folder saved anywhere (it must hold a valid config.json)."""
        cfg = load_config(resolve(results_dir))
        if cfg is None:
            return None
        cfg = cfg.model_copy(update={"results_dir": str(resolve(results_dir))})
        return self.register(cfg)

    def rename(self, slug: str, name: str, client: str, tags: list[str]) -> None:
        entry = self.get(slug)
        if entry is None:
            return
        self.upsert(entry.model_copy(update={"name": name, "client": client, "tags": tags}))
        cfg = load_config(Path(entry.results_dir))
        if cfg is not None:  # keep the folder's own config in step with the sidebar
            write_session(
                cfg.model_copy(update={"session_name": name, "client": client, "tags": tags})
            )

    def archive(self, slug: str) -> None:
        """Remove a session from the sidebar. Its folders on disk are left untouched."""
        rows = [e for e in self.entries(include_archived=True) if e.slug != slug]
        self._save(rows)

    def unique_slug(self, slug: str) -> str:
        taken = {e.slug for e in self.entries(include_archived=True)}
        if slug not in taken:
            return slug
        k = 2
        while f"{slug}-{k}" in taken:
            k += 1
        return f"{slug}-{k}"


def default_dirs(slug: str, root: Path | None = None) -> tuple[str, str]:
    """(data_dir, results_dir) defaults for a session, repository/sandbox-relative when possible."""
    from knotwork.studio import demo_guard

    if root is None and demo_guard.is_public_demo():
        return f"arena/{slug}/data", f"arena/{slug}/results"
    base = (root if root is not None else workspace_root()) / slug
    pair = []
    for leaf in ("data", "results"):
        path = base / leaf
        try:
            pair.append(path.relative_to(REPO_ROOT).as_posix())
        except ValueError:
            pair.append(str(path))
    return pair[0], pair[1]


def clone_config(cfg: SessionConfig, name: str, workspace: Workspace) -> SessionConfig:
    """Same blocks, models and presets under a new name, fresh folders and fresh seeds."""
    slug = workspace.unique_slug(slugify(name))
    data_dir, results_dir = default_dirs(slug, workspace.root)
    blocks = [b.model_copy(update={"seed": (b.seed + 10_007) % 2**31}) for b in cfg.blocks]
    return cfg.model_copy(
        update={"session_name": name, "slug": slug, "data_dir": data_dir,
                "results_dir": results_dir, "data_mode": "generate", "blocks": blocks,
                "created_utc": "", "run_id": ""},
    )  # fmt: skip
