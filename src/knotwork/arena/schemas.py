"""Typed records of a Model Arena session: what to run, what happened, and how to steer it.

Every model here is written to disk as JSON. None of them has a field for an API key, and
`SessionConfig` refuses any value that looks like one, so a key can never reach `config.json`,
`campaign_spec.json` or the sessions index by accident.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1
Domain = Literal["rack_patch", "tangle", "gauge", "panel"]
Preset = Literal["Clean", "Hard logic", "High stress", "CVD"]
Mode = Literal["live", "mock"]
DataMode = Literal["generate", "reuse"]
State = Literal["idle", "running", "paused", "stopped", "done", "budget_stop", "failed"]
TrialStatus = Literal["ok", "api_error"]
ErrorCategory = Literal[
    "wrong_port_or_strand",
    "numeric_tolerance_miss",
    "format_violation",
    "false_premise_error",
    "refusal_or_empty",
    "api_error",
]
DOMAIN_LABELS: dict[str, str] = {
    "rack_patch": "Rack patch-bay",
    "tangle": "Tangle",
    "gauge": "Gauge",
    "panel": "Panel",
}
PRESETS: tuple[str, ...] = ("Clean", "Hard logic", "High stress", "CVD")
MAX_BLOCK_ITEMS = 100_000
MAX_SPEND_USD = 250.0
#: repeats per (item, model) for pass@k / pass^k reliability (ZeroBench, Roberts et al., ICML 2026)
MAX_REPEATS = 5
#: anything shaped like a provider secret (OpenRouter, OpenAI, Anthropic, bearer tokens)
SECRET_RE = re.compile(r"(sk-[A-Za-z0-9_-]{8,}|Bearer\s+[A-Za-z0-9._-]{8,})")


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


def trial_key(block_id: str, item_index: int, model: str, repeat_idx: int = 0) -> str:
    """The resume / dedup key of one trial, (block, item, model, repeat). Repeat 0 keeps the
    pre-repeats form `block:item:model`, so results.jsonl files written before repeats resume
    unchanged; later repeats append `#r`."""
    base = f"{block_id}:{item_index}:{model}"
    return base if repeat_idx == 0 else f"{base}#{repeat_idx}"


def find_secret(value: Any) -> str | None:
    """The first secret-shaped substring anywhere inside ``value`` (nested dicts and lists)."""
    if isinstance(value, str):
        hit = SECRET_RE.search(value)
        return hit.group(0) if hit else None
    if isinstance(value, dict):
        items: list[Any] = [*value.keys(), *value.values()]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return None
    return next((s for s in (find_secret(v) for v in items) if s), None)


class BlockSpec(_Record):
    """One block of a campaign: N certified items of one domain and preset, asked to some models."""

    block_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,40}$")
    domain: Domain
    preset: str = Field(min_length=1, max_length=40)  # a Studio preset, or "Studio batch"
    n: int = Field(ge=1, le=MAX_BLOCK_ITEMS)
    seed: int = Field(ge=0, lt=2**31)
    models: list[str] | None = None  # None: the session's default models

    @property
    def label(self) -> str:
        return f"{DOMAIN_LABELS[self.domain]} · {self.preset}"


class SessionConfig(_Record):
    """Everything a run needs, written as `config.json` in the results folder. Never a secret."""

    schema_version: int = SCHEMA_VERSION
    session_name: str = Field(min_length=1, max_length=120)
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,100}$")
    client: str = ""
    evaluator: str = ""
    notes: str = ""
    tags: list[str] = Field(default_factory=list)
    mode: Mode = "mock"
    data_mode: DataMode = "generate"
    data_dir: str
    results_dir: str
    default_models: list[str] = Field(min_length=1)
    blocks: list[BlockSpec] = Field(min_length=1)
    spend_cap_usd: float = Field(default=0.50, ge=0.0, le=MAX_SPEND_USD)
    max_concurrency: int = Field(default=2, ge=1, le=8)
    rpm: int = Field(default=60, ge=1, le=1000)
    max_tokens: int = Field(default=512, ge=16, le=8192)
    mock_delay_s: float = Field(default=0.0, ge=0.0, le=5.0)
    repeats: int = Field(default=1, ge=1, le=MAX_REPEATS)  # k: asks per (item, model)
    #: sampling temperature of live calls; 0.0 (the default) keeps answers deterministic, > 0 with
    #: k > 1 measures stochastic pass@k vs. pass^k and yields Best-of-N preference pairs
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    created_utc: str = ""
    run_id: str = ""

    @field_validator("default_models")
    @classmethod
    def _models_unique(cls, models: list[str]) -> list[str]:
        return list(dict.fromkeys(m.strip() for m in models if m.strip()))

    @model_validator(mode="after")
    def _no_secrets_and_unique_blocks(self) -> SessionConfig:
        leak = find_secret(self.model_dump(mode="json"))
        if leak:
            raise ValueError("a secret-shaped value was found in the session config")
        ids = [b.block_id for b in self.blocks]
        if len(set(ids)) != len(ids):
            raise ValueError("block ids must be unique")
        return self

    def models_for(self, block: BlockSpec) -> list[str]:
        return list(dict.fromkeys(block.models)) if block.models else list(self.default_models)

    @property
    def total_items(self) -> int:
        return sum(b.n for b in self.blocks)

    @property
    def total_trials(self) -> int:
        return sum(b.n * len(self.models_for(b)) for b in self.blocks) * self.repeats

    def block(self, block_id: str) -> BlockSpec:
        return next(b for b in self.blocks if b.block_id == block_id)


class ItemRecord(_Record):
    """One certified (or quarantined) item of the dataset, as listed in `dataset_manifest.json`."""

    block_id: str
    item_index: int = Field(ge=0)
    seed: int
    domain: str
    preset: str
    family: str
    certified: bool
    bundle_id: str = ""
    dir: str = ""  # relative to the data folder
    question: str = ""
    options: list[str] | None = None
    answer: dict[str, Any] = Field(default_factory=dict)  # the Answer model as JSON
    truth: str = ""
    codes: list[str] = Field(default_factory=list)
    attempts: int = 1
    gen_ms: float = 0.0
    verify_ms: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.block_id}:{self.item_index}"


class TrialRecord(_Record):
    """One model answering one item: a line of `results.jsonl` (the latest line per id wins)."""

    trial_id: str
    block_id: str
    item_index: int
    bundle_id: str
    model: str
    repeat_idx: int = Field(default=0, ge=0)  # 0 for rows written before repeats existed
    status: TrialStatus
    correct: bool | None = None
    truth: str = ""
    answer_type: str = ""
    extracted: str = ""
    reply: str = ""
    error_category: ErrorCategory | None = None
    error: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    attempts: int = 1
    simulated: bool = False
    ts_utc: str = ""


class RunStatus(_Record):
    """The runner's heartbeat, `status.json`, rewritten atomically after every step."""

    state: State = "idle"
    pid: int | None = None
    total_items: int = 0
    items_ready: int = 0
    items_certified: int = 0
    total_trials: int = 0
    completed_trials: int = 0
    failed_trials: int = 0
    spend_usd: float = 0.0
    spend_cap_usd: float = 0.0
    started_utc: str = ""
    updated_utc: str = ""
    finished_utc: str = ""
    message: str = ""
    phase: Literal["setup", "generate", "evaluate", "finished"] = "setup"


class ControlCommand(_Record):
    """`control.json`: the page asks, the runner obeys at its next step (cooperative)."""

    action: Literal["run", "pause", "stop"] = "run"
    issued_utc: str = ""


class SessionIndexEntry(_Record):
    """One row of `sessions_index.json`: where a session lives. Progress is read from its folder."""

    slug: str
    name: str
    client: str = ""
    tags: list[str] = Field(default_factory=list)
    results_dir: str
    data_dir: str
    created_utc: str = ""
    updated_utc: str = ""
    archived: bool = False
