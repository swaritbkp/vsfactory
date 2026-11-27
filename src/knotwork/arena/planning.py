"""Campaign planning: the default blocks, the pre-flight cost and time estimate, and launching.

Estimates are deliberately cautious: the cost uses the benchmark's per-call token figures times
a 1.25 safety margin, so the real bill of a live run stays below the estimate.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from knotwork.arena import catalog, runner, storage
from knotwork.arena.schemas import BlockSpec, SessionConfig

#: measured seconds per certified item on one worker (render + kernel), by domain
SECONDS_PER_ITEM = {"rack_patch": 9.0, "tangle": 6.0, "gauge": 3.5, "panel": 10.0}
API_SECONDS_PER_CALL = 4.0
QUICK_N = (2, 10, 50, 250, 1000)


def default_blocks() -> list[BlockSpec]:
    """Two contrasting high-signal blocks: a dense monochrome rack loom and a hard-logic panel."""
    return [
        BlockSpec(block_id="b1", domain="rack_patch", preset="High stress", n=10, seed=1000),
        BlockSpec(block_id="b2", domain="panel", preset="Hard logic", n=10, seed=2000),
    ]


@dataclass(frozen=True)
class Preflight:
    images_to_generate: int
    images_reused: int
    trials: int
    gen_seconds: float
    eval_seconds: float
    max_cost_usd: float
    over_cap: bool

    @property
    def total_seconds(self) -> float:
        return self.gen_seconds + self.eval_seconds


def call_estimate(model: str, cat: catalog.Catalog | None, max_tokens: int = 512) -> float:
    p, c = catalog.prices(model, cat)
    out = min(runner.EST_COMPLETION_TOKENS, max_tokens)
    return (runner.EST_PROMPT_TOKENS * p + out * c) * runner.COST_MARGIN


def preflight(
    cfg: SessionConfig,
    cat: catalog.Catalog | None = None,
    reused_items: int = 0,
    gen_workers: int = 4,
) -> Preflight:
    """Images to make, trials to run, and the time and worst-case cost of the whole campaign."""
    generate = cfg.data_mode == "generate"
    gen_s = (
        sum(b.n * SECONDS_PER_ITEM.get(b.domain, 8.0) for b in cfg.blocks) / max(gen_workers, 1)
        if generate
        else 0.0
    )
    trials = cfg.total_trials
    by_concurrency = trials * API_SECONDS_PER_CALL / cfg.max_concurrency
    by_rate = trials * 60.0 / cfg.rpm
    eval_s = max(by_concurrency, by_rate) if cfg.mode == "live" else trials * cfg.mock_delay_s
    cost = sum(
        b.n * sum(call_estimate(m, cat, cfg.max_tokens) for m in cfg.models_for(b))
        for b in cfg.blocks
    )
    return Preflight(
        images_to_generate=cfg.total_items if generate else 0,
        images_reused=0 if generate else reused_items,
        trials=trials,
        gen_seconds=gen_s,
        eval_seconds=eval_s,
        max_cost_usd=cost,
        over_cap=cfg.mode == "live" and cost > cfg.spend_cap_usd,
    )


def human_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def price_table(cfg: SessionConfig, cat: catalog.Catalog | None) -> dict[str, list[float]]:
    """USD per token for every model of the campaign, frozen into the results folder."""
    models = {m for b in cfg.blocks for m in cfg.models_for(b)}
    return {m: list(catalog.prices(m, cat)) for m in sorted(models)}


def start(
    cfg: SessionConfig,
    workspace: storage.Workspace,
    cat: catalog.Catalog | None,
    api_key: str | None = None,
    resume: bool = False,
) -> tuple[list[str], object | None]:
    """Validate the folders, write the session files (no secrets), register it and start the
    runner. Returns (problems, process); nothing is started when there are problems."""
    if not cfg.run_id:
        cfg = cfg.model_copy(update={"run_id": uuid.uuid4().hex[:12]})
    if not cfg.created_utc:
        cfg = cfg.model_copy(update={"created_utc": storage.utc_now()})
    problems = storage.check_folders(cfg, resume)
    if cfg.mode == "live" and not api_key:
        problems.append("Live mode needs an OpenRouter API key (from .env or the override field).")
    if problems:
        return problems, None
    results = storage.resolve(cfg.results_dir)
    if not resume:
        storage.write_session(cfg)
        storage.write_json_atomic(results / runner.PRICES_FILE, price_table(cfg, cat))
    workspace.register(cfg)
    storage.send_control(results, "run")
    process = runner.launch(results, cfg.mode == "live", api_key if cfg.mode == "live" else None)
    return [], process


def resume(
    results_dir: Path, workspace: storage.Workspace, api_key: str | None, retry_failed: bool = False
) -> tuple[list[str], object | None]:
    cfg = storage.load_config(results_dir)
    if cfg is None:
        return [f"No valid config.json in {results_dir}."], None
    status = storage.read_status(results_dir)
    if status.state == "running" and runner.pid_alive(status.pid):
        return ["This session is already running."], None
    if cfg.mode == "live" and not api_key:
        return ["Live mode needs an OpenRouter API key (from .env or the override field)."], None
    workspace.register(cfg)
    storage.send_control(results_dir, "run")
    key = api_key if cfg.mode == "live" else None
    return [], runner.launch(results_dir, cfg.mode == "live", key, retry_failed)
