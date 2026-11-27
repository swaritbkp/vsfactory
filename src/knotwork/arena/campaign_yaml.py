"""Declarative `campaign.yaml`: a Model Arena session as a reviewable, versionable file, and the
headless runner behind `knot arena-run`.

    session_name: Rack monochrome audit
    mode: mock                       # mock (offline) | live (key from the environment)
    provider: openrouter             # live only: openrouter | openai_compat
    # base_url: http://localhost:8000/v1   # openai_compat: vLLM, SGLang, Ollama, OpenAI, Gemini
    # model_id: Qwen/Qwen2.5-VL-7B-Instruct # shorthand for default_models: [<model_id>]
    data_mode: generate
    default_models: [openai/gpt-5.4, google/gemini-2.5-flash]
    blocks:
      - {block_id: b1, domain: rack_patch, preset: High stress, n: 20, seed: 0}

Round trip: `load_campaign_yaml(export_campaign_yaml(cfg))` equals `cfg` minus the transient run
fields (`run_id`, `created_utc`). The file never holds a secret: `SessionConfig` has no key field,
rejects unknown fields and any secret-shaped value, and the exporter re-checks its own output. The
OpenRouter key of a live run is read only from the `OPENROUTER_API_KEY` environment variable, and
the optional key of a custom endpoint only from `KNOTWORK_CUSTOM_API_KEY`.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from knotwork.arena import report, storage
from knotwork.arena.regression import ascii_table
from knotwork.arena.schemas import SessionConfig, find_secret

#: run-time fields a campaign file never carries (each run gets fresh ones)
TRANSIENT = ("run_id", "created_utc")
#: field names that would only ever hold a credential; refused with a clear message
SECRET_FIELDS = {"api_key", "openrouter_api_key", "key", "token", "secret", "password"}
HEADER = (
    "# Knotwork Model Arena campaign. Run it headlessly with:\n"
    "#   uv run knot arena-run --campaign campaign.yaml --report\n"
    "# Live mode reads the key from OPENROUTER_API_KEY (provider: openrouter) or the optional\n"
    "# KNOTWORK_CUSTOM_API_KEY (provider: openai_compat) environment variable; never put it here.\n"
)


def export_campaign_yaml(config: SessionConfig) -> str:
    """The session as YAML (no transient fields, never a secret)."""
    data = config.model_dump(mode="json", exclude=set(TRANSIENT))
    if data.get("provider") == "openrouter" and not data.get("base_url"):
        data.pop("provider")  # the pre-v2.1 file shape for OpenRouter and mock campaigns
        data.pop("base_url")
    for block in data["blocks"]:
        if block.get("models") is None:
            block.pop("models")
    text = HEADER + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
    if find_secret(text):
        raise ValueError("refusing to export a campaign that carries a secret-shaped value")
    return text


def _read(path_or_text: str | Path) -> str:
    if isinstance(path_or_text, Path):
        return path_or_text.read_text(encoding="utf-8")
    if "\n" not in path_or_text and path_or_text.strip().endswith((".yaml", ".yml")):
        return Path(path_or_text.strip()).read_text(encoding="utf-8")
    return path_or_text


def _secret_fields(value: Any, where: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            name = f"{where}.{k}" if where else str(k)
            if str(k).lower() in SECRET_FIELDS:
                hits.append(name)
            hits += _secret_fields(v, name)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            hits += _secret_fields(v, f"{where}[{i}]")
    return hits


def load_campaign_yaml(path_or_text: str | Path) -> SessionConfig:
    """Parse and strictly validate a campaign (a file path or the YAML text itself). Missing
    `slug`, `data_dir` and `results_dir` are derived from the session name. Raises ValueError."""
    try:
        raw = yaml.safe_load(_read(path_or_text))
    except OSError as exc:
        raise ValueError(f"cannot read the campaign: {exc.strerror or exc}") from None
    except yaml.YAMLError as exc:
        raise ValueError(f"the campaign is not valid YAML: {exc}") from None
    if not isinstance(raw, dict):
        raise ValueError("a campaign must be a YAML mapping with session_name, models and blocks")
    leaked = _secret_fields(raw)
    if leaked:
        raise ValueError(f"campaign files must not carry credentials ({', '.join(leaked)}); "
                         "set OPENROUTER_API_KEY in the environment instead")  # fmt: skip
    if find_secret(raw):
        raise ValueError("the campaign carries a secret-shaped value; remove it")
    data = {k: v for k, v in raw.items() if k not in TRANSIENT}
    model_id = data.pop("model_id", None)
    if model_id is not None:
        if data.get("default_models"):
            raise ValueError("give either model_id or default_models, not both")
        data["default_models"] = [str(model_id)]
    data.setdefault("session_name", "Untitled campaign")
    data.setdefault("slug", storage.slugify(str(data["session_name"]), ""))
    default_data, default_results = storage.default_dirs(str(data["slug"]))
    data.setdefault("data_dir", default_data)
    data.setdefault("results_dir", default_results)
    try:
        return SessionConfig.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "campaign"
        raise ValueError(f"{where}: {first['msg']}") from None


# ------------------------------------------------------------------ headless runner
def summary_table(cfg: SessionConfig, trials: dict[str, Any]) -> str:
    lb = report.leaderboard(cfg, trials)
    rows = [[c["block"], c["model"], report.pct(c["accuracy"]),
             f"{c['correct']}/{c['n']}", report.usd(c["cost_usd"]), str(c["api_errors"])]
            for c in lb["cells"]]  # fmt: skip
    rows += [["ALL", m["model"], report.pct(m["accuracy"]), f"{m['correct']}/{m['n']}",
              report.usd(m["cost_usd"]), str(m["api_errors"])] for m in lb["models"]]  # fmt: skip
    return ascii_table(["block", "model", "accuracy", "correct/n", "cost", "api errors"], rows)


def run_campaign(
    cfg: SessionConfig,
    results_dir: Path | None = None,
    data_dir: Path | None = None,
    make_report: bool = False,
    gen_workers: int = 1,
    workspace: storage.Workspace | None = None,
) -> tuple[int, str]:
    """Run (or resume) a declarative campaign in this process. Returns (exit code, summary text).

    Exit codes: 0 done | 1 the run ended unfinished or failed | 2 bad folders | 4 live without key."""
    from knotwork.arena.runner import Runner

    update: dict[str, Any] = {}
    if results_dir is not None:
        update["results_dir"] = str(results_dir)
    if data_dir is not None:
        update["data_dir"] = str(data_dir)
    cfg = cfg.model_copy(update=update)
    if (
        cfg.mode == "live"
        and cfg.provider == "openrouter"
        and not os.environ.get("OPENROUTER_API_KEY")
    ):
        return 4, "arena-run: live mode needs OPENROUTER_API_KEY in the environment"
    results = storage.resolve(cfg.results_dir)
    existing = storage.load_config(results)
    resume = (results / storage.RESULTS).is_file()
    if existing is not None and resume:
        cfg = cfg.model_copy(
            update={"run_id": existing.run_id, "created_utc": existing.created_utc}
        )
    cfg = cfg.model_copy(update={"run_id": cfg.run_id or uuid.uuid4().hex[:12],
                                 "created_utc": cfg.created_utc or storage.utc_now()})  # fmt: skip
    problems = storage.check_folders(cfg, resume=True)
    if problems:
        return 2, "arena-run: " + " ".join(problems)
    storage.write_session(cfg)
    (workspace or storage.Workspace()).register(cfg)
    storage.send_control(results, "run")
    rc = Runner(cfg, results, gen_workers=max(1, gen_workers)).run()
    status = storage.read_status(results)
    trials = storage.load_trials(results)
    lines = [summary_table(cfg, trials),
             f"arena-run: {cfg.session_name} [{cfg.mode}] state={status.state} "
             f"trials={status.completed_trials}/{status.total_trials} "
             f"spend={report.usd(status.spend_usd)} results={results.as_posix()}"]  # fmt: skip
    if make_report:
        paths = report.export(cfg)
        lines.append("arena-run: report " + ", ".join(sorted(p.name for p in paths.values())))
    return (0 if rc == 0 and status.state == "done" else 1), "\n".join(lines)


__all__ = ["export_campaign_yaml", "load_campaign_yaml", "run_campaign", "summary_table"]
