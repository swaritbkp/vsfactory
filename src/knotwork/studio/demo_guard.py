"""Public-demo sandbox guard (`KNOTWORK_PUBLIC_DEMO=1`), for a hosted Studio such as Cloud Run.

Off by default: unless the environment variable is exactly `"1"`, every function here is a no-op
(paths pass through unchanged, caps return `None`, the server's API key is readable), so local
`uv run` and self-hosted Docker keep full 100k-item, any-folder capability.

When on:

* every folder a person types (Studio batch output, Arena `data_dir` / `results_dir`, Export output,
  Register existing folder) must resolve, after symlinks and `..`, inside `/tmp/knotwork_demo/`;
  relative paths are placed under it;
* Studio batches, Arena items per block, blocks, repeats and the live spend cap are capped;
* the server never reads `OPENROUTER_API_KEY` (or `.env`), and Model Arena runs Offline Mock only:
  the Live mode and every API key field are hidden, and the launch handler forces `mode="mock"`
  with no key, so no visitor can make the hosted server spend money or reach the network;
* the `Import campaign.yaml` uploader is hidden and external `knotwork.plugins` entry points are
  not loaded (only the four built-in plugins run).

Opt-in second tier, `KNOTWORK_DEMO_ALLOW_BYOK=1` (only meaningful with the demo on): a visitor may
run a **Small Live BYOK Test** with their own key, on OpenRouter or on an HTTPS custom endpoint
that passes `ssrf_guard`. It is capped server-side at `BYOK_LIMITS` (5 items in 1 block, k = 1,
at most $0.50). The server's own `OPENROUTER_API_KEY` and `.env` are still never read, the
visitor's key lives only in their Streamlit session (RAM) and the runner's environment, and every
other demo rule above still holds. Unset or `0`, the demo is exactly the Offline Mock sandbox.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENV = "KNOTWORK_PUBLIC_DEMO"
BYOK_ENV = "KNOTWORK_DEMO_ALLOW_BYOK"
SANDBOX_ROOT = Path("/tmp/knotwork_demo")
BANNER = (
    "Live Interactive Sandbox (Offline Mock only · capped at 10 items/block · ephemeral /tmp "
    "storage)"
)
BYOK_BANNER = (
    "Live Interactive Sandbox (Offline Mock, or a Small Live BYOK Test of up to 5 items with your "
    "own key · ephemeral /tmp storage)"
)
PRIVACY_NOTE = "Key stays in browser session RAM only · never logged or saved"
OFFLINE_MODE = "mock"


@dataclass(frozen=True)
class Limits:
    batch_items: int = 10
    block_items: int = 10
    max_blocks: int = 3
    max_repeats: int = 3
    spend_cap_usd: float = 1.00


DEMO_LIMITS = Limits()


@dataclass(frozen=True)
class ByokLimits:
    """Server-side caps of a Small Live BYOK Test in the public demo."""

    total_items: int = 5
    max_blocks: int = 1
    max_repeats: int = 1
    spend_cap_usd: float = 0.50


BYOK_LIMITS = ByokLimits()


class SandboxViolation(ValueError):
    """A path outside the demo sandbox."""


def is_public_demo() -> bool:
    return os.environ.get(ENV) == "1"


def byok_allowed() -> bool:
    """Whether the public demo offers a Small Live BYOK Test (both variables exactly "1")."""
    return is_public_demo() and os.environ.get(BYOK_ENV) == "1"


def banner() -> str:
    return BYOK_BANNER if byok_allowed() else BANNER


def limits() -> Limits | None:
    """The demo caps, or None (no caps) outside the public demo."""
    return DEMO_LIMITS if is_public_demo() else None


def cap(value: int, ceiling: int) -> int:
    """`value` clipped to `ceiling` in the public demo; unchanged otherwise."""
    return min(value, ceiling) if is_public_demo() else value


def sandbox_root(root: Path | None = None) -> Path:
    """The sandbox folder (created). Only meaningful in the public demo."""
    path = (root or SANDBOX_ROOT).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def confine(path: str | Path, root: Path | None = None) -> Path:
    """In the public demo: the resolved path, which must lie inside the sandbox (relative paths are
    placed under it); raises SandboxViolation otherwise. Outside the demo: `Path(path)` unchanged."""
    if not is_public_demo():
        return Path(path)
    base = (root or SANDBOX_ROOT).resolve()
    raw = Path(str(path).strip()).expanduser()
    target = (raw if raw.is_absolute() else base / raw).resolve()
    if target != base and not target.is_relative_to(base):
        raise SandboxViolation(f"the public demo only reads and writes inside {base.as_posix()}/")
    return target


def path_problem(path: str | Path, root: Path | None = None) -> str | None:
    """A readable reason a path is refused in the demo, or None when it is fine."""
    try:
        confine(path, root)
    except SandboxViolation as exc:
        return str(exc).capitalize() + "."
    return None


def server_api_key(name: str = "OPENROUTER_API_KEY") -> str | None:
    """The server's own key from the environment; always None in the public demo."""
    if is_public_demo():
        return None
    return os.environ.get(name, "").strip() or None


def execution_modes(modes: dict[str, str]) -> dict[str, str]:
    """The Arena execution modes a visitor may pick: Offline Mock only in the public demo, unless
    the BYOK tier is on. Values are `mock` or a live provider (`live` / `openrouter` /
    `openai_compat`); anything other than `mock` counts as live."""
    if not is_public_demo() or byok_allowed():
        return dict(modes)
    return {label: mode for label, mode in modes.items() if mode == OFFLINE_MODE}


def offline_only(cfg: Any) -> Any:
    """In the public demo, the session config forced to Offline Mock (server-side enforcement of
    what the UI hides) unless the BYOK tier is on; unchanged otherwise."""
    if byok_allowed():
        return cfg
    if is_public_demo() and getattr(cfg, "mode", OFFLINE_MODE) != OFFLINE_MODE:
        return cfg.model_copy(update={"mode": OFFLINE_MODE})
    return cfg


def byok_problems(cfg: Any) -> list[str]:
    """Why a live session breaks the Small Live BYOK Test caps (empty list = fine)."""
    lim = BYOK_LIMITS
    problems = []
    if len(cfg.blocks) > lim.max_blocks:
        problems.append(f"A Small Live BYOK Test runs {lim.max_blocks} block.")
    if sum(b.n for b in cfg.blocks) > lim.total_items:
        problems.append(f"A Small Live BYOK Test runs at most {lim.total_items} items.")
    if cfg.repeats > lim.max_repeats:
        problems.append("A Small Live BYOK Test asks each item once (k = 1).")
    if cfg.spend_cap_usd > lim.spend_cap_usd:
        problems.append(f"A Small Live BYOK Test spend cap is at most ${lim.spend_cap_usd:.2f}.")
    if getattr(cfg, "provider", "openrouter") == "openai_compat":
        from knotwork.studio.ssrf_guard import validate_custom_base_url

        check = validate_custom_base_url(cfg.base_url, public_demo=True)
        if not check.ok:
            problems.append(f"Endpoint: {check.reason}")
    return problems


def campaign_import_allowed() -> bool:
    """Whether Model Arena shows the `Import campaign.yaml` uploader (never in the public demo)."""
    return not is_public_demo()


def config_problems(cfg: Any, root: Path | None = None) -> list[str]:
    """Why an Arena SessionConfig breaks the demo limits (empty list = fine, always outside it)."""
    lim = limits()
    if lim is None:
        return []
    problems = []
    if getattr(cfg, "mode", OFFLINE_MODE) != OFFLINE_MODE:
        if byok_allowed():
            problems += byok_problems(cfg)
        else:
            problems.append("The sandbox runs Offline Mock only.")
    if len(cfg.blocks) > lim.max_blocks:
        problems.append(f"The sandbox runs at most {lim.max_blocks} blocks.")
    if any(b.n > lim.block_items for b in cfg.blocks):
        problems.append(f"The sandbox runs at most {lim.block_items} items per block.")
    if cfg.repeats > lim.max_repeats:
        problems.append(f"The sandbox allows at most k = {lim.max_repeats} repeats.")
    if cfg.spend_cap_usd > lim.spend_cap_usd:
        problems.append(f"The sandbox spend cap is at most ${lim.spend_cap_usd:.2f}.")
    for label, folder in (("Dataset folder", cfg.data_dir), ("Results folder", cfg.results_dir)):
        why = path_problem(folder, root)
        if why:
            problems.append(f"{label}: {why}")
    return problems


__all__ = [
    "BANNER", "BYOK_BANNER", "BYOK_ENV", "BYOK_LIMITS", "DEMO_LIMITS", "ENV", "OFFLINE_MODE",
    "PRIVACY_NOTE", "SANDBOX_ROOT", "ByokLimits", "Limits", "SandboxViolation", "banner",
    "byok_allowed", "byok_problems", "campaign_import_allowed", "cap", "config_problems", "confine",
    "execution_modes", "is_public_demo", "limits", "offline_only", "path_problem", "sandbox_root",
    "server_api_key",
]  # fmt: skip
