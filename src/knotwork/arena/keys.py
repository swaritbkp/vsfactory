"""OpenRouter API key discovery for the arena page. Keys are only ever read here, held in memory
and handed to a runner process through its environment; nothing in this module writes a file."""

from __future__ import annotations

import os
from pathlib import Path

from knotwork.render.canvas import REPO_ROOT

ENV_NAME = "OPENROUTER_API_KEY"


def from_dotenv(path: Path) -> str | None:
    """The key from a `.env` file (`OPENROUTER_API_KEY=...`, optional quotes and `export`)."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip().removeprefix("export ").strip()
        if text.startswith(f"{ENV_NAME}="):
            value = text.split("=", 1)[1].strip().strip("'\"")
            return value or None
    return None


def detect(dotenv: Path = REPO_ROOT / ".env") -> tuple[str | None, str]:
    """(key, where it came from): the process environment first, then the repository `.env`."""
    env = os.environ.get(ENV_NAME, "").strip()
    if env:
        return env, "environment"
    found = from_dotenv(dotenv)
    return (found, ".env") if found else (None, "")


def mask(key: str | None) -> str:
    """`sk-or-…a1b2`: enough to recognise a key, never enough to use it."""
    if not key:
        return ""
    head = "sk-or-" if key.startswith("sk-or-") else key[:3]
    return f"{head}…{key[-4:]}" if len(key) > 8 else "…"
