"""Helpers shared by plugin and pipeline tests: profile + a parallel certify runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from knotwork.pipeline.certify import certify_job
from knotwork.pipeline.generate import run_jobs

REPO = Path(__file__).resolve().parents[2]
PROFILE: dict[str, Any] = yaml.safe_load(
    (REPO / "configs" / "default_profile.yaml").read_text(encoding="utf-8")
)


def certify_many(
    plugin: str,
    families: list[str],
    n: int,
    work: Path,
    root: int = 2026,
    fault: str | None = None,
    tag: str = "t",
) -> list[dict[str, Any]]:
    """Sample + render + certify ``n`` scenes in worker processes (families cycled)."""
    jobs = [
        {
            "plugin": plugin,
            "root": root,
            "path": [plugin, tag, i],
            "family": families[i % len(families)],
            "fault": fault,
            "work": str(work),
            "profile": PROFILE,
        }
        for i in range(n)
    ]
    return run_jobs(jobs, certify_job, 0)


def jobs_for(
    plugin: str,
    families: list[str],
    n: int,
    work: Path,
    root: int = 2026,
    fault: str | None = None,
    tag: str = "t",
) -> list[dict[str, Any]]:
    return [
        {
            "plugin": plugin,
            "root": root,
            "path": [plugin, tag, i],
            "family": families[i % len(families)],
            "fault": fault,
            "work": str(work),
            "profile": PROFILE,
            "tag": tag,
        }
        for i in range(n)
    ]


def run_all(jobs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Run every job in one process pool; results grouped by job tag."""
    out: dict[str, list[dict[str, Any]]] = {}
    for job, res in zip(jobs, run_jobs(jobs, certify_job, 0), strict=True):
        out.setdefault(job["tag"], []).append(res)
    return out
