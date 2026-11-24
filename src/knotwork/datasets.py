"""Dataset discovery shared by probes, diversity and bench.

A *dataset* is a directory written by ``knot generate`` (index.jsonl + freshness_receipt.json). A data
argument may be one dataset or a directory of datasets (e.g. ``artifacts/phase3/bench_split/<plugin>``
or the selftest's ``data/<plugin>`` with a combined top-level index).
"""

from __future__ import annotations

from pathlib import Path


def dataset_dirs(data: Path) -> list[Path]:
    data = Path(data)
    if (data / "index.jsonl").is_file() and (data / "freshness_receipt.json").is_file():
        return [data]
    subs = sorted(
        p
        for p in data.iterdir()
        if p.is_dir() and (p / "index.jsonl").is_file() and (p / "freshness_receipt.json").is_file()
    )
    if subs:
        return subs
    return [data] if (data / "index.jsonl").is_file() else []


__all__ = ["dataset_dirs"]
