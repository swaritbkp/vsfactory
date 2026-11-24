"""Probe items: one certified base bundle reduced to what a shortcut probe may look at, plus the
prediction target and its per-item chance level (SPEC 11.2).

Targets are class strings so every probe is scored the same way:

* label answers with options -> ``idx<k>`` (position of the answer among the options), chance 1/|options|;
* int answers -> the integer, chance 1/7 for T2 (targets 0..6) and 1/(n_gauges + 1) for COUNT;
* zone answers -> the zone word, chance 1/3;
* float answers -> ``bin<b>`` over 10 equal-width bins of the queried gauge's printed scale, chance =
  majority-bin frequency of the family;
* premise answers -> ``<yes|no>|bin<b>``, chance = 1/2 x majority-bin frequency of the family.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from knotwork.datasets import dataset_dirs

N_BINS = 10
PREMISE_FAMILIES = ("P5_PREMISE_CONTROL", "T_GASLIGHT")


@dataclass
class Item:
    bundle_id: str
    plugin: str
    family: str
    scene_id: str
    dir: Path
    text: str
    options: list[str] | None
    answer: dict[str, Any]
    manifest: dict[str, Any]
    steps: list[dict[str, Any]]
    program: list[dict[str, Any]]
    difficulty: dict[str, float]
    target: str = ""
    chance: float = 0.0
    n_options: int = 0
    option_index: int | None = None
    value_bin: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def group(self) -> str:
        """Training group: the family, with the two premise families merged (a probe cannot tell
        a control from a gaslight item except through the answer itself)."""
        return "PREMISE" if self.family in PREMISE_FAMILIES else self.family

    @property
    def text_with_options(self) -> str:
        return self.text + (" OPTIONS: " + " ".join(self.options) if self.options else "")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def gauge_scale(manifest: dict[str, Any], gauge_id: str) -> tuple[float, float]:
    """(min, max) of the printed numeric tick labels of ``gauge_id`` (declarative manifest data)."""
    labels = {o["obj_id"]: o.get("label_text") for o in manifest["objects"]}
    for g in manifest["gauges"]:
        if g["gauge_id"] == gauge_id:
            vals = [float(labels[t["label_obj"]]) for t in g["ticks"]]
            return min(vals), max(vals)
    raise KeyError(gauge_id)


def value_bin(v: float, lo: float, hi: float) -> int:
    if hi <= lo:
        return 0
    return min(N_BINS - 1, max(0, math.floor(N_BINS * (v - lo) / (hi - lo))))


def _read_gauge(steps: list[dict[str, Any]]) -> str | None:
    for s in steps:
        if s["op"] == "READ":
            return str(s["inputs"][0])
    return None


def load_items(data: Path) -> list[Item]:
    """Base items of every dataset under ``data`` in index order, with targets and chance."""
    items: list[Item] = []
    for ds in dataset_dirs(data):
        for line in (ds / "index.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["role"] != "base":
                continue
            d = ds / row["dir"]
            task = _read(d / "task.json")
            q = task["question"]
            items.append(
                Item(
                    bundle_id=row["bundle_id"],
                    plugin=row["plugin"],
                    family=row["family"],
                    scene_id=row["scene_id"],
                    dir=d,
                    text=q["text"],
                    options=q.get("options"),
                    answer=task["answer"],
                    manifest=_read(d / "manifest.json"),
                    steps=task["certificate"]["steps"],
                    program=q["steps"],
                    difficulty=q.get("difficulty", {}),
                )
            )
    assign_targets(items)
    return items


def assign_targets(items: list[Item]) -> None:
    """Fill target / chance / option_index / value_bin (chance of float and premise families needs
    the family's bin distribution, so this runs over the whole list)."""
    for it in items:
        a, t = it.answer, it.answer["type"]
        if t in ("float", "premise"):
            v = float(a["value"]["value"] if t == "premise" else a["value"])
            gid = _read_gauge(it.steps)
            lo, hi = gauge_scale(it.manifest, gid) if gid else (0.0, 100.0)
            it.value_bin = value_bin(v, lo, hi)
            if t == "float":
                it.target = f"bin{it.value_bin}"
            else:
                holds = "yes" if a["value"]["premise_holds"] else "no"
                it.target = f"{holds}|bin{it.value_bin}"
        elif t == "label":
            opts = it.options or []
            it.n_options = len(opts)
            it.option_index = opts.index(str(a["value"])) if str(a["value"]) in opts else None
            it.target = f"idx{it.option_index}"
            it.chance = 1.0 / max(1, len(opts))
        elif t == "int":
            it.target = str(a["value"])
            if it.options:
                it.n_options, it.chance = len(it.options), 1.0 / len(it.options)
            elif it.family == "T2_CROSS":
                it.n_options, it.chance = 7, 1.0 / 7.0
            else:
                n = int(it.difficulty.get("n_gauges", 6))
                it.n_options, it.chance = n + 1, 1.0 / (n + 1)
            it.option_index = int(a["value"])
        elif t == "zone":
            opts = it.options or ["green", "amber", "red"]
            it.target, it.n_options = str(a["value"]), len(opts)
            it.option_index = opts.index(str(a["value"]))
            it.chance = 1.0 / len(opts)
        else:
            raise ValueError(f"unknown answer type {t}")
    by_family: dict[str, list[Item]] = {}
    for it in items:
        if it.value_bin is not None:
            by_family.setdefault(it.family, []).append(it)
    for fam_items in by_family.values():
        counts = Counter(i.value_bin for i in fam_items)
        maj = max(counts.values()) / len(fam_items)
        for it in fam_items:
            it.chance = maj * (0.5 if it.answer["type"] == "premise" else 1.0)


__all__ = [
    "N_BINS",
    "Item",
    "assign_targets",
    "dataset_dirs",
    "gauge_scale",
    "load_items",
    "value_bin",
]
