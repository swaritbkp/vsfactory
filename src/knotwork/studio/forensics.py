"""Dataset discovery, quarantine forensics and lineage views."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from knotwork.diversity import parse_path
from knotwork.freshness import lineage
from knotwork.knotcore.contracts import DifficultySpec, Family
from knotwork.knotcore.seeds import SeedTree
from knotwork.pipeline.certify import Candidate, run_candidate
from knotwork.pipeline.counterfactual import run_counterfactual
from knotwork.pipeline.generate import run_generate
from knotwork.pipeline.io import append_jsonl, read_json, read_jsonl
from knotwork.pipeline.telemetry import pareto
from knotwork.plugins.registry import get_plugin
from knotwork.render.canvas import REPO_ROOT
from knotwork.studio.factory import PROFILE_PATH, profile

SEARCH_ROOTS = (REPO_ROOT / "artifacts", REPO_ROOT / "tests" / "bench" / "fixtures" / "data")
SEARCH_DEPTH = 4


def find_datasets(roots: tuple[Path, ...] = SEARCH_ROOTS) -> list[Path]:
    """Directories with an `index.jsonl` up to `SEARCH_DEPTH` levels under the search roots."""
    found: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for depth in range(SEARCH_DEPTH):
            pattern = "/".join(["*"] * depth + ["index.jsonl"])
            found.update(p.parent for p in root.glob(pattern) if (p.parent / "bundles").is_dir())
    return sorted(found)


def bundles(dataset: Path) -> list[dict[str, Any]]:
    return read_jsonl(dataset / "index.jsonl")


def quarantine(dataset: Path) -> list[dict[str, Any]]:
    return read_jsonl(dataset / "quarantine.jsonl")


def rejection_pareto(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from collections import Counter

    return pareto(Counter(code for r in records for code in r.get("codes", [])))


def reproduce(dataset: Path, record: dict[str, Any], out: Path) -> Candidate:
    """Re-sample a quarantined base candidate from its seed path and certify it again."""
    if record.get("stage") == "twin" or "twin" in str(record.get("seed_path", "")):
        raise ValueError("twin rejections are not reproducible on their own")
    root = int(read_json(dataset / "freshness_receipt.json")["root_seed"])
    path = parse_path(record["seed_path"])
    stats = read_json(dataset / "stats.json")
    slot = int(path[2])
    family = stats["schedule"][slot]
    split = stats.get("split") or bundles(dataset)[0]["split"]
    prof = dict(
        profile() | (stats.get("profile_overrides") or {}),
        _family=family,
        _split=split,
        _ordinal=stats["schedule"][:slot].count(family),
        _balance_seed=root,
    )
    spec = DifficultySpec.model_validate(stats["difficulty"]) if stats.get("difficulty") else None
    plugin = get_plugin(str(record["plugin"]))
    at = SeedTree(root, path)
    scene = plugin.sample_scene(at.child("scene"), prof, spec)
    question = plugin.questions(scene, at.child("question"), Family(family))
    return run_candidate(plugin, scene, question, out, prof)


def demo_suite(out: Path) -> Path:
    """Generate one fresh tangle item and build its twin suite. No model failed it: the failure row
    only names the bundle, so the report shows no replies."""
    data = out / "demo_data"
    code = run_generate(
        "tangle", 1, data, 2, "offline", PROFILE_PATH, "dev", False, ["T1_TRACE"], 1,
        difficulty={"reasoning_load": 0.4},
    )  # fmt: skip
    if code != 0:
        raise RuntimeError("the demo item did not certify")
    failures = out / "demo_failures.jsonl"
    failures.unlink(missing_ok=True)
    append_jsonl(failures, [{"bundle_id": bundles(data)[0]["bundle_id"]}])
    if run_counterfactual(data, failures, out) != 0:
        raise RuntimeError("no twin suite was built")
    return out / "suites.json"


def dataset_suite(data: Path, out: Path) -> Path:
    """Counterfactual twins for the first base item of an existing dataset (the Working Set). No
    model failed it, so the report shows twins and certified answers but no replies."""
    base = next((r for r in bundles(data) if r.get("role", "base") == "base"), None)
    if base is None:
        raise RuntimeError("the dataset has no base item")
    out.mkdir(parents=True, exist_ok=True)
    failures = out / "dataset_failures.jsonl"
    failures.unlink(missing_ok=True)
    append_jsonl(failures, [{"bundle_id": base["bundle_id"]}])
    if run_counterfactual(data, failures, out) != 0:
        raise RuntimeError("no twin suite was built")
    return out / "suites.json"


def lineage_nodes(dataset: Path) -> list[dict[str, Any]]:
    """The stored DAG, or one recomputed from the task files when none was stored."""
    return lineage.read(dataset / lineage.FILENAME) or lineage.build(dataset)


def lineage_slice(nodes: list[dict[str, Any]], bundle_id: str) -> list[dict[str, Any]]:
    """Every node connected to a bundle: its ancestors, descendants and their ancestors."""
    by_id = {n["id"]: n for n in nodes}
    children: dict[str, list[str]] = {}
    for n in nodes:
        for parent in n["parents"]:
            children.setdefault(parent, []).append(n["id"])
    seeds = [n["id"] for n in nodes if n["meta"].get("bundle_id") == bundle_id]
    keep: set[str] = set()

    def up(i: str) -> None:
        if i not in keep and i in by_id:
            keep.add(i)
            for parent in by_id[i]["parents"]:
                up(parent)

    for i in seeds:
        up(i)
        for child in children.get(i, []):
            up(child)
    return [n for n in nodes if n["id"] in keep]


def lineage_dot(nodes: list[dict[str, Any]]) -> str:
    """Graphviz source of a lineage slice."""
    colors = {"scene": "#e8eefc", "render": "#e6f4ea", "certificate": "#fef3d7",
              "twin": "#f3e8fd", "eval": "#fde8e8"}  # fmt: skip
    lines = [
        "digraph lineage {",
        "rankdir=LR;",
        'node [shape=box, style="rounded,filled", fontsize=10];',
    ]
    for n in nodes:
        detail = n["meta"].get("role") or n["meta"].get("bundle_id") or ""
        label = f"{n['kind']}\\n{n['id'][:10]}" + (f"\\n{detail}" if detail else "")
        lines.append(f'"{n["id"]}" [label="{label}", fillcolor="{colors[n["kind"]]}"];')
    known = {n["id"] for n in nodes}
    lines += [f'"{p}" -> "{n["id"]}";' for n in nodes for p in n["parents"] if p in known]
    return "\n".join([*lines, "}"])
