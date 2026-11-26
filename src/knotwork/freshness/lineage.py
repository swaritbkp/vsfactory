"""Content-addressed lineage: scene -> render -> certificate -> twin -> eval.

Every node's id is the SHA-256 of its kind, its content digest and its parents' ids, so changing
anything upstream changes every id downstream. A dataset stores the DAG as `lineage.jsonl`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from knotwork.knotcore.seeds import json_digest

KINDS = ("scene", "render", "certificate", "twin", "eval")
FILENAME = "lineage.jsonl"
Node = dict[str, Any]


def node_id(kind: str, digest: str, parents: Sequence[str]) -> str:
    return json_digest({"kind": kind, "digest": digest, "parents": sorted(parents)})


def make_node(kind: str, digest: str, parents: Sequence[str], **meta: Any) -> Node:
    if kind not in KINDS:
        raise ValueError(f"unknown lineage kind {kind!r}")
    return {
        "id": node_id(kind, digest, parents),
        "kind": kind,
        "digest": digest,
        "parents": sorted(parents),
        "meta": meta,
    }


def task_nodes(task: dict[str, Any], certificates: dict[str, str]) -> list[Node]:
    """Nodes of one bundle's `task.json`. `certificates` maps bundle id -> certificate node id and is
    updated; a twin links its base's certificate to its own, so pass bases first."""
    bid = task["bundle_id"]
    scene = make_node("scene", task["scene_id"], [], seed_path=task["seed_path"])
    render = make_node("render", json_digest(task["files"]), [scene["id"]], bundle_id=bid)
    cert = make_node(
        "certificate", json_digest(task["certificate"]), [render["id"]], bundle_id=bid,
        kernel_sha256=task["certificate"]["kernel_sha256"], verdict=task["certificate"]["verdict"],
    )  # fmt: skip
    certificates[bid] = cert["id"]
    nodes = [scene, render, cert]
    twins = task["twins"]
    base = certificates.get(twins.get("base_id") or "")
    if twins["role"] != "base" and base:
        nodes.append(
            make_node("twin", json_digest(twins), [base, cert["id"]], bundle_id=bid,
                      role=twins["role"], edit=twins.get("edit"))
        )  # fmt: skip
    return nodes


def eval_nodes(rows: Iterable[dict[str, Any]], certificates: dict[str, str]) -> list[Node]:
    """One node per evaluation record of a known bundle (records need a `bundle_id`)."""
    out = []
    for row in rows:
        parent = certificates.get(str(row.get("bundle_id")))
        if parent:
            meta = {k: row[k] for k in ("bundle_id", "model", "slot", "correct") if k in row}
            out.append(make_node("eval", json_digest(row), [parent], **meta))
    return out


def read(path: Path) -> list[Node]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def build(dataset: Path, evals: Iterable[dict[str, Any]] = ()) -> list[Node]:
    """The DAG recomputed from a dataset directory's task files (deduplicated, in index order)."""
    certificates: dict[str, str] = {}
    nodes: dict[str, Node] = {}
    index = dataset / "index.jsonl"
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task = json.loads((dataset / row["dir"] / "task.json").read_text(encoding="utf-8"))
        for n in task_nodes(task, certificates):
            nodes.setdefault(n["id"], n)
    for n in eval_nodes(evals, certificates):
        nodes.setdefault(n["id"], n)
    return list(nodes.values())


def verify(nodes: Sequence[Node]) -> list[str]:
    """Problems in a stored DAG: ids that do not match their content, and missing parents."""
    known = {n["id"] for n in nodes}
    problems = []
    for n in nodes:
        if node_id(n["kind"], n["digest"], n["parents"]) != n["id"]:
            problems.append(f"{n['kind']} {n['id'][:12]}: id does not match content")
        problems += [
            f"{n['kind']} {n['id'][:12]}: missing parent {p[:12]}"
            for p in n["parents"]
            if p not in known
        ]
    return problems


def check_dataset(dataset: Path) -> list[str]:
    """Compare the stored `lineage.jsonl` with the DAG recomputed from the files."""
    stored = read(dataset / FILENAME)
    if not stored:
        return [f"no {FILENAME} in {dataset}"]
    fresh = {n["id"] for n in build(dataset)}
    kept = {n["id"] for n in stored if n["kind"] != "eval"}
    problems = verify(stored)
    problems += [f"stored node {i[:12]} no longer matches the files" for i in sorted(kept - fresh)]
    problems += [
        f"file content {i[:12]} is not in the stored lineage" for i in sorted(fresh - kept)
    ]
    return problems
