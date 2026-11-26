"""Vendi Score diversity audit vs. a recolor-only baseline (SPEC.md section 11.4).

Embedding per item: [L2-normalised 32x32 grayscale of beauty.png (1024-d) ; L2-normalised one-hots of
(family, n_strands or n_gauges, layout, style)], L2-normalised. ``K = E E^T`` (cosine), ``lambda =
eigvalsh(K / n)`` clipped at 0, ``VS = exp(-sum lambda log lambda)``.

The baseline re-creates one base scene of the plugin exactly (from the dataset's receipt root and the
bundle's seed path), then renders ``n`` variants whose *only* change is colour: every ``*_rgb`` field
and the cable colour list of the scene payload are tinted / permuted (same geometry, text, values).
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from knotwork.datasets import dataset_dirs
from knotwork.knotcore.contracts import SceneIR
from knotwork.knotcore.seeds import SeedTree, json_digest
from knotwork.render.canvas import REPO_ROOT

EMBED_PX = 32
PROFILE = REPO_ROOT / "configs" / "default_profile.yaml"


def vendi(emb: np.ndarray) -> float:
    """Vendi score of row-normalised embeddings (cosine kernel)."""
    e = np.asarray(emb, dtype=np.float64)
    n = len(e)
    if n == 0:
        return 0.0
    k = e @ e.T / n
    lam = np.clip(np.linalg.eigvalsh(k), 0.0, None)
    lam = lam[lam > 0]
    return float(np.exp(-np.sum(lam * np.log(lam))))


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def image_vector(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        g = im.convert("L").resize((EMBED_PX, EMBED_PX), Image.Resampling.BOX)
    return _unit(np.asarray(g, dtype=np.float64).reshape(-1) / 255.0)


def tokens(family: str, params: dict[str, Any]) -> tuple[str, str, str, str]:
    count = params.get("n_strands") if "n_gauges" not in params else params["n_gauges"]
    layout = params.get("layout", params.get("scale", "-"))
    return (
        family,
        f"n={count if count is not None else 1}",
        f"layout={layout}",
        f"style={params.get('style', '-')}",
    )


def embed(images: list[Path], toks: Sequence[tuple[str, ...]]) -> np.ndarray:
    vocab = [sorted({t[i] for t in toks}) for i in range(4)]
    rows = []
    for img, t in zip(images, toks, strict=True):
        onehot = np.concatenate(
            [np.eye(len(vocab[i]))[vocab[i].index(t[i])] for i in range(4)]
        ).astype(np.float64)
        rows.append(_unit(np.concatenate([image_vector(img), _unit(onehot)])))
    return np.vstack(rows)


def parse_path(path_str: str) -> tuple[str | int, ...]:
    parts: list[str | int] = []
    for comp in path_str.split("/"):
        tag, _, val = comp.partition(":")
        parts.append(int(val) if tag == "i" else val)
    return tuple(parts)


def slot_ordinal(ds: Path, path: tuple[str | int, ...]) -> int | None:
    """Index of the bundle's slot among same-family slots (the generator's balancing hint, D-34),
    from the slot schedule recorded in stats.json; None for datasets without one."""
    stats = ds / "stats.json"
    if len(path) < 3 or path[1] != "slot" or not stats.is_file():
        return None
    plan = json.loads(stats.read_text(encoding="utf-8")).get("schedule")
    if not plan:
        return None
    i = int(path[2])
    return int(plan[:i].count(plan[i]))


def rebuild_scene(ds: Path, task: dict[str, Any], profile: dict[str, Any]) -> SceneIR:
    """Re-sample a bundle's scene exactly as `generate` did (checked against its scene_id)."""
    from knotwork.plugins.registry import get_plugin

    root = int(json.loads((ds / "freshness_receipt.json").read_text(encoding="utf-8"))["root_seed"])
    path = parse_path(task["seed_path"])
    at = SeedTree(root, path)
    prof = dict(
        profile,
        _family=task["family"],
        _split=task["split"],
        _ordinal=slot_ordinal(ds, path),
        _balance_seed=root,
    )
    scene = get_plugin(task["plugin"]).sample_scene(at.child("scene"), prof)
    if json_digest(scene.model_dump(mode="json")) != task["scene_id"]:
        raise ValueError(f"scene of {task['bundle_id']} does not reproduce from its seed path")
    return scene


def _tint(rgb: list[Any], rng: np.random.Generator) -> list[int]:
    return [int(np.clip(round(float(c) * float(rng.uniform(0.75, 1.25))), 0, 255)) for c in rgb]


def recolor(node: Any, rng: np.random.Generator) -> Any:
    """Copy of a payload with every colour tinted and cable colour lists permuted; nothing else."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for k, v in node.items():
            if (k.endswith("rgb") or k == "rgb") and isinstance(v, list) and len(v) == 3:
                out[k] = _tint(v, rng)
            elif k == "colors" and isinstance(v, list):
                perm = [int(i) for i in rng.permutation(len(v))]
                out[k] = [_tint(v[i], rng) for i in perm]
            else:
                out[k] = recolor(v, rng)
        return out
    if isinstance(node, list):
        return [recolor(v, rng) for v in node]
    return copy.deepcopy(node)


def _render_variant(job: dict[str, Any]) -> str:
    from knotwork.plugins.registry import get_plugin

    scene = SceneIR.model_validate(job["scene"])
    rng = SeedTree(int(job["root"])).child("diversity", job["plugin"], job["i"]).rng()
    pay = recolor(scene.payload, rng)
    variant = scene.model_copy(update={"payload": pay})
    out = Path(job["dir"]) / f"v{job['i']:04d}"
    get_plugin(job["plugin"]).render(variant, out)
    return str(out / "beauty.png")


def _map(fn: Any, jobs: list[dict[str, Any]], workers: int) -> list[Any]:
    workers = min(max(1, (os.cpu_count() or 2) - 1) if workers == 0 else workers, len(jobs))
    if workers <= 1:
        return [fn(j) for j in jobs]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, jobs))


def run_diversity(data: Path, out: Path, n: int, seed: int, workers: int = 0) -> int:
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    by_plugin: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for ds in dataset_dirs(Path(data)):
        for line in (ds / "index.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line) if line.strip() else None
            if row and row["role"] == "base":
                by_plugin[row["plugin"]].append((ds, row))
    report: dict[str, Any] = {"plugins": {}, "n": n, "seed": seed}
    rng = SeedTree(seed).child("diversity").rng()
    for plugin in sorted(by_plugin):
        rows = by_plugin[plugin]
        pick = sorted(int(i) for i in rng.choice(len(rows), size=min(n, len(rows)), replace=False))
        tasks, images = [], []
        for i in pick:
            ds, row = rows[i]
            tasks.append(
                (ds, json.loads((ds / row["dir"] / "task.json").read_text(encoding="utf-8")))
            )
            images.append(ds / row["dir"] / "beauty.png")
        scenes = [rebuild_scene(ds, t, profile) for ds, t in tasks]
        toks = [tokens(t["family"], s.params) for (_, t), s in zip(tasks, scenes, strict=True)]
        ours = vendi(embed(images, toks))
        base_ds, base_task = tasks[0]
        tmp = Path(tempfile.mkdtemp(prefix=f"recolor_{plugin}_"))
        try:
            jobs = [{"plugin": plugin, "scene": scenes[0].model_dump(mode="json"), "i": k, "dir": str(tmp),
                     "root": seed} for k in range(len(pick))]  # fmt: skip
            variants = [Path(p) for p in _map(_render_variant, jobs, workers)]
            recol = vendi(embed(variants, [toks[0]] * len(variants)))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        del base_ds
        report["plugins"][plugin] = {
            "n": len(pick),
            "vendi_ours": round(ours, 4),
            "vendi_recolor": round(recol, 4),
            "ratio": round(ours / recol, 4) if recol > 0 else 0.0,
            "recolor_base": base_task["bundle_id"],
        }
        print(f"diversity: {plugin} n={len(pick)} ours={ours:.2f} recolor={recol:.2f}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    return 0


__all__ = ["embed", "rebuild_scene", "recolor", "run_diversity", "tokens", "vendi"]
