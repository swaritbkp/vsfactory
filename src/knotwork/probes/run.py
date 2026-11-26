"""`knot probe` (SPEC 11.2): shortcut probes, answer balance and near-duplicate audit.

Every learned probe is cross-validated with GroupKFold by ``scene_id`` (one fold assignment per
plugin, shared by all probes); model seeds come from ``("probe", name, fold)``. A probe is scored by
exact equality of its predicted target class with the item's target (see `probes.items`).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from knotwork.knotcore.seeds import SeedTree
from knotwork.probes import baselines, gbdt, heuristics, text_only
from knotwork.probes.items import N_BINS, Item, load_items
from knotwork.probes.stats import chi2_uniform, near_duplicate_pairs, phash_bits, wilson_upper

MIN_BALANCE_ITEMS = 50
DEDUP_HAMMING = 4


def fold_ids(items: list[Item], folds: int) -> np.ndarray:
    """GroupKFold fold index per item, grouped by scene_id (deterministic, no shuffling)."""
    from sklearn.model_selection import GroupKFold

    k = max(2, min(folds, len({it.scene_id for it in items})))
    out = np.zeros(len(items), dtype=np.int64)
    idx = np.arange(len(items))
    groups = [it.scene_id for it in items]
    for f, (_, test) in enumerate(GroupKFold(n_splits=k).split(idx, groups=groups)):
        out[test] = f
    return out


def _seed(root: int, name: str, fold: int) -> int:
    return SeedTree(root).child("probe", name, fold).int_seed(32) & 0x7FFFFFFF


def learned_predictions(
    items: list[Item],
    fold: np.ndarray,
    root: int,
    name: str,
    fit: Callable[[list[int], list[int], int], list[str]],
) -> dict[int, str]:
    """Out-of-fold predictions per item index, trained separately per training group."""
    preds: dict[int, str] = {}
    by_group: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_group[it.group].append(i)
    for members in by_group.values():
        for f in sorted(set(int(fold[i]) for i in members)):
            test = [i for i in members if fold[i] == f]
            train = [i for i in members if fold[i] != f]
            if not train:
                continue
            for i, p in zip(test, fit(train, test, _seed(root, name, f)), strict=True):
                preds[i] = p
    return preds


def heuristic_predictions(
    items: list[Item], fold: np.ndarray, name: str
) -> tuple[dict[int, str], dict[int, bool]]:
    """Predicted target per applicable item (+ trace hit for panel items).

    Tangle T1: the heuristic's end label maps directly to an option index. Panel trace families:
    the heuristic picks a gauge; the prediction is the majority training-fold target among items of
    the same family whose heuristic picked the same gauge label (the gauge's true reading is never
    used)."""
    fn = heuristics.HEURISTICS[name]
    picks: dict[int, str | None] = {}
    for i, it in enumerate(items):
        if it.family in heuristics.APPLIES[name]:
            picks[i] = fn(it)
    preds: dict[int, str] = {}
    hits: dict[int, bool] = {}
    for i, label in picks.items():
        it = items[i]
        if it.plugin != "panel":
            opts = it.options or []
            preds[i] = f"idx{opts.index(label)}" if label in opts else "none"
        else:
            hits[i] = label is not None and label == heuristics.true_target_label(it)
            train = [
                j
                for j in picks
                if fold[j] != fold[i] and items[j].family == it.family and picks[j] == label
            ]
            pool = train or [
                j for j in picks if fold[j] != fold[i] and items[j].family == it.family
            ]
            counts = Counter(items[j].target for j in pool)
            preds[i] = min(counts, key=lambda t: (-counts[t], t)) if counts else "none"
    return preds, hits


def scoring_family(item: Item) -> str:
    """Row a probe is reported under: the two premise families form one binary yes/no task."""
    return item.group if item.group == "PREMISE" else item.family


def _entry(items: list[Item], idx: list[int], preds: dict[int, str]) -> dict[str, Any]:
    n = len(idx)
    k = sum(preds.get(i) == items[i].target for i in idx)
    return {
        "acc": round(k / n, 6) if n else 0.0,
        "n": n,
        "wilson_upper": round(wilson_upper(k, n), 6),
        "chance": round(float(np.mean([items[i].chance for i in idx])), 6) if n else 0.0,
    }


def probe_plugin(items: list[Item], folds: int, root: int) -> dict[str, Any]:
    fold = fold_ids(items, folds)
    feats = gbdt.load_features([it.dir / "beauty.png" for it in items])
    texts = [it.text_with_options for it in items]
    targets = [it.target for it in items]

    def fit_gbdt(train: list[int], test: list[int], seed: int) -> list[str]:
        return gbdt.fit_predict(feats[train], [targets[i] for i in train], feats[test], seed)

    def fit_text(train: list[int], test: list[int], seed: int) -> list[str]:
        return text_only.fit_predict(
            [texts[i] for i in train], [targets[i] for i in train], [texts[i] for i in test], seed
        )

    all_preds: dict[str, dict[int, str]] = {}
    trace_hits: dict[str, dict[int, bool]] = {}
    families = {it.family for it in items}
    for name in ("straight_across", "nearest", "color_match"):
        if families & heuristics.APPLIES[name]:
            all_preds[name], trace_hits[name] = heuristic_predictions(items, fold, name)
    all_preds["gbdt_image_stats"] = learned_predictions(
        items, fold, root, "gbdt_image_stats", fit_gbdt
    )
    all_preds["text_only"] = learned_predictions(items, fold, root, "text_only", fit_text)

    pooled: dict[str, Any] = {}
    fam_rows: dict[str, dict[str, Any]] = {}
    for name, preds in all_preds.items():
        idx = sorted(preds)
        if not idx:
            continue
        pooled[name] = _entry(items, idx, preds)
        if trace_hits.get(name):
            h = trace_hits[name]
            pooled[name]["trace_hit_rate"] = round(sum(h.values()) / len(h), 6)
        for fam in sorted({scoring_family(items[i]) for i in idx}):
            fidx = [i for i in idx if scoring_family(items[i]) == fam]
            fam_rows.setdefault(fam, {})[name] = _entry(items, fidx, preds)
    families_out = {}
    for fam in sorted(fam_rows):
        fitems = [it for it in items if scoring_family(it) == fam]
        families_out[fam] = {
            "n": len(fitems),
            "chance": round(float(np.mean([it.chance for it in fitems])), 6),
            "probes": fam_rows[fam],
        }
    return {
        "n": len(items),
        "pooled": {
            "chance": round(float(np.mean([it.chance for it in items])), 6),
            "probes": pooled,
        },
        "families": families_out,
        "baselines": baselines.evaluate(items, fold),
    }


def balance_tests(items: list[Item]) -> dict[str, dict[str, float]]:
    """Chi-square of the correct option index vs uniform per (family, #options) group with >= 50
    items, and of numeric answers over 10 bins per family with >= 50 items."""
    out: dict[str, dict[str, float]] = {}
    groups: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    bins: dict[tuple[str, str], list[int]] = defaultdict(list)
    for it in items:
        if it.option_index is not None and it.n_options >= 2 and it.group != "PREMISE":
            groups[(it.plugin, it.family, it.n_options)].append(it.option_index)
        if it.value_bin is not None:
            bins[(it.plugin, it.family)].append(it.value_bin)
        if it.group == "PREMISE":
            groups[(it.plugin, "PREMISE_TRUTH", 2)].append(int(it.target == "yes"))
    for (plugin, fam, m), vals in sorted(groups.items()):
        if len(vals) >= MIN_BALANCE_ITEMS:
            c = Counter(vals)
            out[f"{plugin}/{fam}/m={m}"] = chi2_uniform([c.get(k, 0) for k in range(m)])
    for (plugin, fam), vals in sorted(bins.items()):
        if len(vals) >= MIN_BALANCE_ITEMS:
            c = Counter(vals)
            out[f"{plugin}/{fam}/bins"] = chi2_uniform([c.get(k, 0) for k in range(N_BINS)])
    return {k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in out.items()}


def dedup(items: list[Item]) -> dict[str, Any]:
    bits = np.vstack([phash_bits(it.dir / "beauty.png") for it in items])
    pairs = near_duplicate_pairs(bits, DEDUP_HAMMING)
    return {"pairs_within_4": pairs, "dup_rate": round(pairs / len(items), 6)}


def run_probes(data: Path, out: Path, seed: int, folds: int) -> int:
    items = load_items(Path(data))
    if not items:
        print(f"probe: no base items under {data}")
        return 1
    by_plugin: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_plugin[it.plugin].append(it)
    report: dict[str, Any] = {
        "plugins": {},
        "balance": {},
        "dedup": {},
        "folds": folds,
        "seed": seed,
    }
    for plugin in sorted(by_plugin):
        its = by_plugin[plugin]
        report["plugins"][plugin] = probe_plugin(its, folds, seed)
        report["balance"].update(balance_tests(its))
        report["dedup"][plugin] = dedup(its)
        pooled = report["plugins"][plugin]["pooled"]
        summary = ", ".join(f"{k}={v['acc']:.3f}" for k, v in pooled["probes"].items())
        print(f"probe: {plugin} n={len(its)} chance={pooled['chance']:.3f} {summary}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    return 0


__all__ = ["balance_tests", "dedup", "fold_ids", "probe_plugin", "run_probes"]
