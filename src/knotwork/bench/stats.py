"""Benchmark statistics (SPEC 12.3): per-crossing error MLE with bootstrap CI, Abbott chance
correction, composition product law, colour ablation, track-switch attribution, twin consistency,
gaslight failure rate. All resampling uses seeded generators."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

N_BOOT = 1000
EPS = 1e-6


def p_correct(p: float, k: np.ndarray, m: np.ndarray) -> np.ndarray:
    s = (1.0 - p) ** k
    return np.asarray(s + (1.0 - s) / m, dtype=np.float64)


def mle_p(k: Sequence[int] | np.ndarray, m: Sequence[float] | np.ndarray,
          correct: Sequence[bool] | np.ndarray) -> float:  # fmt: skip
    """argmax_p sum log P(correct | k, m) (bounded scalar search on (1e-6, 1 - 1e-6))."""
    kk, mm = np.asarray(k, dtype=np.float64), np.asarray(m, dtype=np.float64)
    cc = np.asarray(correct, dtype=bool)
    if len(kk) == 0:
        return float("nan")

    def nll(p: float) -> float:
        q = np.clip(p_correct(p, kk, mm), 1e-12, 1.0 - 1e-12)
        return -float(np.sum(np.where(cc, np.log(q), np.log(1.0 - q))))

    res = minimize_scalar(nll, bounds=(EPS, 1.0 - EPS), method="bounded", options={"xatol": 1e-7})
    return float(res.x)


def bootstrap(stat: Callable[[np.ndarray], float], n: int, rng: np.random.Generator,
              reps: int = N_BOOT) -> tuple[float, float]:  # fmt: skip
    """2.5 / 97.5 percentiles of ``stat(index_sample)`` over ``reps`` resamples of range(n)."""
    if n == 0:
        return float("nan"), float("nan")
    vals = np.array([stat(rng.integers(0, n, size=n)) for _ in range(reps)], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def per_crossing(trials: list[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    k = np.array([t["k"] for t in trials], dtype=np.float64)
    m = np.array([t["m"] for t in trials], dtype=np.float64)
    c = np.array([t["correct"] for t in trials], dtype=bool)
    p_hat = mle_p(k, m, c)
    lo, hi = bootstrap(lambda ix: mle_p(k[ix], m[ix], c[ix]), len(k), rng)
    return {"p_hat": _r(p_hat), "ci": [_r(lo), _r(hi)], "n": int(len(k))}


def abbott(p_obs: float, chance: float) -> float:
    """P* = (P_obs - c) / (1 - c)."""
    return (p_obs - chance) / (1.0 - chance) if chance < 1.0 else float("nan")


def composition(rows: list[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    """rows: per P1 image {trace, read, comp (bools), c_trace, c_read, c_comp (chance)}."""
    if not rows:
        return {"n": 0, "p_trace": None, "p_read": None, "p_comp": None, "chance": {},
                "abbott": {"gap_ci": [None, None]}}  # fmt: skip
    a = {key: np.array([r[key] for r in rows], dtype=np.float64)
         for key in ("trace", "read", "comp", "c_trace", "c_read", "c_comp")}  # fmt: skip

    def parts(ix: np.ndarray) -> tuple[float, float, float, float]:
        t = abbott(float(a["trace"][ix].mean()), float(a["c_trace"][ix].mean()))
        r = abbott(float(a["read"][ix].mean()), float(a["c_read"][ix].mean()))
        c = abbott(float(a["comp"][ix].mean()), float(a["c_comp"][ix].mean()))
        return t, r, c, t * r

    ix_all = np.arange(len(rows))
    t, r, c, pred = parts(ix_all)
    lo, hi = bootstrap(lambda ix: parts(ix)[2] - parts(ix)[3], len(rows), rng)
    return {
        "n": len(rows),
        "p_trace": _r(a["trace"].mean()), "p_read": _r(a["read"].mean()), "p_comp": _r(a["comp"].mean()),
        "chance": {"trace": _r(a["c_trace"].mean()), "read": _r(a["c_read"].mean()), "comp": _r(a["c_comp"].mean())},
        "abbott": {"trace": _r(t), "read": _r(r), "comp_obs": _r(c), "comp_pred": _r(pred),
                   "gap": _r(c - pred), "gap_ci": [_r(lo), _r(hi)]},
    }  # fmt: skip


def color_ablation(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    """pairs: (correct on unique-colour render, correct on matched mono render)."""
    if not pairs:
        return {"n": 0, "acc_unique": None, "acc_mono": None, "delta": None}
    u = float(np.mean([p[0] for p in pairs]))
    m = float(np.mean([p[1] for p in pairs]))
    return {"n": len(pairs), "acc_unique": _r(u), "acc_mono": _r(m), "delta": _r(u - m)}


def track_switch(errors: list[tuple[str, set[str]]]) -> dict[str, Any]:
    """errors: (predicted label, labels reachable by one switch) for each wrong T1 answer."""
    if not errors:
        return {"n_errors": 0, "single_switch_explained_frac": None}
    hits = sum(pred in reach for pred, reach in errors)
    return {"n_errors": len(errors), "single_switch_explained_frac": _r(hits / len(errors))}


def twins(triples: list[tuple[bool, bool, str, str]]) -> dict[str, Any]:
    """triples: (base correct, causal correct, base answer, nuisance answer)."""
    if not triples:
        return {"n": 0, "causal_consistency": None, "nuisance_consistency": None}
    return {
        "n": len(triples),
        "causal_consistency": _r(np.mean([b and c for b, c, _, _ in triples])),
        "nuisance_consistency": _r(np.mean([a == n for _, _, a, n in triples])),
    }


def gaslight(tokens: list[str | None]) -> dict[str, Any]:
    """Fraction of T_GASLIGHT trials whose premise token is "yes" (accepted the false premise)."""
    if not tokens:
        return {"n": 0, "failure_rate": None}
    return {"n": len(tokens), "failure_rate": _r(np.mean([t == "yes" for t in tokens]))}


def _r(x: Any) -> float | None:
    v = float(x)
    return None if not math.isfinite(v) else round(v, 6)


__all__ = ["abbott", "bootstrap", "color_ablation", "composition", "gaslight", "mle_p", "per_crossing",
           "track_switch", "twins"]  # fmt: skip
