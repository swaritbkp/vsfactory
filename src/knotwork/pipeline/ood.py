"""Structural out-of-distribution splits.

A holdout names question families and/or ranges of generation parameters. `--split ood` generates
only held-out structure; every other split given the same holdout generates the complement, so the
two are disjoint by construction in the held-out dimension.

Parameter syntax: ``name=lo:hi`` or ``name=v`` for numeric ranges, ``name=a,b`` for choices.
A numeric holdout must sit at one end of the profile range so the complement stays one range.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

RANGE_PARAMS = {
    "tangle": (
        "strands",
        "braid_length",
        "stroke_px",
        "jitter_amp_px",
        "identity_slots",
        "r2_distractor_pairs",
    ),
    "rack_patch": (
        "strands",
        "braid_length",
        "stroke_px",
        "jitter_amp_px",
        "identity_slots",
        "r2_distractor_pairs",
    ),
    "gauge": ("radius_px",),
    "panel": ("gauges", "extra_strands"),
}
CHOICE_PARAMS = {
    "tangle": ("color_regimes", "styles", "layouts"),
    "rack_patch": ("color_regimes", "styles", "layouts"),
    "gauge": ("scale_mix",),
    "panel": (),
}


@dataclass(frozen=True)
class Holdout:
    profile: dict[str, Any]
    families: list[str] | None
    record: dict[str, Any]


def _bounds(name: str, text: str) -> tuple[int, int]:
    parts = text.replace("..", ":").split(":")
    try:
        lo, hi = int(parts[0]), int(parts[-1])
    except ValueError:
        raise ValueError(f"holdout {name}: expected integers lo:hi, got {text!r}") from None
    if len(parts) > 2 or lo > hi:
        raise ValueError(f"holdout {name}: bad range {text!r}")
    return lo, hi


def _apply_range(section: dict[str, Any], name: str, text: str, ood: bool) -> list[int]:
    lo, hi = _bounds(name, text)
    low, high = (int(v) for v in section[name])
    if ood:
        section[name] = [lo, hi]
    elif hi < low or lo > high:
        pass  # the held-out range lies outside the profile already
    elif lo <= low and hi < high:
        section[name] = [hi + 1, high]
    elif hi >= high and lo > low:
        section[name] = [low, lo - 1]
    else:
        raise ValueError(f"holdout {name}={text} must cover exactly one end of [{low}, {high}]")
    return [lo, hi]


def _apply_choice(section: dict[str, Any], name: str, text: str, ood: bool) -> list[str]:
    held = [c.strip() for c in text.split(",") if c.strip()]
    known = section[name]
    unknown = sorted(set(held) - set(known))
    if unknown or not held:
        raise ValueError(f"holdout {name}: unknown choices {unknown}; known {sorted(known)}")
    kept = {k: v for k, v in known.items() if (k in held) is ood}
    if not kept:
        raise ValueError(f"holdout {name} leaves nothing to sample")
    section[name] = kept
    return held


def resolve(
    plugin: str,
    split: str,
    profile: dict[str, Any],
    families: list[str] | None,
    holdout_families: list[str] | None = None,
    holdout_params: list[str] | None = None,
) -> Holdout:
    """Profile and family list for this split under the holdout (ValueError on a bad request)."""
    held_families, held_params = list(holdout_families or []), list(holdout_params or [])
    ood = split == "ood"
    if ood and not held_families and not held_params:
        raise ValueError("--split ood needs --holdout-family and/or --holdout-param")
    if not held_families and not held_params:
        return Holdout(profile, families, {})
    out = copy.deepcopy(profile)
    section = out[plugin]
    known = list(section["families"])
    unknown = sorted(set(held_families) - set(known))
    if unknown:
        raise ValueError(f"unknown holdout families for {plugin}: {unknown}; known {known}")
    chosen = list(families or known)
    if held_families:
        chosen = held_families if ood else [f for f in chosen if f not in held_families]
        if not chosen:
            raise ValueError("the holdout removes every family")
    record: dict[str, Any] = {"split": split, "families": held_families, "params": {}}
    for item in held_params:
        name, sep, text = item.partition("=")
        name = name.removeprefix(f"{plugin}.")
        if not sep:
            raise ValueError(f"holdout parameter {item!r} must look like name=value")
        if name in RANGE_PARAMS[plugin]:
            record["params"][name] = _apply_range(section, name, text, ood)
        elif name in CHOICE_PARAMS[plugin]:
            record["params"][name] = _apply_choice(section, name, text, ood)
            if ood and name == "color_regimes":
                section["benchmark_color_regimes"] = list(section[name])
        else:
            allowed = sorted(RANGE_PARAMS[plugin] + CHOICE_PARAMS[plugin])
            raise ValueError(f"unknown holdout parameter {name!r} for {plugin}; known {allowed}")
    return Holdout(out, chosen if held_families else families, record)
