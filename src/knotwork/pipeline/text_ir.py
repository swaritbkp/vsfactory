"""Text twins: the same task semantics as structured text, with no image.

A model that solves the text twin but not the picture failed to see; one that fails both failed to
reason. The text states the primitive facts a viewer would have to read (row exchanges, wiring,
tick and needle angles), never the derived answer.
"""

from __future__ import annotations

from typing import Any

from knotwork.knotcore.contracts import SceneIR
from knotwork.plugins.gauge.scale import ZONES, angle
from knotwork.plugins.tangle import braid


def _rows(n: int, word: list[Any]) -> list[str]:
    """The braid as row exchanges, naming the upper / lower arrival that passes over."""
    letters = [None if x is None else tuple(int(v) for v in x) for x in word]
    row = list(range(n))  # row[r] = strand currently in row r
    lines: list[str] = []
    for c in sorted(braid.compose(n, letters)[1], key=lambda c: c.letter):
        letter = letters[c.letter]
        assert letter is not None
        i = letter[0]
        upper = "upper" if row[i] == c.over else "lower"
        lines.append(
            f"{len(lines) + 1}. The cables in rows {i + 1} and {i + 2} exchange rows; the one "
            f"arriving in the {upper} row passes over."
        )
        row[i], row[i + 1] = row[i + 1], row[i]
    return lines or ["No cable changes row."]


def _dial(label: str, d: dict[str, Any]) -> list[str]:
    vmin, vmax = float(d["vmin"]), float(d["vmax"])
    count = len(d["labels"])
    ticks = ", ".join(
        f"{text} at {angle(vmin + k * (vmax - vmin) / (count - 1), vmin, vmax):.1f}"
        for k, text in enumerate(d["labels"])
    )
    bounds = [vmin, *d["zones"], vmax]
    arcs = "; ".join(
        f"{name} from {angle(bounds[k], vmin, vmax):.1f} to {angle(bounds[k + 1], vmin, vmax):.1f}"
        for k, name in enumerate(d.get("zone_order") or ZONES)
    )
    return [
        f"{label}: printed tick labels with their angles (degrees counter-clockwise from the right): {ticks}.",
        f"{label}: coloured arcs: {arcs}.",
        f"{label}: the needle points at {angle(float(d['value']), vmin, vmax):.1f} degrees.",
    ]


def from_scene(scene: SceneIR, question: str) -> str:
    """Structured description of a sampled scene followed by the question."""
    pay = scene.payload
    lines = ["This is a text description of a diagram. Answer from the description alone.", ""]
    if scene.plugin == "gauge":
        lines += _dial("Gauge", pay["dial"])
    else:
        n = int(pay["n"])
        if scene.plugin in ("tangle", "rack_patch"):
            starts = ", ".join(f"row {r + 1}: {t}" for r, t in enumerate(pay["start_labels"]))
        else:
            starts = ", ".join(
                f"row {r + 1}: {pay['sockets'].get(str(r), 'unlabelled cable')}" for r in range(n)
            )
        lines += [
            f"{n} cables run from left to right in rows 1 (top) to {n}.",
            f"Left ends -- {starts}.",
            "",
        ]
        lines += ["Along the way, from left to right:", *_rows(n, pay["word"]), ""]
        if scene.plugin in ("tangle", "rack_patch"):
            ends = ", ".join(f"row {r + 1}: {t}" for r, t in enumerate(pay["end_labels"]))
            lines.append(f"Right ends -- {ends}.")
        else:
            wiring = ", ".join(
                f"row {r + 1}: " + (f"gauge G{g + 1}" if g >= 0 else "a blank pad")
                for r, g in enumerate(pay["dest_gauge"])
            )
            ordered = "Right ends" if not pay.get("mirror") else "Far ends"
            lines.append(f"{ordered} -- {wiring}.")
            for i, d in enumerate(pay["dials"]):
                lines += ["", *_dial(f"Gauge G{i + 1}", d)]
    return "\n".join([*lines, "", question])


def from_certificate(task: dict[str, Any], question: str | None = None) -> str | None:
    """Fallback for a bundle whose scene cannot be re-sampled: a gauge reading or zone question can
    be restated from the kernel's own evidence (tick angles, needle angle). `question` restates a
    failed read sub-question of a panel item. Other tasks would leak their answer, so None."""
    steps = task["certificate"]["steps"]
    if not steps or (task["plugin"] != "gauge" and question is None):
        return None
    last = steps[-1]
    evidence = last.get("evidence", {})
    lines = ["This is a text description of a gauge. Answer from the description alone.", ""]
    if "ticks" in evidence:
        ticks = ", ".join(f"{t['ocr']} at {t['theta']:.1f}" for t in evidence["ticks"])
        lines += [
            f"Printed tick labels with their angles (degrees counter-clockwise from the right): {ticks}.",
            f"The needle points at {evidence['theta_needle']:.1f} degrees.",
        ]
    elif "zones" in evidence:
        arcs = "; ".join(
            f"{name} covers {lo:.1f} to {hi:.1f}" for name, (lo, hi) in evidence["zones"].items()
        )
        lines += [
            f"Coloured arcs, in degrees of clockwise travel from the scale start: {arcs}.",
            f"The needle has travelled {evidence['phi_needle']:.1f} degrees.",
        ]
    else:
        return None
    return "\n".join([*lines, "", question or str(task["question"]["text"])])
