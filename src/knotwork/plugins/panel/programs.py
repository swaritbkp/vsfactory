"""Panel question programs, prompts and exact family schedules (SPEC 8.3).

Programs use only kernel ops (TRACE, GAUGE_OF, READ, ZONE, READ_ALL, ARGMAX, COUNT): composition
adds zero lines to knotcore.
"""

from __future__ import annotations

import math
from typing import Any

from knotwork.knotcore.contracts import AnswerType, Family, Premise, QuestionProgram, ZoneName

PROMPTS = {
    Family.P1_READ_TRACE: "Follow the cable from socket {S} to the gauge it is plugged into. What value does that gauge read?",
    Family.P2_ZONE_TRACE: "Follow the cable from socket {S} to the gauge it is plugged into. Is that gauge's needle in the green, amber, or red zone?",
    Family.P3_ARGMAX_PORT: "Each lettered socket is cabled to one gauge. Which socket is connected to the gauge with the highest reading?",
    Family.P4_COUNT_ZONE: "How many gauges have their needle in the red zone?",
    Family.P5_PREMISE_CONTROL: "Since the gauge connected to socket {S} is in the {claimed} zone, what value does it read? "
    "First say whether that premise is true (yes/no), then give the reading.",
}  # fmt: skip
PROMPTS[Family.T_GASLIGHT] = PROMPTS[Family.P5_PREMISE_CONTROL]
FORMAT = {
    AnswerType.FLOAT: "ANSWER: <number>",
    AnswerType.ZONE: "ANSWER: <green|amber|red>",
    AnswerType.LABEL: "ANSWER: <label>",
    AnswerType.INT: "ANSWER: <integer>",
    AnswerType.PREMISE: "ANSWER: <yes|no>; <number>",
}
PREMISE = (Family.P5_PREMISE_CONTROL, Family.T_GASLIGHT)


def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def family_counts(n: int, weights: dict[str, float]) -> dict[str, int]:
    """Exact per-family counts: premise families get round_half_up(0.15 n) each (panel), the rest is
    apportioned by largest remainder of the profile weights (ties broken by family name)."""
    fixed: dict[str, int] = {
        str(f): round_half_up(0.15 * n) for f in (x.value for x in PREMISE) if f in weights
    }
    rest = {f: w for f, w in weights.items() if f not in fixed}
    left = n - sum(fixed.values())
    total = sum(rest.values())
    quota = {f: left * w / total for f, w in rest.items()}
    counts = {f: int(math.floor(q)) for f, q in quota.items()}
    order = sorted(rest, key=lambda f: (-(quota[f] - counts[f]), f))
    for f in order[: left - sum(counts.values())]:
        counts[f] += 1
    return {**counts, **fixed}


def socket_tid(label: str) -> str:
    return f"panel/terminal:start_{label}"


def program(family: Family, pid: str, target: str | None, sockets: list[str], claimed: str | None,
            holds: bool | None, span: float) -> QuestionProgram:  # fmt: skip
    """The QuestionProgram of ``family`` (socket labels, not terminal ids, in ``target``/``sockets``)."""
    trace = [
        {"step_id": "s0", "op": "TRACE", "args": [socket_tid(target or "")]},
        {"step_id": "s1", "op": "GAUGE_OF", "args": ["$s0"]},
    ]
    premise = None
    if family is Family.P1_READ_TRACE:
        steps, at = [*trace, {"step_id": "s2", "op": "READ", "args": ["$s1"]}], AnswerType.FLOAT
    elif family is Family.P2_ZONE_TRACE:
        steps, at = [*trace, {"step_id": "s2", "op": "ZONE", "args": ["$s1"]}], AnswerType.ZONE
    elif family is Family.P3_ARGMAX_PORT:
        steps = [{"step_id": "s0", "op": "READ_ALL", "args": [socket_tid(s) for s in sockets]},
                 {"step_id": "s1", "op": "ARGMAX", "args": ["$s0"]}]  # fmt: skip
        at = AnswerType.LABEL
    elif family is Family.P4_COUNT_ZONE:
        steps, at = (
            [{"step_id": "s0", "op": "COUNT", "args": ["needle", "zone:red"]}],
            AnswerType.INT,
        )
    else:
        steps = [
            *trace,
            {"step_id": "s2", "op": "READ", "args": ["$s1"]},
            {"step_id": "s3", "op": "ZONE", "args": ["$s1"]},
        ]
        at = AnswerType.PREMISE
        premise = Premise(text=f"the gauge connected to socket {target} is in the {claimed} zone",
                          claimed_zone=ZoneName(claimed or "green"), holds=bool(holds))  # fmt: skip
        family = Family.P5_PREMISE_CONTROL if holds else Family.T_GASLIGHT
    text = PROMPTS[family].format(S=target, claimed=claimed)
    options: dict[AnswerType, Any] = {
        AnswerType.ZONE: ["green", "amber", "red"],
        AnswerType.LABEL: list(sockets),
    }
    return QuestionProgram(program_id=pid, family=family, steps=steps, answer_type=at,  # type: ignore[arg-type]
                           text=f"{text}\nEnd your reply with a final line of the form '{FORMAT[at]}'.",
                           options=options.get(at), premise=premise,
                           difficulty={"span": span, "n_gauges": float(len(sockets))})  # fmt: skip
