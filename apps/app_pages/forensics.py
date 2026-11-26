"""Forensics: run KPIs, a failure beside its counterfactual twins, quarantine and fault proofs."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from knotwork.knotcore.contracts import FAULT_EXPECTED, FAULT_PLUGIN, FaultCode
from knotwork.pipeline.counterfactual import run_counterfactual
from knotwork.pipeline.io import read_json
from knotwork.render.canvas import REPO_ROOT
from knotwork.studio import factory, forensics, present
from studio_ui import OUT, dataset_select, relative, resolve, run_with_status, verdict_badge

SECTIONS = ("Counterfactual twins", "Quarantine", "Fault proofs")
CARDS = (
    ("original", "Original"),
    ("perceptual_clean", "Perceptual Clean"),
    ("palette_swap", "Palette Swap"),
    ("reasoning_step_down", "Step Down"),
    ("text_ir_twin", "Text IR"),
)
CARD_HEIGHT = 250


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _kpis(dataset: Path | None) -> None:
    kpi = present.run_kpis(dataset) if dataset else {"yield_rate": None, "quarantine_rate": None}
    cover = present.fault_coverage()
    a, b, c = st.columns(3)
    a.metric("Yield", _percent(kpi["yield_rate"]), border=True,
             help="Share of rendered candidates the kernel certified in the selected run.")  # fmt: skip
    b.metric("Quarantine rate", _percent(kpi["quarantine_rate"]), border=True,
             help="Share of clean candidates the kernel rejected.")  # fmt: skip
    caught = "n/a" if cover["caught"] is None else f"{cover['caught']} / {cover['total']}"
    source = f"From {relative(cover['report'])}." if cover["report"] else "Run `knot mutate`."
    c.metric("Fault matrix coverage", caught, border=True,
             help=f"Injected faults caught with their exact code. {source}")  # fmt: skip


# ------------------------------------------------------------------ twins
def _build(data: str, failures: str, out: Path) -> None:
    code = run_with_status(
        "Build twin suites", lambda: run_counterfactual(resolve(data), resolve(failures), out)
    )
    if code not in (None, 0):
        st.warning("No suite was built: the data folder does not hold the failed bundles.")


def _source_controls() -> Path:
    with st.container(horizontal=True, vertical_alignment="bottom"):
        folder = st.text_input("Report folder", "artifacts/studio/counterfactual", key="cf_out")
        with st.popover("Build from failures", icon=":material/construction:"):
            data = st.text_input("Data folder", "samples/live_failures", key="cf_data")
            failures = st.text_input(
                "Failures file or folder", "samples/live_failures", key="cf_fail"
            )
            build = st.button("Build twin suites", type="primary", width="stretch", key="cf_go")
        demo = st.button("Demo suite", icon=":material/auto_awesome:", key="cf_demo",
                         help="Generate a fresh item and build all its twins. No model failed it.")  # fmt: skip
    out = resolve(folder)
    if build:
        _build(data, failures, out)
    if demo:
        run_with_status("Generate a demo item and its twins", lambda: forensics.demo_suite(out))
    return out


def _card_body(kind: str, suite: dict[str, Any]) -> None:
    if kind == "original":
        if suite["image"] and Path(suite["image"]).is_file():
            st.image(suite["image"], width="stretch", alt="The failed item")
        st.markdown(f"Certified **{suite['answer']['canonical']}**")
        for model, reply in sorted(suite["replies"].items()):
            st.caption(f"{present.short_model_name(model)}: {reply[:60]}")
        return
    twin = suite["twins"][kind]
    if twin["status"] == "unavailable":
        st.badge("Unavailable", color="gray")
        st.caption(twin["reason"])
        return
    if twin.get("image") and Path(twin["image"]).is_file():
        st.image(twin["image"], width="stretch", alt=f"{kind} twin")
    if twin.get("prompt"):
        with st.container(height=CARD_HEIGHT, border=False):
            st.text(twin["prompt"])
    st.markdown(f"Certified **{twin['answer']['canonical']}**")
    if twin["status"] == "quarantined":
        st.badge("Quarantined", color="red", help=", ".join(twin.get("codes", [])))


def _twins() -> None:
    out = _source_controls()
    report = out / "suites.json"
    if not report.is_file():
        st.info(
            "No report here yet. Press **Demo suite** for a full set of twins, or open **Build from "
            "failures** to mine the committed live failures (text twins only).",
            icon=":material/troubleshoot:",
        )
        return
    suites = read_json(report)["suites"]
    if not suites:
        st.warning("The report holds no suites.")
        return
    by_label = {f"{s['family']} · {s['bundle_id']}": s for s in suites}
    suite = by_label[st.selectbox("Item", list(by_label), key="cf_item")]
    bid = suite["bundle_id"]
    recorded = present.load_twin_results(out).get(bid, {})
    replies = {
        kind: str(st.session_state.get(f"reply_{bid}_{kind}", recorded.get(kind, "")))
        for kind in present.DIAGNOSTIC_KINDS
    }
    solved = present.grade_twins(suite, replies)
    verdict, reason, colour = present.diagnose(solved)
    with st.container(border=True, horizontal=True, vertical_alignment="center"):
        st.badge(verdict, color=colour, icon=":material/diagnosis:")
        st.markdown(reason, width="content")
    st.markdown(suite.get("failed_question") or suite["question"].split("\n")[0])
    if not suite["replies"]:
        st.caption("No model reply is recorded for this item.")
    for column, (kind, title) in zip(st.columns(len(CARDS), border=True), CARDS, strict=True):
        with column:
            st.markdown(f"**{title}**")
            _card_body(kind, suite)
            if kind != "original" and suite["twins"][kind]["status"] in ("certified", "text"):
                st.text_input(
                    f"Model reply to {title}", recorded.get(kind, ""), key=f"reply_{bid}_{kind}",
                    placeholder="model reply", label_visibility="collapsed",
                )  # fmt: skip
                if kind in solved:
                    st.badge("Solved" if solved[kind] else "Failed",
                             color="green" if solved[kind] else "red")  # fmt: skip


# ------------------------------------------------------------------ quarantine
def _quarantine(dataset: Path | None) -> None:
    if dataset is None:
        return
    records = forensics.quarantine(dataset)
    rows = forensics.rejection_pareto(records)
    if not rows:
        st.success("No candidate was quarantined in this run.", icon=":material/check_circle:")
        return
    chart, detail = st.columns([3, 2], gap="large")
    with chart:
        frame = pd.DataFrame(rows)
        base = alt.Chart(frame).encode(y=alt.Y("code:N", sort=None, title=None))
        bars = base.mark_bar().encode(
            x=alt.X("count:Q", title="rejections"), tooltip=list(frame.columns)
        )
        st.altair_chart(bars, width="stretch", alt="Rejection codes by frequency")
    with detail:
        labels = {
            f"{i + 1}. {r['primary_code']} · {r['seed_path']}": r for i, r in enumerate(records)
        }
        record = labels[st.selectbox("Quarantined candidate", list(labels), key="q_record")]
        st.caption(record.get("detail", ""))
        if st.button("Re-render and re-certify", icon=":material/replay:", key="q_replay"):
            target = OUT / "quarantine" / uuid.uuid4().hex[:8]
            got = run_with_status(
                "Reproduce the candidate", lambda: forensics.reproduce(dataset, record, target)
            )
            if got is not None:
                cert = got.certificate.model_dump(mode="json")
                verdict_badge(cert["verdict"])
                st.image(factory.witness_overlay(got.dir, cert), width="stretch",
                         alt="Reproduced candidate with failed checks marked")  # fmt: skip


# ------------------------------------------------------------------ faults
def _proof(fault: FaultCode, seed: int) -> list[factory.Generated]:
    run = OUT / "faults" / uuid.uuid4().hex[:8]
    plugin = FAULT_PLUGIN[fault]
    for family in factory.families_of(plugin):
        try:
            bad = factory.generate_one(plugin, family, seed, run / "faulty", fault=fault)
        except RuntimeError:
            continue
        # the same attempt without the fault is the clean control
        good = factory.generate_one(
            plugin, family, seed, run / "clean", first_attempt=bad.attempts - 1
        )
        return [good, bad]
    raise RuntimeError("no scene admits this fault at this seed")


def _faults() -> None:
    with st.container(horizontal=True, vertical_alignment="bottom"):
        fault = FaultCode(
            st.selectbox(
                "Fault", [f.value for f in FaultCode], key="fp_fault",
                format_func=lambda v: f"{v} · {FAULT_PLUGIN[FaultCode(v)]} · {FAULT_EXPECTED[FaultCode(v)].value}",
            )
        )  # fmt: skip
        seed = int(st.number_input("Seed", 0, 2**31 - 1, 3, key="fp_seed"))
        go = st.button("Build proof strip", type="primary", key="fp_go")
    if not go:
        st.caption("Renders one scene twice, with and without the fault, and certifies both.")
        return
    pair = run_with_status("Inject and certify", lambda: _proof(fault, seed))
    if pair is None:
        return
    expected = FAULT_EXPECTED[fault].value
    for column, label, made in zip(
        st.columns(2, border=True), ("Clean control", "Faulty"), pair, strict=True
    ):
        cert = made.candidate.certificate.model_dump(mode="json")
        with column:
            st.markdown(f"**{label}**")
            verdict_badge(cert["verdict"])
            st.image(factory.witness_overlay(made.candidate.dir, cert), width="stretch", alt=label)
            if label == "Faulty":
                caught = expected in cert["codes"]
                st.badge(f"{expected} {'raised' if caught else 'NOT raised'}",
                         color="green" if caught else "red")  # fmt: skip
            st.caption(", ".join(cert["codes"]) or "no codes")


# ------------------------------------------------------------------ page
prefer = relative(REPO_ROOT / "artifacts" / "selftest" / "data" / "panel")
picker, _ = st.columns([2, 3])
with picker:
    run = dataset_select("Run", "fx_dataset", prefer)
_kpis(run)
section = st.segmented_control(
    "Section", SECTIONS, default=SECTIONS[0], required=True, key="fx_section",
    label_visibility="collapsed",
)  # fmt: skip
if section == "Quarantine":
    _quarantine(run)
elif section == "Fault proofs":
    _faults()
else:
    _twins()
