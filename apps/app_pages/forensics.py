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
from knotwork.studio import factory, fault_proofs, forensics, present
from studio_ui import OUT, dataset_select, relative, resolve, run_with_status, verdict_badge

SECTIONS = ("Counterfactual twins", "Quarantine", "Fault proofs")
IMAGE_TWINS = (
    ("perceptual_clean", "Perceptual Clean"),
    ("palette_swap", "Palette Swap"),
    ("reasoning_step_down", "Step Down"),
)
TWIN_TITLES = {**dict(IMAGE_TWINS), "text_ir_twin": "Text IR"}
TWIN_NOTES = {
    "perceptual_clean": "Clutter and surface noise removed; same answer.",
    "palette_swap": "Colours changed; same answer.",
    "reasoning_step_down": "One reasoning step removed; new answer.",
}
MAX_SEGMENTS = 4
CARD_HEIGHT = 380
SAMPLE_REPORT = "samples/demo_forensics"  # committed sample: real twins, written replies
BUILD_REPORT = "artifacts/studio/counterfactual"


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
def _request(action: str) -> None:
    """Button callback: new reports are written outside the committed sample and then shown."""
    st.session_state.cf_out = BUILD_REPORT
    st.session_state.cf_action = action


def _source_controls() -> Path:
    st.session_state.setdefault("cf_out", SAMPLE_REPORT)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.text_input("Report folder", key="cf_out")
        with st.popover("Build from failures", icon=":material/construction:"):
            data = st.text_input("Data folder", "samples/live_failures", key="cf_data")
            failures = st.text_input(
                "Failures file or folder", "samples/live_failures", key="cf_fail"
            )
            st.button("Build twin suites", type="primary", width="stretch", key="cf_go",
                      on_click=_request, args=("build",))  # fmt: skip
        st.button("Demo suite", icon=":material/auto_awesome:", key="cf_demo", on_click=_request,
                  args=("demo",), help="Generate a fresh item and build all its twins. No model "
                  "failed it.")  # fmt: skip
    out = resolve(str(st.session_state.cf_out))
    action = st.session_state.pop("cf_action", None)
    if action == "build":
        code = run_with_status(
            "Build twin suites",
            lambda: run_counterfactual(resolve(data), resolve(failures), out),
        )
        if code not in (None, 0):
            st.warning("No suite was built: the data folder does not hold the failed bundles.")
    elif action == "demo":
        run_with_status("Generate a demo item and its twins", lambda: forensics.demo_suite(out))
    return out


def _image_path(text: str | None) -> Path | None:
    """Report images are absolute, or repository-relative in the committed sample."""
    if not text:
        return None
    path = resolve(text)
    return path if path.is_file() else None


def _reply_lines(suite: dict[str, Any]) -> None:
    if not suite["replies"]:
        st.caption("No model reply is recorded for this item.")
    for model, reply in sorted(suite["replies"].items()):
        st.markdown(f"{present.short_model_name(model)} replied **{reply[:60]}**")


def _solved_badge(kind: str, twin: dict[str, Any], solved: dict[str, bool]) -> None:
    if twin["status"] == "quarantined":
        st.badge("Quarantined", color="red", icon=":material/block:",
                 help=", ".join(twin.get("codes", [])))  # fmt: skip
    elif kind not in solved:
        st.badge("No reply yet", color="gray", icon=":material/hourglass_empty:")
    elif solved[kind]:
        st.badge("Model solved it", color="green", icon=":material/check_circle:")
    else:
        st.badge("Model failed", color="red", icon=":material/cancel:")


def _twin_card(kind: str, title: str, suite: dict[str, Any], solved: dict[str, bool]) -> None:
    """One image twin: picture, certified answer and whether the model solved it."""
    twin = suite["twins"][kind]
    st.markdown(f"**{title}**")
    st.caption(TWIN_NOTES[kind])
    if twin["status"] == "unavailable":
        st.badge("Not available", color="gray")
        st.caption(twin["reason"])
        return
    picture = _image_path(twin.get("image"))
    if picture:
        st.image(str(picture), width="stretch", alt=f"{title} twin")
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown(f"Certified **{twin['answer']['canonical']}**", width="content")
        _solved_badge(kind, twin, solved)


def _diagnosis_card(suite: dict[str, Any], solved: dict[str, bool]) -> None:
    verdict, reason, colour = present.diagnose(solved)
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.badge(verdict, color=colour, icon=":material/diagnosis:")
            if suite.get("sample"):
                st.badge("Sample demo replies", color="gray", icon=":material/science:")
        st.markdown(reason)
    text = suite["twins"].get("text_ir_twin", {})
    with st.container(border=True):
        st.markdown("**Text IR twin**")
        st.caption("The same question with the scene given as text: no seeing required.")
        if text.get("status") == "unavailable" or not text:
            st.badge("Not available", color="gray")
            st.caption(text.get("reason", "This item has no text twin."))
            return
        if text.get("prompt"):
            with st.container(height=CARD_HEIGHT, border=False):
                st.code(text["prompt"], language=None, wrap_lines=True)
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f"Certified **{text['answer']['canonical']}**", width="content")
            _solved_badge("text_ir_twin", text, solved)


def _twins() -> None:
    out = _source_controls()
    report = out / "suites.json"
    if not report.is_file():
        st.info(
            "No twin report in this folder yet. Press **Demo suite** for a full set of twins, or "
            "open **Build from failures** to mine the committed live failures (text twins only).",
            icon=":material/troubleshoot:",
        )
        return
    suites = read_json(report)["suites"]
    if not suites:
        st.warning("This report has no suites. Build one with Demo suite.")
        return
    recorded_all = present.load_twin_results(out)
    if any(s.get("sample") for s in suites):
        st.warning(
            "Sample data: the items and twins are real certified renders, but the model replies "
            "were written for this demo. They are not results from any model.",
            icon=":material/science:",
        )
    by_label: dict[str, dict[str, Any]] = {}
    for s in suites:
        graded = present.grade_twins(s, recorded_all.get(s["bundle_id"], {}))
        name = s.get("sample_title") or f"{s['family']} {s['bundle_id'][-6:]}"
        by_label[f"{name}: {present.diagnose(graded)[0]}"] = s
    labels = list(by_label)
    if len(labels) <= MAX_SEGMENTS:
        chosen = st.segmented_control("Item", labels, default=labels[0], required=True,
                                      key="cf_item", label_visibility="collapsed")  # fmt: skip
    else:
        chosen = st.selectbox("Item", labels, key="cf_item")
    suite = by_label[str(chosen)]
    bid = suite["bundle_id"]
    recorded = recorded_all.get(bid, {})
    replies = {
        kind: str(st.session_state.get(f"reply_{bid}_{kind}", recorded.get(kind, "")))
        for kind in present.DIAGNOSTIC_KINDS
    }
    solved = present.grade_twins(suite, replies)
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        st.markdown("**Original item**")
        original = _image_path(suite["image"])
        if original:
            st.image(str(original), width="stretch", alt="The item the model failed")
        st.markdown(suite.get("failed_question") or suite["question"].split("\n")[0])
        st.markdown(f"Certified answer **{suite['answer']['canonical']}**")
        _reply_lines(suite)
    with right:
        _diagnosis_card(suite, solved)
    for column, (kind, title) in zip(st.columns(3, border=True), IMAGE_TWINS, strict=True):
        with column:
            _twin_card(kind, title, suite, solved)
    with st.expander("Test custom model replies", icon=":material/edit_note:"):
        st.caption("Type what a model answered on each twin; the diagnosis updates as you go.")
        inputs = st.columns(len(present.DIAGNOSTIC_KINDS))
        for column, kind in zip(inputs, present.DIAGNOSTIC_KINDS, strict=True):
            twin = suite["twins"].get(kind, {})
            column.text_input(
                TWIN_TITLES[kind], recorded.get(kind, ""), key=f"reply_{bid}_{kind}",
                placeholder="model reply", disabled=twin.get("status") not in ("certified", "text"),
            )  # fmt: skip


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
@st.cache_data(show_spinner=False, max_entries=32)
def _proof(fault_id: str, seed: int) -> list[fault_proofs.ProofSide]:
    """A proof pair per (fault, seed): built once per server, then instant."""
    return fault_proofs.build(FaultCode(fault_id), seed, OUT / "faults")


@st.cache_resource(show_spinner=False)
def _sample() -> tuple[FaultCode, int, list[fault_proofs.ProofSide]] | None:
    return fault_proofs.load_sample()


def _fault_label(value: str) -> str:
    fault = FaultCode(value)
    return f"{value} · {FAULT_PLUGIN[fault]} · {FAULT_EXPECTED[fault].value}"


def _strip(fault: FaultCode, sides: list[fault_proofs.ProofSide]) -> None:
    expected = fault_proofs.expected_code(fault)
    for column, side in zip(st.columns(2, border=True), sides, strict=True):
        with column:
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(f"**{side.label}**", width="content")
                verdict_badge(side.verdict)
                if side.label == fault_proofs.LABELS[1]:
                    hit = fault_proofs.caught(fault, side)
                    st.badge(f"{expected} {'raised' if hit else 'NOT raised'}",
                             color="green" if hit else "red",
                             icon=":material/gpp_good:" if hit else ":material/gpp_bad:")  # fmt: skip
            st.image(side.image, width="stretch", alt=f"{side.label} with witness overlay")
            st.caption(", ".join(side.codes) or "no codes: certified")


def _faults() -> None:
    sample = _sample()
    values = [f.value for f in FaultCode]
    default_fault, default_seed = (sample[0], sample[1]) if sample else (FaultCode.M1, 3)
    st.session_state.setdefault("fp_fault", default_fault.value)
    st.session_state.setdefault("fp_seed", default_seed)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        fault = FaultCode(st.selectbox("Fault", values, key="fp_fault", format_func=_fault_label))
        seed = int(st.number_input("Seed", 0, 2**31 - 1, key="fp_seed"))
        go = st.button("Build proof strip", type="primary", key="fp_go")
    st.caption("One scene rendered twice, with and without the fault; both go through the kernel.")
    if go:
        st.session_state.fp_built = (fault.value, seed)
    if sample and (fault, seed) == (sample[0], sample[1]):
        _strip(fault, sample[2])
        st.caption("Committed default proof (`samples/fault_proof/`): shown instantly on open.")
        return
    if st.session_state.get("fp_built") != (fault.value, seed):
        st.info(
            "Press **Build proof strip** to render this fault and seed.", icon=":material/build:"
        )
        return
    sides = run_with_status("Inject and certify", lambda: _proof(fault.value, seed))
    if sides is not None:
        _strip(fault, sides)


# ------------------------------------------------------------------ page
prefer = relative(REPO_ROOT / "artifacts" / "selftest" / "data" / "panel")
picker, _ = st.columns([2, 3])
with picker:
    run = dataset_select("Run", "fx_dataset", prefer)
_kpis(run)
st.session_state.setdefault("fx_section", SECTIONS[0])  # Model Arena may preselect a section
section = st.segmented_control(
    "Section", SECTIONS, required=True, key="fx_section",
    label_visibility="collapsed",
)  # fmt: skip
if section == "Quarantine":
    _quarantine(run)
elif section == "Fault proofs":
    _faults()
else:
    _twins()
