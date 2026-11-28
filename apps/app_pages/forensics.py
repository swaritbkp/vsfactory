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
from knotwork.studio import factory, fault_proofs, forensics, guide, present
from studio_ui import OUT, dataset_select, relative, resolve, run_with_status, verdict_badge
from ui_chrome import working_set

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
WS_REPORT = OUT / "counterfactual" / "working_set"
COMMITTED = "samples/"


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _kpis(dataset: Path | None) -> None:
    kpi = present.run_kpis(dataset) if dataset else {"yield_rate": None, "quarantine_rate": None}
    cover = present.fault_coverage()
    a, b, c = st.columns(3)
    a.metric("Yield", _percent(kpi["yield_rate"]), border=True,
             help=guide.tip("kpi_yield"))  # fmt: skip
    b.metric("Quarantine rate", _percent(kpi["quarantine_rate"]), border=True,
             help=guide.tip("kpi_quarantine"))  # fmt: skip
    caught = "n/a" if cover["caught"] is None else f"{cover['caught']} / {cover['total']}"
    source = f"From {relative(cover['report'])}." if cover["report"] else "Run `knot mutate`."
    c.metric("Fault matrix coverage", caught, border=True,
             help=f"{guide.tip('kpi_faults')} {source}")  # fmt: skip


# ------------------------------------------------------------------ twins
def _request(action: str) -> None:
    """Button callback: new reports are written outside the committed sample and then shown."""
    st.session_state.cf_out = BUILD_REPORT
    st.session_state.cf_action = action


def _ws_twins_card() -> None:
    """Sleek single-row bar while the committed sample is shown: build twins for the Working Set."""
    ws_report = relative(WS_REPORT)
    if st.session_state.get("cf_out") == ws_report:
        return
    with st.container(border=True, key="cf_ws_card"):
        text, go = st.columns([3.5, 1.5], vertical_alignment="center")
        built = (WS_REPORT / "suites.json").is_file()
        text.markdown(
            "**Working Set twins** — :gray[sample below; generate clean, palette-swap, "
            "step-down & text twins for your item.]"
        )
        label = "Open Working Set twins" if built else "Generate counterfactual twins"
        clicked = go.button(label, type="primary", icon=":material/difference:", width="stretch",
                            key="cf_ws_go", help=guide.tip("fx_ws_twins"))  # fmt: skip
    if clicked and (
        built
        or run_with_status(
            "Render twins for the first Working Set item",
            lambda: forensics.dataset_suite(working_set().root, WS_REPORT),
        )
    ):
        st.session_state.cf_out = ws_report
        st.rerun()


def _source_popover() -> Path:
    st.session_state.setdefault("cf_out", SAMPLE_REPORT)
    with st.popover("Source & build", icon=":material/construction:", width="stretch",
                    help=guide.tip("cf_build")):  # fmt: skip
        st.text_input("Report folder", key="cf_out", help=guide.tip("cf_out"))
        data = st.text_input("Data folder", "samples/live_failures", key="cf_data",
                             help=guide.tip("cf_data"))  # fmt: skip
        failures = st.text_input(
            "Failures file or folder",
            "samples/live_failures",
            key="cf_fail",
            help=guide.tip("cf_fail"),
        )
        c1, c2 = st.columns(2)
        c1.button("Build twin suites", type="primary", width="stretch", key="cf_go",
                  on_click=_request, args=("build",), help=guide.tip("cf_build"))  # fmt: skip
        c2.button("Demo suite", icon=":material/auto_awesome:", width="stretch", key="cf_demo",
                  on_click=_request, args=("demo",), help=guide.tip("cf_demo"))  # fmt: skip
    out = _read(str(st.session_state.cf_out))
    action = st.session_state.pop("cf_action", None)
    if action == "build":
        code = run_with_status(
            "Build twin suites",
            lambda: run_counterfactual(_read(data), _read(failures), out),
        )
        if code not in (None, 0):
            st.warning("No suite was built: the data folder does not hold the failed bundles.")
    elif action == "demo":
        run_with_status("Generate a demo item and its twins", lambda: forensics.demo_suite(out))
    return out


def _read(text: str) -> Path:
    """A folder to read. Committed `samples/` are read from the repository even in the public
    demo (they are read-only files that ship with the image); anything else goes through
    `resolve`, which confines it to the sandbox there."""
    clean = text.strip().replace("\\", "/")
    if clean.startswith(COMMITTED) and ".." not in Path(clean).parts:
        return REPO_ROOT / clean
    return resolve(text)


def _image_path(text: str | None) -> Path | None:
    """Report images are absolute, or repository-relative in the committed sample."""
    if not text:
        return None
    path = _read(text)
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
    _ws_twins_card()
    item_col, src_col = st.columns([3.2, 1.0], vertical_alignment="center")
    with src_col:
        out = _source_popover()
    report = out / "suites.json"
    if not report.is_file():
        st.caption(
            "No twin report in this folder yet. Press Demo suite for a full set of twins, or "
            "open Source & build to mine the committed live failures (text twins only)."
        )
        return
    suites = read_json(report)["suites"]
    if not suites:
        st.warning("This report has no suites. Build one with Demo suite.")
        return
    recorded_all = present.load_twin_results(out)
    by_label: dict[str, dict[str, Any]] = {}
    for s in suites:
        graded = present.grade_twins(s, recorded_all.get(s["bundle_id"], {}))
        name = s.get("sample_title") or f"{s['family']} {s['bundle_id'][-6:]}"
        by_label[f"{name}: {present.diagnose(graded)[0]}"] = s
    labels = list(by_label)
    with item_col:
        if len(labels) <= MAX_SEGMENTS:
            chosen = st.segmented_control("Item", labels, default=labels[0], required=True,
                                          key="cf_item", label_visibility="collapsed",
                                          help=guide.tip("cf_item"))  # fmt: skip
        else:
            chosen = st.selectbox("Item", labels, key="cf_item", help=guide.tip("cf_item"))
    if any(s.get("sample") for s in suites):
        st.warning(
            "Sample data: real certified renders; model replies were written for this demo.",
            icon=":material/science:",
        )
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
                help=guide.tip("cf_reply"),
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
        record = labels[st.selectbox("Quarantined candidate", list(labels), key="q_record",
                                     help=guide.tip("q_record"))]  # fmt: skip
        st.caption(record.get("detail", ""))
        if st.button("Re-render and re-certify", icon=":material/replay:", key="q_replay",
                     help=guide.tip("q_replay")):  # fmt: skip
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


@st.fragment
def _faults() -> None:
    """Live Fault Check: inject a renderer fault, run the kernel, show the verdict and its code.
    A fragment, so a check redraws only this section."""
    sample = _sample()
    values = [f.value for f in FaultCode]
    default_fault, default_seed = (sample[0], sample[1]) if sample else (FaultCode.M1, 3)
    st.session_state.setdefault("fp_fault", default_fault.value)
    st.session_state.setdefault("fp_seed", default_seed)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        fault = FaultCode(st.selectbox("Fault", values, key="fp_fault", format_func=_fault_label,
                                       help=guide.tip("fp_fault")))  # fmt: skip
        seed = int(st.number_input("Seed", 0, 2**31 - 1, key="fp_seed", help=guide.tip("fp_seed")))
        go = st.button("Run live fault check", type="primary", icon=":material/bug_report:",
                       key="fp_go", help=guide.tip("fp_live"))  # fmt: skip
    st.caption("One scene rendered twice, with and without the fault; both go through the kernel.")
    if go:
        st.session_state.fp_built = (fault.value, seed)
    if sample and (fault, seed) == (sample[0], sample[1]):
        _strip(fault, sample[2])
        st.caption("Committed default proof (`samples/fault_proof/`): shown instantly on open.")
        return
    if st.session_state.get("fp_built") != (fault.value, seed):
        st.caption("Press Run live fault check to inject this fault and certify both renders.")
        return
    sides = run_with_status("Inject and certify", lambda: _proof(fault.value, seed))
    if sides is not None:
        _verdict_line(fault, sides)
        _strip(fault, sides)


def _verdict_line(fault: FaultCode, sides: list[fault_proofs.ProofSide]) -> None:
    faulty = sides[-1]
    expected = fault_proofs.expected_code(fault)
    caught = fault_proofs.caught(fault, faulty)
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown("**Kernel verdict on the faulty render**", width="content")
        if faulty.verdict == "CERTIFIED":
            st.badge("CERTIFIED", color="red", icon=":material/gpp_bad:")
        else:
            st.badge("QUARANTINED", color="green", icon=":material/block:")
        st.badge(expected if caught else f"{expected} missing", color="green" if caught else "red",
                 icon=":material/code:")  # fmt: skip


# ------------------------------------------------------------------ page
run_col, sec_col = st.columns([1.6, 2.4], vertical_alignment="bottom")
with run_col:
    run = dataset_select("Run", "fx_dataset", help=guide.tip("fx_run"))
st.session_state.setdefault("fx_section", SECTIONS[0])  # Model Arena may preselect a section
with sec_col:
    section = st.segmented_control(
        "Section", SECTIONS, required=True, key="fx_section",
        label_visibility="collapsed", width="stretch", help=guide.tip("fx_section"),
    )  # fmt: skip
_kpis(run)
if section == "Quarantine":
    _quarantine(run)
elif section == "Fault proofs":
    _faults()
else:
    _twins()
