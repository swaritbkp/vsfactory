"""Visual Task Factory Studio: `uv run streamlit run apps/inspector.py`.

Four tabs over the live Python API (offline only): Factory Studio, Forensics & Counterfactuals,
Human Calibration Lab, Export & Lineage. All logic lives in `knotwork.studio` and the pipeline; this
file only lays out widgets. Output goes under `artifacts/studio/`.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from knotwork.export.manifest import check_manifest
from knotwork.export.run import run_export
from knotwork.freshness import lineage
from knotwork.knotcore.contracts import CONSUMPTION_PX, FAULT_EXPECTED, FAULT_PLUGIN, FaultCode
from knotwork.pipeline import telemetry
from knotwork.pipeline.counterfactual import KINDS, run_counterfactual
from knotwork.pipeline.io import read_json
from knotwork.render.canvas import REPO_ROOT
from knotwork.studio import calibration, factory, forensics

OUT = REPO_ROOT / "artifacts" / "studio"
AUTO = "profile default"


def _session() -> str:
    if "session" not in st.session_state:
        st.session_state.session = uuid.uuid4().hex[:8]
    return str(st.session_state.session)


def _run(label: str, action: Callable[[], Any]) -> Any:
    """Run a live action with a spinner; show failures in the page instead of a traceback."""
    with st.spinner(label):
        try:
            return action()
        except Exception as exc:  # the UI boundary: report every failure to the user
            st.error(f"{type(exc).__name__}: {exc}")
            return None


def _dial(label: str, key: str, lo: float, hi: float, default: float, step: float) -> float | None:
    """A slider that is only applied when its switch is on."""
    if not st.toggle(label, key=f"{key}_on"):
        return None
    return float(st.slider(label, lo, hi, default, step, key=key, label_visibility="collapsed"))


def _dataset_picker(key: str) -> Path | None:
    found = [str(p) for p in forensics.find_datasets()]
    custom = st.text_input("Dataset directory (leave empty to pick below)", key=f"{key}_path")
    picked = st.selectbox("Datasets found", found, key=f"{key}_pick") if found else None
    chosen = custom.strip() or picked
    if not chosen:
        st.info("No dataset yet. Build one in the Factory Studio tab or enter a directory.")
        return None
    path = Path(chosen)
    path = path if path.is_absolute() else REPO_ROOT / path
    if not (path / "index.jsonl").is_file():
        st.warning(f"{path} has no index.jsonl.")
        return None
    return path


def _pareto_chart(rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.success("No rejections recorded.")
        return
    frame = pd.DataFrame(rows)
    base = alt.Chart(frame).encode(x=alt.X("code:N", sort=None, title=None))
    bars = base.mark_bar().encode(
        y=alt.Y("count:Q", title="rejections"), tooltip=list(frame.columns)
    )
    line = base.mark_line(point=True, color="#d55e00").encode(
        y=alt.Y("cumulative_share:Q", title="cumulative share", scale=alt.Scale(domain=[0, 1]))
    )
    st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent"), width="stretch")
    st.dataframe(frame, width="stretch", hide_index=True)


# ====================================================================== tab 1
def _show_item(bundle: Path, certificate: dict[str, Any], key: str) -> None:
    left, right = st.columns(2)
    left.image(factory.beauty(bundle), caption="beauty (1024 px)", width="stretch")
    right.image(
        factory.witness_overlay(bundle, certificate),
        caption="kernel witness overlay: terminals, crossings, needle axis, failed checks",
        width="stretch",
    )
    objects = factory.manifest(bundle).objects
    names = {f"{o.obj_id}: {o.name}": o.obj_id for o in objects}
    a, b = st.columns(2)
    chosen = a.selectbox("Amodal mask", list(names), key=f"{key}_mask")
    a.image(factory.mask_overlay(bundle, names[chosen]), width="stretch")
    b.write("Modal id buffer")
    b.image(factory.id_buffer_image(bundle), width="stretch")
    if st.toggle("Colour-vision preview", key=f"{key}_cvd"):
        p, d = st.columns(2)
        p.image(factory.cvd_preview(bundle, "protanopia"), caption="protanopia", width="stretch")
        d.image(
            factory.cvd_preview(bundle, "deuteranopia"), caption="deuteranopia", width="stretch"
        )
    failed = [c for c in certificate["checks"] if not c["passed"]]
    if failed:
        st.dataframe(pd.DataFrame(failed), width="stretch", hide_index=True)
    with st.expander("Certificate JSON"):
        st.json(certificate)


def _factory_single() -> None:
    controls, view = st.columns([1, 2])
    with controls:
        plugin = st.selectbox("Plugin", factory.plugin_names(), key="fs_plugin")
        family = st.selectbox("Family", factory.families_of(plugin), key="fs_family")
        seed = int(st.number_input("Seed", 0, 2**31 - 1, 7, key="fs_seed"))
        st.markdown("**Reasoning load**")
        load = _dial("Reasoning load (hops, crossings, gauges)", "fs_load", 0.0, 1.0, 0.5, 0.05)
        occlusion = _dial("Occlusion level (distractors)", "fs_occ", 0.0, 1.0, 0.5, 0.05)
        st.markdown("**Visual stress**")
        stress = _dial("Overall visual stress", "fs_stress", 0.0, 1.0, 0.5, 0.05)
        stroke = _dial("Stroke width (px, tangle)", "fs_stroke", 4.0, 14.0, 8.0, 1.0)
        jitter = _dial("Jitter amplitude (px, tangle)", "fs_jitter", 0.0, 24.0, 7.0, 1.0)
        realism = _dial("Surface realism", "fs_real", 0.0, 1.0, 0.5, 0.05)
        noise = _dial("Sensor noise", "fs_noise", 0.0, 1.0, 0.2, 0.05)
        dpi = _dial("Capture resolution scale", "fs_dpi", 0.5, 1.0, 0.8, 0.05)
        regime = st.selectbox("Colour regime", [AUTO, "unique", "partial", "mono"], key="fs_regime")
        faults = [f.value for f in FaultCode if FAULT_PLUGIN[f] == plugin]
        fault = st.selectbox("Inject fault", ["none", *faults], key="fs_fault")
        go = st.button("Generate and certify", type="primary", key="fs_go")
    if go:
        dials = {
            "reasoning_load": load, "occlusion_level": occlusion, "visual_stress": stress,
            "stroke_px": stroke, "jitter_px": jitter, "realism": realism, "noise": noise,
            "dpi_scale": dpi, "color_regime": None if regime == AUTO else regime,
        }  # fmt: skip
        target = OUT / "single" / f"{plugin}-{uuid.uuid4().hex[:8]}"
        made = _run(
            "Rendering and certifying...",
            lambda: factory.generate_one(
                plugin, family, seed, target, factory.difficulty_from(dials),
                None if fault == "none" else FaultCode(fault),
            ),
        )  # fmt: skip
        if made is not None:
            c = made.candidate
            st.session_state.single = {
                "dir": str(c.dir),
                "certificate": c.certificate.model_dump(mode="json"),
                "question": c.question.model_dump(mode="json"),
                "answer": c.answer.canonical,
                "attempts": made.attempts,
                "fault": fault,
            }
    with view:
        item = st.session_state.get("single")
        if not item:
            st.info("Choose the dials and press Generate and certify.")
            return
        cert = item["certificate"]
        if cert["verdict"] == "CERTIFIED":
            st.success(f"CERTIFIED  |  answer {item['answer']}  |  attempt {item['attempts']}")
        else:
            st.error(f"QUARANTINED  |  {', '.join(cert['codes'])}")
        if item["fault"] != "none":
            expected = FAULT_EXPECTED[FaultCode(item["fault"])].value
            caught = "caught" if expected in cert["codes"] else "NOT caught"
            st.write(f"Fault {item['fault']} expects `{expected}`: **{caught}**.")
        st.write(item["question"]["text"])
        _show_item(Path(item["dir"]), cert, "single")


def _factory_batch() -> None:
    ready = [p for p in factory.plugin_names() if factory.batch_ready(p)]
    a, b, c = st.columns(3)
    plugin = a.selectbox("Plugin", ready, key="bt_plugin")
    n = int(b.number_input("Certified base items (N)", 1, 1_000_000, 20, key="bt_n"))
    seed = int(c.number_input("Seed", 0, 2**31 - 1, 0, key="bt_seed"))
    split = a.selectbox("Split", ["dev", "test", "sealed", "ood"], key="bt_split")
    workers = int(b.number_input("Workers (0 = auto)", 0, 64, 0, key="bt_workers"))
    shard = int(c.number_input("Slots per shard (0 = flat)", 0, 100_000, 0, key="bt_shard"))
    twins = a.checkbox("Causal and nuisance twins", True, key="bt_twins")
    held = st.multiselect("Held-out families", factory.families_of(plugin), key="bt_hold")
    params = st.text_input(
        "Held-out parameters, space separated (e.g. strands=9:10 color_regimes=unique)",
        key="bt_param",
    )
    d1, d2, d3 = st.columns(3)
    with d1:
        load = _dial("Reasoning load", "bt_load", 0.0, 1.0, 0.5, 0.05)
    with d2:
        stress = _dial("Visual stress", "bt_stress", 0.0, 1.0, 0.5, 0.05)
    with d3:
        occlusion = _dial("Occlusion level", "bt_occ", 0.0, 1.0, 0.5, 0.05)
    name = st.text_input(
        "Output directory", f"artifacts/studio/batch/{plugin}-{split}-{n}", key="bt_out"
    )
    out = REPO_ROOT / name
    spec = factory.difficulty_from(
        {"reasoning_load": load, "visual_stress": stress, "occlusion_level": occlusion}
    )
    args = factory.batch_command(
        plugin, n, out, seed, split, twins, workers, shard, held, params.split(), spec
    )
    st.code(factory.command_line(args), language="bash")
    job = st.session_state.get("batch")
    running = job is not None and job["process"].poll() is None
    if split == "ood" and not (held or params.strip()):
        st.warning("The ood split needs at least one held-out family or parameter.")
    elif st.button("Launch batch build", type="primary", disabled=running, key="bt_go"):
        log = out.parent / f"{out.name}.log"
        st.session_state.batch = {
            "process": factory.start_batch(args, log), "out": str(out), "n": n, "log": str(log),
        }  # fmt: skip
        st.rerun()
    if job is None:
        return
    done = factory.batch_progress(Path(job["out"]))
    code = job["process"].poll()
    st.progress(min(done / job["n"], 1.0), text=f"{done} of {job['n']} base items")
    if code is None:
        st.button("Refresh progress", key="bt_refresh")
        if st.button("Stop build", key="bt_stop"):
            job["process"].terminate()
    else:
        (st.success if code == 0 else st.error)(f"knot generate exited with code {code}")
        log = Path(job["log"])
        st.text(log.read_text(encoding="utf-8", errors="replace")[-3000:] if log.is_file() else "")
        report = Path(job["out"]) / telemetry.FILENAME
        if report.is_file():
            st.json(read_json(report), expanded=False)


def factory_tab() -> None:
    single, batch = st.tabs(["Single item", "Batch build"])
    with single:
        _factory_single()
    with batch:
        _factory_batch()


# ====================================================================== tab 2
def _quarantine_view() -> None:
    dataset = _dataset_picker("fx")
    if dataset is None:
        return
    records = forensics.quarantine(dataset)
    st.write(f"{len(records)} quarantined candidates")
    _pareto_chart(forensics.rejection_pareto(records))
    report = dataset / telemetry.FILENAME
    if report.is_file():
        with st.expander("Factory telemetry"):
            st.json(read_json(report), expanded=False)
    if not records:
        return
    labels = {f"{i}: {r['primary_code']} {r['seed_path']}": r for i, r in enumerate(records)}
    chosen = labels[st.selectbox("Quarantined candidate", list(labels), key="fx_record")]
    st.json(chosen, expanded=False)
    if st.button("Re-render and re-certify", key="fx_repro"):
        target = OUT / "quarantine" / uuid.uuid4().hex[:8]
        got = _run("Reproducing...", lambda: forensics.reproduce(dataset, chosen, target))
        if got is not None:
            _show_item(got.dir, got.certificate.model_dump(mode="json"), "repro")


def _fault_view() -> None:
    a, b = st.columns(2)
    fault = FaultCode(a.selectbox("Fault", [f.value for f in FaultCode], key="fp_fault"))
    seed = int(b.number_input("Seed", 0, 2**31 - 1, 3, key="fp_seed"))
    plugin = FAULT_PLUGIN[fault]
    st.write(
        f"{fault.value} is injected into `{plugin}` and must raise `{FAULT_EXPECTED[fault].value}`."
    )
    if not st.button("Build proof strip", type="primary", key="fp_go"):
        return

    def build() -> list[Any]:
        run = OUT / "faults" / uuid.uuid4().hex[:8]
        for family in factory.families_of(plugin):
            try:
                bad = factory.generate_one(plugin, family, seed, run / "faulty", fault=fault)
            except RuntimeError:
                continue
            # the same seed path without the fault is the clean control
            good = factory.generate_one(plugin, family, seed, run / "clean")
            return [good, bad]
        raise RuntimeError("no scene admits this fault at this seed")

    pair = _run("Injecting the fault...", build)
    if pair is None:
        return
    for column, label, made in zip(st.columns(2), ("clean control", "faulty"), pair, strict=True):
        cert = made.candidate.certificate.model_dump(mode="json")
        column.image(factory.witness_overlay(made.candidate.dir, cert), width="stretch")
        column.write(f"**{label}**: {cert['verdict']} {', '.join(cert['codes'])}")


def _counterfactual_view() -> None:
    st.write(
        "Visual twins need the failed item's dataset as generated by this version; the committed live "
        "failures come from the v1 run, so they get text twins only where the kernel evidence allows."
    )
    data = st.text_input("Data directory", "samples/live_failures", key="cf_data")
    failures = st.text_input(
        "Failures (jsonl, json or folder)", "samples/live_failures", key="cf_fail"
    )
    out = st.text_input("Report directory", "artifacts/studio/counterfactual", key="cf_out")
    if st.button("Build diagnostic twin suites", type="primary", key="cf_go"):
        code = _run(
            "Building twins...",
            lambda: run_counterfactual(REPO_ROOT / data, REPO_ROOT / failures, REPO_ROOT / out),
        )
        if code not in (None, 0):
            st.warning(
                "No suite was built; check that the data directory holds the failed bundles."
            )
    report = REPO_ROOT / out / "suites.json"
    if not report.is_file():
        return
    for suite in read_json(report)["suites"]:
        st.subheader(f"{suite['bundle_id']} | {suite['family']}")
        st.write(suite.get("failed_question") or suite["question"])
        st.write(
            f"Certified answer **{suite['answer']['canonical']}**; failed replies: {suite['replies']}"
        )
        columns = st.columns(len(KINDS) + 1)
        if suite["image"] and Path(suite["image"]).is_file():
            columns[0].image(suite["image"], caption="original", width="stretch")
        for column, kind in zip(columns[1:], KINDS, strict=True):
            twin = suite["twins"][kind]
            column.markdown(f"**{kind}**")
            if twin.get("image") and Path(twin["image"]).is_file():
                column.image(twin["image"], width="stretch")
            if twin["status"] == "unavailable":
                column.caption(twin["reason"])
                continue
            column.caption(f"{twin['status']} | answer {twin['answer']['canonical']}")
            if twin.get("prompt"):
                column.text(twin["prompt"])


def forensics_tab() -> None:
    quarantined, proofs, twins = st.tabs(["Quarantine", "Fault proofs", "Live failures and twins"])
    with quarantined:
        _quarantine_view()
    with proofs:
        _fault_view()
    with twins:
        _counterfactual_view()


# ====================================================================== tab 3
def calibration_tab() -> None:
    st.write("Blind annotation: the certified answer is never shown while you work.")
    dataset = _dataset_picker("hc")
    if dataset is not None:
        who = st.text_input("Participant / session id", _session(), key="hc_session")
        items = calibration.queue(dataset, who)
        answered = calibration.done_ids(who)
        left = [i for i in items if i["bundle_id"] not in answered]
        st.progress((len(items) - len(left)) / max(len(items), 1), text=f"{len(left)} items left")
        if not left:
            st.success("All items of this dataset are answered.")
        elif st.session_state.get("trial", {}).get("bundle_id") != left[0]["bundle_id"]:
            if st.button("Show next item", type="primary", key="hc_next"):
                st.session_state.trial = {
                    "bundle_id": left[0]["bundle_id"],
                    "start": time.monotonic(),
                }
                st.rerun()
        else:
            item, started = left[0], st.session_state.trial["start"]
            st.image(factory.consumption(Path(item["dir"])), width=CONSUMPTION_PX)
            st.write(item["question"])
            with st.form(f"hc_form_{item['bundle_id']}", clear_on_submit=True):
                if item["options"]:
                    reply = st.radio("Answer", item["options"], index=None, horizontal=True)
                else:
                    reply = st.text_input("Answer (for premise questions: 'yes; 42')")
                if st.form_submit_button("Submit"):
                    if reply in (None, ""):
                        st.warning("Enter an answer first.")
                    else:
                        calibration.record(item, str(reply), time.monotonic() - started, who)
                        del st.session_state.trial
                        st.rerun()
    st.divider()
    if not st.toggle("Show results (reveals answers)", key="hc_results"):
        return
    result = calibration.summary()
    a, b = st.columns(2)
    a.metric("Human responses", result["n_human"])
    b.metric(
        "Human accuracy",
        "-" if result["human_overall"] is None else f"{result['human_overall']:.1%}",
    )
    for title, key in (
        ("Human accuracy by family", "human_by_family"),
        ("Model accuracy by slot (committed live run)", "model_by_slot"),
        ("Model accuracy by task block", "model_by_block"),
        ("Head to head on shared bundles", "matched"),
        ("Human confusions (truth vs response)", "confusion"),
    ):
        st.markdown(f"**{title}**")
        if result[key]:
            st.dataframe(pd.DataFrame(result[key]), width="stretch", hide_index=True)
        else:
            st.caption("No data yet.")
    if calibration.LOG.is_file():
        st.download_button(
            "Download human_calibration.jsonl",
            calibration.LOG.read_bytes(),
            "human_calibration.jsonl",
        )


# ====================================================================== tab 4
def export_tab() -> None:
    dataset = _dataset_picker("ex")
    if dataset is None:
        return
    a, b, c = st.columns(3)
    fmt = a.selectbox("Format", ["all", "hf", "harbor", "rlvr", "manifest"], key="ex_fmt")
    shard = b.number_input("Max parquet shard (MB, 0 = one file)", 0.0, 4096.0, 0.0, key="ex_shard")
    limit = int(c.number_input("Limit bundles (0 = all)", 0, 10_000_000, 0, key="ex_limit"))
    out = REPO_ROOT / st.text_input(
        "Export directory", f"artifacts/studio/export/{dataset.name}", key="ex_out"
    )
    if st.button("Export dataset", type="primary", key="ex_go"):
        code = _run(
            "Exporting...",
            lambda: run_export(dataset, fmt, out, limit or None, float(shard) or None),
        )
        if code == 0:
            st.success(f"Exported to {out}")
        elif code is not None:
            st.error("Nothing to export.")
    path = out / "dataset_manifest.json"
    if path.is_file():
        problems = check_manifest(out)
        if problems:
            st.warning("; ".join(problems[:5]))
        else:
            st.caption("Every exported file matches the manifest hashes.")
        st.download_button(
            "Download dataset_manifest.json", path.read_bytes(), "dataset_manifest.json"
        )
        with st.expander("dataset_manifest.json"):
            st.json(read_json(path))
    st.divider()
    st.subheader("Lineage")
    rows = forensics.bundles(dataset)
    if not rows:
        return
    nodes = forensics.lineage_nodes(dataset)
    st.write(f"{len(nodes)} nodes: scene, render, certificate, twin, eval")
    if st.button("Verify lineage against the files", key="ln_check"):
        if (dataset / lineage.FILENAME).is_file():
            found = _run("Checking...", lambda: lineage.check_dataset(dataset))
        else:
            found = lineage.verify(nodes)
        if found == []:
            st.success("Lineage is consistent.")
        elif found:
            st.error("; ".join(found[:5]))
    bid = st.selectbox("Bundle", [r["bundle_id"] for r in rows], key="ln_bundle")
    part = forensics.lineage_slice(nodes, bid)
    if part:
        st.graphviz_chart(forensics.lineage_dot(part))
        st.dataframe(pd.DataFrame(part), width="stretch", hide_index=True)


def main() -> None:
    os.environ["KNOTWORK_NETWORK"] = "deny"  # the studio never touches the network
    st.set_page_config(page_title="Visual Task Factory Studio", layout="wide")
    st.title("Visual Task Factory Studio")
    names = [
        "Factory Studio",
        "Forensics & Counterfactuals",
        "Human Calibration Lab",
        "Export & Lineage",
    ]
    views = (factory_tab, forensics_tab, calibration_tab, export_tab)
    for tab, view in zip(st.tabs(names), views, strict=True):
        with tab:
            view()


if __name__ == "__main__":
    main()
