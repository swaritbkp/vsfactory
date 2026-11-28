"""Task Studio: a control rail (30%) and a hero stage (70%) for one certified task at a time."""

from __future__ import annotations

import io
import json
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

from knotwork.posttrain import vcot
from knotwork.studio import demo_guard, factory, forensics, guide, present, working_set
from knotwork.studio.present import AUTO
from studio_ui import relative, resolve, run_with_status, studio_out, verdict_badge
from ui_chrome import qa_review, switch_factory_page, sync_demo_session
from ui_chrome import working_set as active_set

VIEWS = ("Beauty", "Witness Overlay", "ID Buffer", "CVD Simulation", "vCoT Steps")
VIEW_SHORT = {
    "Beauty": "Beauty",
    "Witness Overlay": "Witness",
    "ID Buffer": "ID Buffer",
    "CVD Simulation": "CVD",
    "vCoT Steps": "vCoT",
}
CAPTIONS = {
    "Beauty": "What a model sees (rendered at 1024 px).",
    "Witness Overlay": "Kernel evidence: terminals (green), crossings (cyan), needle axis (magenta), "
    "OCR labels (amber); failed checks in red.",
    "ID Buffer": "Modal object ids: the visible owner of every pixel.",
    "CVD Simulation": "The same image under simulated colour-vision deficiency.",
    "vCoT Steps": "Grounded reasoning steps, numbered in order, boxed on the object masks they read.",
}
LEVELS = [AUTO, *(i / 10 for i in range(11))]
IMAGE_PX = 400
CERTIFY_ATTEMPTS = 3
HOLDOUT_EXAMPLES = {
    "tangle": "strands=9:10",
    "rack_patch": "strands=12:14",
    "gauge": "radius_px=200:240",
    "panel": "gauges=4",
}
SAMPLE = ("tangle", "T1_TRACE", 1)
PLUGIN_LABELS = {"rack_patch": "Rack patch-bay"}


def plugin_label(name: str) -> str:
    return PLUGIN_LABELS.get(
        name, name.replace("_", " ").title()
    )  # a seven-strand braid: the flagship domain on first load


# ------------------------------------------------------------------ state
def _init_state() -> None:
    ss = st.session_state
    if "item" not in ss:
        starter = _load_starter_item()
        if starter is not None:
            ss.item = starter
    item = ss.get("item")
    default_plugin = str(item["plugin"]) if item else SAMPLE[0]
    default_family = str(item["family"]) if item else SAMPLE[1]
    default_seed = int(item["seed"]) if item else SAMPLE[2] + 1
    ss.setdefault("plugin", default_plugin)
    ss.setdefault(f"family_{default_plugin}", default_family)
    ss.setdefault("view", VIEWS[0])
    ss.setdefault("seed", default_seed)  # synced with starter item on cold load
    for name, value in present.KNOB_DEFAULTS.items():
        ss.setdefault(f"knob_{name}", value)
    for name in ("stroke_px", "jitter_px", "noise", "dpi_scale"):
        ss.setdefault(f"knob_{name}", AUTO)
    bumped_from = ss.pop("bump_seed", None)
    if bumped_from is not None and ss.get("seed") == bumped_from:
        ss.seed += 1


def _apply_preset() -> None:
    name = st.session_state.get("preset")
    if not name:
        return
    for knob, value in present.PRESETS[name].items():
        st.session_state[f"knob_{knob}"] = value
    for knob in ("stroke_px", "jitter_px", "noise", "dpi_scale"):
        st.session_state[f"knob_{knob}"] = AUTO
    st.session_state.view = "CVD Simulation" if name == "CVD" else "Beauty"


def _dials(names: tuple[str, ...]) -> dict[str, Any]:
    values = {n: st.session_state.get(f"knob_{n}", AUTO) for n in names}
    return {k: v for k, v in values.items() if v != AUTO}


# ------------------------------------------------------------------ generation
def _view_of(made: factory.Generated, seed: int) -> dict[str, Any]:
    c = made.candidate
    return {
        "dir": str(c.dir),
        "plugin": c.scene.plugin,
        "family": c.question.family.value,
        "seed": seed,
        "attempts": made.attempts,
        "certificate": c.certificate.model_dump(mode="json"),
        "question": c.question.model_dump(mode="json"),
        "answer": c.answer.canonical,
        "task": made.task,
        "scores": present.difficulty_scores(c.scene.plugin, c.scene.params, c.scene.payload),
    }


def _load_starter_item() -> dict[str, Any] | None:
    """Load the first certified item from the active Working Set for instant cold start."""
    ws = active_set()
    rows = ws.rows()
    ordered = sorted(rows, key=lambda r: 0 if r.get("plugin") == SAMPLE[0] else 1)
    for row in ordered:
        bundle = ws.root / str(row["dir"])
        if not ((bundle / "beauty.png").is_file() and (bundle / "manifest.json").is_file()):
            continue
        try:
            task = json.loads((bundle / "task.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(task, dict) or not working_set.certified(task):
            continue
        question = task["question"]
        seed_val = row.get("seed")
        seed = int(seed_val) if seed_val is not None else SAMPLE[2]
        payload = {
            **dict(question.get("difficulty") or {}),
            "difficulty_spec": question.get("difficulty_spec"),
        }
        return {
            "dir": str(bundle),
            "plugin": str(task["plugin"]),
            "family": str(task["family"]),
            "seed": seed,
            "attempts": 1,
            "certificate": task["certificate"],
            "question": question,
            "answer": str(task["answer"]["canonical"]),
            "task": task,
            "scores": present.difficulty_scores(str(task["plugin"]), {}, payload),
        }
    return None


@st.cache_resource(show_spinner=False)
def _first_sample(kernel: str) -> dict[str, Any]:
    """One real certified task for the first visit, rendered once per server and kernel.

    Each server process renders into its own directory: a shared fixed path let concurrent
    servers (or pytest-xdist workers) rewrite PNGs that another process was certifying or
    reading, which surfaced as "image file is truncated"."""
    plugin, family, seed = SAMPLE
    made = factory.generate_one(
        plugin,
        family,
        seed,
        studio_out() / "sample" / kernel / f"{plugin}-{uuid.uuid4().hex[:8]}",
        certify_attempts=CERTIFY_ATTEMPTS,
    )
    return _view_of(made, seed)


def _generate(plugin: str, family: str) -> None:
    """Render and certify a new task. The current stage stays on screen until it is ready."""
    seed = int(st.session_state.seed)
    target = studio_out() / "single" / f"{plugin}-{uuid.uuid4().hex[:8]}"

    def action() -> dict[str, Any]:
        dials = _dials(
            tuple(k.removeprefix("knob_") for k in st.session_state if k.startswith("knob_"))
        )
        made = factory.generate_one(
            plugin, family, seed, target, factory.difficulty_from(dials),
            certify_attempts=CERTIFY_ATTEMPTS,
        )  # fmt: skip
        return _view_of(made, seed)

    item = run_with_status("Render and certify", action)
    if item is not None:
        st.session_state.item = item
        st.session_state.bump_seed = seed


# ------------------------------------------------------------------ batch build
@st.fragment(run_every="2s")
def _batch_live() -> None:
    sync_demo_session()
    job = st.session_state.get("batch")
    if job is None:
        return
    done = factory.batch_progress(Path(job["out"]))
    st.progress(min(done / job["n"], 1.0), text=f"{done} of {job['n']} base items")
    if job["process"].poll() is not None:
        st.rerun()
    if st.button("Stop build", icon=":material/stop:", key="bt_stop", width="stretch"):
        job["process"].terminate()


def _batch_result(job: dict[str, Any]) -> None:
    code = job["process"].poll()
    done = factory.batch_progress(Path(job["out"]))
    if code == 0:
        out = Path(job["out"])
        st.session_state.last_batch_dir = relative(out)
        total = len(forensics.bundles(out))
        rate = present.run_kpis(out)["yield_rate"]
        yield_text = "n/a" if rate is None else f"{rate:.1%}"
        st.success(
            f"**Batch built.** {done} certified base items, {total} bundles in total "
            f"(twins included), yield {yield_text}.  \n"
            f"Saved to `{demo_guard.display_path(out)}`. "
            "It is preselected on the Export page.",
            icon=":material/check_circle:",
        )
        return
    st.error(f"knot generate exited with code {code}.", icon=":material/error:")
    log = Path(job["log"])
    if log.is_file():
        st.code(log.read_text(encoding="utf-8", errors="replace")[-2000:], language=None)


def _switch_to_single_task() -> None:
    st.session_state.studio_mode = "Single Task"


def _batch_panel(plugin: str) -> None:
    if not factory.batch_ready(plugin):
        st.caption("This plugin has no profile section, so it supports single items only.")
        st.button(
            "Switch to Single Task",
            icon=":material/arrow_back:",
            type="primary",
            width="stretch",
            key="bt_to_single",
            on_click=_switch_to_single_task,
        )
        return
    most = demo_guard.cap(1_000_000, demo_guard.DEMO_LIMITS.batch_items)
    n = int(st.number_input("Certified base items", 1, most, min(20, most), key="bt_n",
                            help=guide.tip("bt_n")))  # fmt: skip
    split = st.segmented_control(
        "Split",
        ["dev", "test", "sealed", "ood"],
        default="dev",
        required=True,
        key="bt_split",
        help=guide.tip("bt_split"),
    )
    held = st.multiselect("Held-out families", factory.families_of(plugin), key=f"bt_hold_{plugin}",
                          help=guide.tip("bt_held_families"))  # fmt: skip
    params = st.text_input(
        "Held-out parameters", placeholder=HOLDOUT_EXAMPLES.get(plugin, "name=lo:hi"),
        key="bt_params", help=guide.tip("bt_held_params"),
    )  # fmt: skip
    twins = st.toggle("Causal and nuisance twins", True, key="bt_twins", help=guide.tip("bt_twins"))
    with st.popover("Run options", icon=":material/tune:", width="stretch"):
        seed = int(st.number_input("Batch seed", 0, 2**31 - 1, 0, key="bt_seed",
                                   help=guide.tip("bt_seed")))  # fmt: skip
        workers = int(st.number_input("Workers (0 = auto)", 0, 64, 0, key="bt_workers",
                                      help=guide.tip("bt_workers")))  # fmt: skip
        shard = int(st.number_input("Slots per shard (0 = flat)", 0, 100_000, 0, key="bt_shard",
                                    help=guide.tip("bt_shard")))  # fmt: skip
    if demo_guard.is_public_demo():
        folder = f"studio/batch/{plugin}-{split}"
        refused = None
        out = resolve(folder)
        st.caption(f"Output: `{demo_guard.display_path(out)}` (auto-managed)")
    else:
        folder = st.text_input(
            "Output folder",
            f"artifacts/studio/batch/{plugin}-{split}",
            key=f"bt_out_{plugin}_{split}",
            help=guide.tip("bt_out"),
        )
        refused = demo_guard.path_problem(folder)
        if refused:
            st.error(refused, icon=":material/folder_off:")
            return
        out = resolve(folder)
    spec = factory.difficulty_from(
        _dials(("reasoning_load", "visual_stress", "occlusion_level", "color_regime"))
    )
    args = factory.batch_command(
        plugin,
        n,
        Path(relative(out)),
        seed,
        str(split),
        twins,
        workers,
        shard,
        held,
        params.split(),
        spec,
    )
    with st.expander("Show CLI command", icon=":material/terminal:"):
        st.code(factory.command_line(args), language="powershell", wrap_lines=True)
        st.caption(
            "Batches take the load, stress, occlusion and colour dials from Advanced Controls."
        )
    job = st.session_state.get("batch")
    running = job is not None and job["process"].poll() is None
    missing = split == "ood" and not (held or params.strip())
    if missing:
        st.warning("An ood split needs a held-out family or parameter.", icon=":material/rule:")
    if st.button("Launch batch build", icon=":material/rocket_launch:", width="stretch",
                 disabled=running or missing, key="bt_go", help=guide.tip("bt_launch")):  # fmt: skip
        log = out.parent / f"{out.name}.log"
        st.session_state.batch = {
            "process": factory.start_batch(args, log), "out": str(out), "n": n, "log": str(log),
        }  # fmt: skip
        st.rerun()
    if job is None:
        st.caption("No batch built yet. Set a count and press Launch batch build.")
    elif running:
        _batch_live()
    else:
        _batch_result(job)


# ------------------------------------------------------------------ stage
@st.cache_data(show_spinner=False, max_entries=48)
def _image(directory: str, view: str, deficiency: str, certificate: str) -> Image.Image:
    bundle = Path(directory)
    if view == "Witness Overlay":
        return factory.witness_overlay(bundle, json.loads(certificate))
    if view == "ID Buffer":
        return factory.id_buffer_image(bundle)
    if view == "CVD Simulation":
        return factory.cvd_preview(bundle, deficiency)
    if view == "vCoT Steps":
        trace = _trace(directory, certificate)
        if trace is not None:
            return vcot.overlay(factory.beauty(bundle), vcot.VCoTTrace.model_validate(trace))
    return factory.beauty(bundle)


@st.cache_data(show_spinner=False, max_entries=48)
def _trace(directory: str, certificate: str) -> dict[str, Any] | None:
    """The grounded vCoT trace of a certified item (None when it is not certified)."""
    if json.loads(certificate).get("verdict") != "CERTIFIED":
        return None
    trace = vcot.try_extract(Path(directory))
    return None if trace is None else trace.model_dump()


def _vcot_panel(item: dict[str, Any]) -> None:
    """Step table and the exact SFT target string of the grounded trace."""
    trace = _trace(item["dir"], json.dumps(item["certificate"]))
    with st.expander("Grounded Visual CoT (SFT trace)", icon=":material/route:",
                     expanded=st.session_state.get("view") == "vCoT Steps"):  # fmt: skip
        if trace is None:
            st.caption("Only certified items get a grounded trace: the kernel must agree first.")
            return
        st.caption(guide.tip("vcot_steps"))
        rows = [{"Step": s["step_idx"], "Phase": s["phase"], "Evidence": s["evidence_label"],
                 "Box (0-1000)": str(s["bbox_norm_1000"]), "Reasoning": s["text"]}
                for s in trace["steps"]]  # fmt: skip
        st.dataframe(rows, hide_index=True, width="stretch", column_config={
            "Step": st.column_config.NumberColumn(width="small"),
            "Box (0-1000)": st.column_config.TextColumn(help=guide.tip("vcot_box")),
        })  # fmt: skip
        st.code(trace["formatted_cot"], language=None, wrap_lines=True)


def _truth_card(item: dict[str, Any]) -> None:
    """Question prominently first, immediately followed by certified truth, verdict and metrics."""
    cert, scores = item["certificate"], item["scores"]
    certified = cert["verdict"] == "CERTIFIED"
    question = item["question"]
    q_text = str(question["text"]).split("\nEnd your reply")[0]
    with st.container(border=True, key="studio_truth_card"):
        st.caption("Question")
        st.markdown(q_text.replace("\n", "  \n"))
        if question.get("options"):
            st.caption("Options: " + " · ".join(question["options"]))
        if certified:
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.markdown(f":material/verified: **Certified truth:** `{item['answer']}`",
                            width="content")  # fmt: skip
                st.badge(
                    f"Certified · {len(cert['codes'])} E_* errors",
                    icon=":material/verified_user:",
                    color="green",
                )
        else:
            verdict_badge(cert["verdict"])
            st.error(
                "The kernel rejected this render: " + ", ".join(cert["codes"]),
                icon=":material/gpp_maybe:",
            )
        c1, c2 = st.columns([1.15, 1.35])
        c1.metric("Certified answer", item["answer"] if certified else "none",
                  help=guide.tip("certified_answer"))  # fmt: skip
        c2.metric(
            "Plugin / family", plugin_label(item["plugin"]), item["family"],
            delta_color="off", delta_arrow="off", help=guide.tip("plugin_family"),
        )  # fmt: skip
        st.caption(
            f"Reasoning {scores['reasoning'][0]}/100 · "
            f"Visual {scores['visual'][0]}/100 · seed {item['seed']}"
        )


def _witness_pill(cert: dict[str, Any]) -> None:
    """One line that says what the overlay proves, in the CVD gate's pill style."""
    checks = cert["checks"]
    passed = sum(bool(c["passed"]) for c in checks)
    oracle = [c for c in checks if c["name"].startswith("oracle_")]
    oracle_ok = bool(oracle) and all(c["passed"] for c in oracle)
    codes = len(cert["codes"])
    parts = [
        "D1 = D3 verified" if cert["agree"] else "D1 and D3 disagree",
        f"{codes} E_* code{'s' if codes != 1 else ''}",
        f"{passed}/{len(checks)} checks passed",
    ]
    if oracle:
        parts.append("beauty-only oracle pass" if oracle_ok else "beauty-only oracle failed")
    clean = cert["agree"] and not codes and (oracle_ok or not oracle)
    st.badge("Kernel witness: " + " · ".join(parts), color="green" if clean else "red",
             icon=":material/verified_user:" if clean else ":material/gpp_bad:")  # fmt: skip


# ------------------------------------------------------------------ workflow action bar
@st.cache_data(show_spinner=False, max_entries=16)
def _bundle_zip(directory: str, certificate: str) -> bytes:
    """beauty, witness overlay, manifest, task and vCoT SFT line of one item, zipped in memory."""
    buf = io.BytesIO()
    _image(directory, "Witness Overlay", "deuteranopia", certificate).save(buf, format="PNG")
    bundle = Path(directory)
    return working_set.bundle_zip(bundle, buf.getvalue(), working_set.vcot_line(bundle))


def _add(item: dict[str, Any]) -> working_set.AddResult:
    return active_set().add(Path(item["dir"]), "studio", item["seed"], item["attempts"])


def _workflow(item: dict[str, Any]) -> None:
    """Add to Working Set, Quick test in Arena, Download bundle and item-scoped auto-save QA."""
    if item["certificate"]["verdict"] != "CERTIFIED" or not item["task"]:
        return
    bid = str(item["task"]["bundle_id"])
    ws = active_set()
    in_set = bid in ws.ids()
    with st.container(border=True, key="studio_workflow"):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            add = st.button(
                "In the Working Set" if in_set else "Add to Working Set",
                icon=":material/library_add_check:" if in_set else ":material/library_add:",
                disabled=in_set, key="st_add", help=guide.tip("st_add_ws"),
            )  # fmt: skip
            quick = st.button("Quick test in Arena", icon=":material/arrow_forward:",
                              icon_position="right", key="st_quick",
                              help=guide.tip("st_quick_arena"))  # fmt: skip
            certificate = json.dumps(item["certificate"])
            st.download_button(
                "Download bundle (.zip)", lambda: _bundle_zip(item["dir"], certificate),
                file_name=f"{bid}.zip", mime="application/zip", icon=":material/download:",
                on_click="ignore", key="st_zip", help=guide.tip("st_download"),
            )  # fmt: skip
        item_key = f"{bid}_{item['seed']}"
        qa_review(ws, bid, item_key, Path(item["dir"]), item["seed"], item["attempts"])
    if add:
        result = _add(item)
        unit = "item" if result.count == 1 else "items"
        st.toast(f"Added to the Working Set: {result.count} {unit}" if result.added
                 else result.reason, icon=":material/library_add_check:")  # fmt: skip
        st.rerun()
    if quick:
        _add(item)
        st.session_state.ar_quick_ids = [bid]
        st.session_state.ar_quick_pending = True
        st.session_state._arena_open_new_session = True
        switch_factory_page("model_arena")


def _stage() -> None:
    item = st.session_state.item
    left_stage, right_stage = st.columns([1.0, 1.12], gap="medium")
    with left_stage:
        view = st.segmented_control(
            "View",
            VIEWS,
            key="view",
            format_func=lambda v: VIEW_SHORT.get(v, v),
            label_visibility="collapsed",
            width="stretch",
            help=guide.tip("view"),
        )
        if view is None:
            view = VIEWS[0]
        deficiency = "deuteranopia"
        if view == "CVD Simulation":
            chosen = st.segmented_control(
                "Deficiency", ["Deuteranopia", "Protanopia"], default="Deuteranopia",
                key="deficiency", label_visibility="collapsed", help=guide.tip("deficiency"),
            )  # fmt: skip
            deficiency = str(chosen or "Deuteranopia").lower()
        with st.container(horizontal_alignment="center", gap="small"):
            picture = _image(item["dir"], str(view), deficiency, json.dumps(item["certificate"]))
            st.image(picture, width=IMAGE_PX, alt=f"{view} of the generated task")
            st.caption(CAPTIONS[str(view)], width="content")
            if view == "CVD Simulation":
                gates = [c for c in item["certificate"]["checks"] if c["name"].startswith("cvd_")]
                if gates:
                    worst = min(gates, key=lambda c: c["metric"])
                    st.badge(
                        f"Kernel colour gate: separation {worst['metric']:.1f} "
                        f"(floor {worst['threshold']:g})",
                        color="green" if all(c["passed"] for c in gates) else "red",
                        icon=":material/palette:",
                    )
                else:
                    st.caption(
                        "Colour carries no meaning in this scene, so the gate does not apply."
                    )
            if view == "Witness Overlay":
                _witness_pill(item["certificate"])
            if view == "vCoT Steps":
                trace = _trace(item["dir"], json.dumps(item["certificate"]))
                if trace is None:
                    st.caption("Not certified, so there is no grounded trace to draw.")
                else:
                    st.badge(
                        f"{len(trace['steps'])} grounded steps · answer {trace['final_answer']}",
                        color="green", icon=":material/route:",
                    )  # fmt: skip
    with right_stage:
        _truth_card(item)
        _workflow(item)
    _vcot_panel(item)
    with st.expander("certificate.json and task.json", icon=":material/data_object:"):
        certificate, task = st.tabs(["certificate.json", "task.json"])
        certificate.json(item["certificate"], expanded=1)
        if item["task"]:
            task.json(item["task"], expanded=1)
            task.caption(
                f"Written to `{demo_guard.display_path(item['dir'])}`; `knot verify` can re-check it."
            )
        else:
            task.caption("Only certified tasks are written as task.json.")


# ------------------------------------------------------------------ page
_init_state()
rail, stage = st.columns([2.7, 7.3], gap="medium")
with rail:
    workspace_mode = st.segmented_control(
        "Workspace mode",
        ["Single Task", "Batch Factory"],
        default="Single Task",
        key="studio_mode",
        label_visibility="collapsed",
        width="stretch",
    )
    if workspace_mode is None:
        workspace_mode = "Single Task"
    names = factory.plugin_names()
    plugin = str(
        st.segmented_control(
            "Plugin",
            names,
            format_func=plugin_label,
            key="plugin",
            required=True,
            width="stretch",
            help=guide.tip("plugin"),
        )
    )
    clicked = False
    family = factory.families_of(plugin)[0]
    if workspace_mode == "Single Task":
        st.pills(
            "Scenario preset", list(present.PRESETS), key="preset", on_change=_apply_preset,
            help=guide.tip("preset"),
        )  # fmt: skip
        clicked = st.button(
            "Generate & Certify Task", type="primary", width="stretch",
            icon=":material/play_arrow:", key="generate", help=guide.tip("generate"),
        )  # fmt: skip
        with st.expander("Advanced Controls", icon=":material/tune:"):
            family = str(
                st.selectbox(
                    "Question family",
                    factory.families_of(plugin),
                    key=f"family_{plugin}",
                    help=guide.tip("family"),
                )  # fmt: skip
            )
            st.number_input("Seed", 0, 2**31 - 1, key="seed",
                            help=guide.tip("seed") + " Advances after each generation.")  # fmt: skip
            for knob, label, hint in (
                ("reasoning_load", "Reasoning load", "reasoning_load"),
                ("visual_stress", "Visual stress", "visual_stress"),
                ("occlusion_level", "Occlusion", "occlusion"),
                ("realism", "Surface realism", "realism"),
            ):
                st.select_slider(label, LEVELS, key=f"knob_{knob}", help=guide.tip(hint))
            st.segmented_control(
                "Color regime", [AUTO, "unique", "partial", "mono"], key="knob_color_regime",
                required=True, width="stretch", help=guide.tip("color_regime"),
            )  # fmt: skip
            if st.toggle("Show fine render knobs", value=False, key="fine_knobs",
                         help=guide.tip("fine_knobs")):  # fmt: skip
                st.select_slider("Stroke width (px)", [AUTO, *range(5, 15)], key="knob_stroke_px",
                                 help=guide.tip("stroke_px"))  # fmt: skip
                st.select_slider("Jitter (px)", [AUTO, *range(0, 25, 4)], key="knob_jitter_px",
                                 help=guide.tip("jitter_px"))  # fmt: skip
                st.select_slider("Sensor noise", LEVELS, key="knob_noise", help=guide.tip("noise"))
                st.select_slider(
                    "Capture resolution",
                    [AUTO, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    key="knob_dpi_scale",
                    help=guide.tip("dpi_scale"),
                )
    else:
        _batch_panel(plugin)

with stage:
    progress = st.container()  # fixed slot above the stage: the old task stays visible meanwhile
    if "item" not in st.session_state:
        with progress, st.spinner("Rendering and certifying a first sample..."):
            st.session_state.item = _first_sample(str(present.repo_status()["kernel"]))
    if clicked:
        with progress:
            _generate(plugin, family)
    current = st.session_state.get("item")
    if workspace_mode == "Single Task" and current is not None and current.get("plugin") != plugin:
        selected_label = plugin_label(plugin)
        shown_label = plugin_label(str(current["plugin"]))
        st.info(
            f"Selected **{selected_label}** — currently previewing **{shown_label}**. "
            f"Press **Generate & Certify Task** to render {selected_label}.",
            icon=":material/info:",
        )
    _stage()
