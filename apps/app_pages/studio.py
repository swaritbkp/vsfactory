"""Task Studio: a control rail (30%) and a hero stage (70%) for one certified task at a time."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

from knotwork.posttrain import vcot
from knotwork.studio import factory, forensics, guide, present
from knotwork.studio.present import AUTO
from studio_ui import OUT, relative, resolve, run_with_status, verdict_badge

VIEWS = ("Beauty", "Witness Overlay", "ID Buffer", "CVD Simulation", "vCoT Steps")
CAPTIONS = {
    "Beauty": "What a model sees (rendered at 1024 px).",
    "Witness Overlay": "Kernel evidence: terminals (green), crossings (cyan), needle axis (magenta), "
    "OCR labels (amber); failed checks in red.",
    "ID Buffer": "Modal object ids: the visible owner of every pixel.",
    "CVD Simulation": "The same image under simulated colour-vision deficiency.",
    "vCoT Steps": "Grounded reasoning steps, numbered in order, boxed on the object masks they read.",
}
LEVELS = [AUTO, *(i / 10 for i in range(11))]
IMAGE_PX = 560
CERTIFY_ATTEMPTS = 3
HOLDOUT_EXAMPLES = {"tangle": "strands=9:10", "gauge": "radius_px=200:240", "panel": "gauges=4"}
SAMPLE = ("tangle", "T1_TRACE", 1)
PLUGIN_LABELS = {"rack_patch": "Rack patch-bay"}


def plugin_label(name: str) -> str:
    return PLUGIN_LABELS.get(
        name, name.replace("_", " ").title()
    )  # a seven-strand braid: the flagship domain on first load


# ------------------------------------------------------------------ state
def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("plugin", SAMPLE[0])
    ss.setdefault("view", VIEWS[0])
    ss.setdefault("seed", SAMPLE[2] + 1)  # the first Generate shows a new scene, not the sample
    for name, value in present.KNOB_DEFAULTS.items():
        ss.setdefault(f"knob_{name}", value)
    for name in ("stroke_px", "jitter_px", "noise", "dpi_scale"):
        ss.setdefault(f"knob_{name}", AUTO)
    if ss.pop("bump_seed", False):
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


@st.cache_resource(show_spinner=False)
def _first_sample(kernel: str) -> dict[str, Any]:
    """One real certified task for the first visit, rendered once per server and kernel."""
    plugin, family, seed = SAMPLE
    made = factory.generate_one(
        plugin,
        family,
        seed,
        OUT / "sample" / kernel / plugin,
        certify_attempts=CERTIFY_ATTEMPTS,
    )
    return _view_of(made, seed)


def _generate(plugin: str, family: str) -> None:
    """Render and certify a new task. The current stage stays on screen until it is ready."""
    seed = int(st.session_state.seed)
    target = OUT / "single" / f"{plugin}-{uuid.uuid4().hex[:8]}"

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
        st.session_state.bump_seed = True


# ------------------------------------------------------------------ batch build
@st.fragment(run_every="2s")
def _batch_live() -> None:
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
            f"Saved to `{relative(out)}`. "
            "It is preselected on the Export page.",
            icon=":material/check_circle:",
        )
        return
    st.error(f"knot generate exited with code {code}.", icon=":material/error:")
    log = Path(job["log"])
    if log.is_file():
        st.code(log.read_text(encoding="utf-8", errors="replace")[-2000:], language=None)


@st.fragment
def _batch_panel(plugin: str) -> None:
    if not factory.batch_ready(plugin):
        st.caption("This plugin has no profile section, so it supports single items only.")
        return
    n = int(st.number_input("Certified base items", 1, 1_000_000, 20, key="bt_n",
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
    folder = st.text_input(
        "Output folder",
        f"artifacts/studio/batch/{plugin}-{split}",
        key=f"bt_out_{plugin}_{split}",
        help=guide.tip("bt_out"),
    )
    spec = factory.difficulty_from(
        _dials(("reasoning_load", "visual_stress", "occlusion_level", "color_regime"))
    )
    out = resolve(folder)
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
        st.rerun(scope="fragment")
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


def _summary(item: dict[str, Any]) -> None:
    cert, scores = item["certificate"], item["scores"]
    certified = cert["verdict"] == "CERTIFIED"
    with st.container(border=True):
        status, answer, task, load, stress = st.columns(
            [1.25, 1.1, 1.5, 1, 1], vertical_alignment="center"
        )
        with status:
            verdict_badge(cert["verdict"])
            st.caption(f"seed {item['seed']} · attempt {item['attempts']}")
        answer.metric("Certified answer", item["answer"] if certified else "none",
                      help=guide.tip("certified_answer"))  # fmt: skip
        task.metric(
            "Plugin / family", plugin_label(item["plugin"]), item["family"],
            delta_color="off", delta_arrow="off", help=guide.tip("plugin_family"),
        )  # fmt: skip
        load.metric("Reasoning", f"{scores['reasoning'][0]}/100", help=scores["reasoning"][1])
        stress.metric("Visual stress", f"{scores['visual'][0]}/100", help=scores["visual"][1])
    if not certified:
        st.error(
            "The kernel rejected this render: " + ", ".join(cert["codes"]),
            icon=":material/gpp_maybe:",
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


@st.fragment
def _stage() -> None:
    item = st.session_state.item
    _summary(item)
    view = st.segmented_control(
        "View",
        VIEWS,
        key="view",
        required=True,
        label_visibility="collapsed",
        help=guide.tip("view"),
    )
    deficiency = "deuteranopia"
    if view == "CVD Simulation":
        chosen = st.segmented_control(
            "Deficiency", ["Deuteranopia", "Protanopia"], default="Deuteranopia",
            required=True, key="deficiency", label_visibility="collapsed",
            help=guide.tip("deficiency"),
        )  # fmt: skip
        deficiency = str(chosen).lower()
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
                st.caption("Colour carries no meaning in this scene, so the gate does not apply.")
        if view == "Witness Overlay":
            _witness_pill(item["certificate"])
        if view == "vCoT Steps":
            trace = _trace(item["dir"], json.dumps(item["certificate"]))
            if trace is None:
                st.caption("Not certified, so there is no grounded trace to draw.")
            else:
                st.badge(f"{len(trace['steps'])} grounded steps · answer {trace['final_answer']}",
                         color="green", icon=":material/route:")  # fmt: skip
    with st.container(border=True):
        st.caption("Question")
        question = item["question"]
        st.markdown(question["text"].replace("\n", "  \n"))
        if question.get("options"):
            st.caption("Options: " + " · ".join(question["options"]))
    _vcot_panel(item)
    with st.expander("certificate.json and task.json", icon=":material/data_object:"):
        certificate, task = st.tabs(["certificate.json", "task.json"])
        certificate.json(item["certificate"], expanded=1)
        if item["task"]:
            task.json(item["task"], expanded=1)
            task.caption(f"Written to `{item['dir']}`; `knot verify` can re-check it.")
        else:
            task.caption("Only certified tasks are written as task.json.")


# ------------------------------------------------------------------ page
_init_state()
rail, stage = st.columns([3, 7], gap="large")
with rail:
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
    st.pills(
        "Scenario preset", list(present.PRESETS), key="preset", on_change=_apply_preset,
        help=guide.tip("preset"),
    )  # fmt: skip
    clicked = st.button(
        "Generate & Certify Task", type="primary", width="stretch", icon=":material/play_arrow:",
        key="generate", help=guide.tip("generate"),
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
    with st.expander("Batch & OOD Build", icon=":material/library_add:"):
        _batch_panel(plugin)

with stage:
    progress = st.container()  # fixed slot above the stage: the old task stays visible meanwhile
    if "item" not in st.session_state:
        with progress, st.spinner("Rendering and certifying a first sample..."):
            st.session_state.item = _first_sample(str(present.repo_status()["kernel"]))
    if clicked:
        with progress:
            _generate(plugin, family)
    _stage()
