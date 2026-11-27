"""Model Arena: multi-model campaigns over certified items, with a sessions rail, separate dataset
and results folders, a resumable background runner, live progress and research-grade reports.

The page only lays out widgets; the logic lives in `knotwork.arena`. It works fully offline: the
model list falls back to a curated catalog, and Offline Mock mode needs no key. A live OpenRouter
call happens only after a click on Launch, Resume, Retry, Refresh live catalog or Synthesize.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pydantic import ValidationError

from knotwork.arena import catalog, keys, planning, report, runner, storage
from knotwork.arena import forensics as arena_forensics
from knotwork.arena.schemas import (
    DOMAIN_LABELS,
    MAX_REPEATS,
    PRESETS,
    BlockSpec,
    ItemRecord,
    SessionConfig,
    TrialRecord,
)
from knotwork.pipeline.io import read_json
from knotwork.studio import factory, forensics, guide
from posttrain_ui import posttrain_tab
from studio_ui import relative, run_with_status

WS = storage.Workspace()
MODES = {"Offline Mock (No Key)": "mock", "Live (OpenRouter)": "live"}
DATA_MODES = {"Generate new dataset": "generate", "Reuse existing certified dataset": "reuse"}
FILTERS = ("All", "Running", "Paused", "Done")
STATE_PILL = {
    "running": ("RUNNING", "blue", ":material/autorenew:"),
    "paused": ("PAUSED", "orange", ":material/pause_circle:"),
    "stopped": ("STOPPED", "gray", ":material/stop_circle:"),
    "done": ("DONE", "green", ":material/check_circle:"),
    "budget_stop": ("BUDGET STOP", "orange", ":material/savings:"),
    "failed": ("FAILED", "red", ":material/error:"),
    "idle": ("NOT STARTED", "gray", ":material/hourglass_empty:"),
}
FILTER_STATES = {"Running": {"running"}, "Paused": {"paused", "stopped", "budget_stop"},
                 "Done": {"done"}}  # fmt: skip
IMAGE_PX = 500
SYNTH_MODELS = ("google/gemini-2.5-flash", "openai/gpt-5.4", "anthropic/claude-sonnet-4.5")
SECRET_KEY = "ar_api_key"  # session-only widget state; never written anywhere


# ------------------------------------------------------------------ cached reads
@st.cache_data(ttl=600, show_spinner=False)
def _catalog(live: bool) -> catalog.Catalog:
    """The live OpenRouter list (only after a click), else the curated offline catalog."""
    return catalog.fetch(live=live)


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.is_file() else 0.0


@st.cache_data(ttl=5, show_spinner=False)
def _card_facts(results_dir: str, _stamp: tuple[float, float]) -> dict[str, Any]:
    """Status, progress, best accuracy and spend of one session (keyed by file mtimes)."""
    path = Path(results_dir)
    status = storage.read_status(path)
    trials = storage.load_trials(path)
    tally: dict[str, list[int]] = {}
    for t in trials.values():
        if t.status == "ok":
            cell = tally.setdefault(t.model, [0, 0])
            cell[0] += int(bool(t.correct))
            cell[1] += 1
    best = max((k / n for k, n in tally.values() if n), default=None)
    models = sorted({t.model for t in trials.values()})
    state = status.state
    if state == "running" and not runner.pid_alive(status.pid):
        state = "stopped"  # the process is gone: the run can be resumed
    return {"state": state, "done": status.completed_trials, "total": status.total_trials,
            "best": best, "spend": sum(t.cost_usd for t in trials.values()),
            "updated": status.updated_utc, "models": models}  # fmt: skip


def _facts(entry_dir: str) -> dict[str, Any]:
    path = Path(entry_dir)
    stamp = (_mtime(path / storage.STATUS), _mtime(path / storage.RESULTS))
    return _card_facts(entry_dir, stamp)


def _api_key() -> tuple[str | None, str]:
    """The key to hand a run: the session-only override when switched on, else the detected one."""
    if st.session_state.get("ar_override") and st.session_state.get(SECRET_KEY):
        return str(st.session_state[SECRET_KEY]).strip(), "session override"
    return keys.detect()


def _pill(state: str) -> None:
    text, color, icon = STATE_PILL.get(state, STATE_PILL["idle"])
    st.badge(text, color=color, icon=icon)  # type: ignore[arg-type]


# ------------------------------------------------------------------ new-session form state
def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("ar_selected", None)
    ss.setdefault("ar_name", "Rack monochrome audit")
    ss.setdefault("ar_blocks", [b.model_dump(mode="json") for b in planning.default_blocks()])
    ss.setdefault("ar_custom_models", [])
    ss.setdefault("ar_live_catalog", False)
    ss.setdefault("ar_procs", {})
    slug = storage.slugify(str(ss.ar_name))
    data_dir, results_dir = storage.default_dirs(slug)
    ss.setdefault("ar_data_dir", data_dir)
    ss.setdefault("ar_results_dir", results_dir)
    ss.setdefault("ar_models", list(catalog.DEFAULT_MODELS))


def _on_name() -> None:
    """Folders that still hold the previous default follow the new session name."""
    ss = st.session_state
    old = ss.get("ar_last_slug") or ""
    new = storage.slugify(str(ss.ar_name))
    old_dirs = storage.default_dirs(old) if old else ("", "")
    new_dirs = storage.default_dirs(new)
    for key, before, after in zip(
        ("ar_data_dir", "ar_results_dir"), old_dirs, new_dirs, strict=True
    ):
        if not ss.get(key) or ss.get(key) == before:
            ss[key] = after
    ss.ar_last_slug = new


def _load_form(cfg: SessionConfig) -> None:
    """Pre-fill the New Session form from a config (Clone config)."""
    ss = st.session_state
    ss.ar_selected = None
    ss.ar_name = cfg.session_name
    ss.ar_last_slug = cfg.slug
    ss.ar_client, ss.ar_eval, ss.ar_notes = cfg.client, cfg.evaluator, cfg.notes
    ss.ar_data_mode = "Generate new dataset"
    ss.ar_data_dir, ss.ar_results_dir = cfg.data_dir, cfg.results_dir
    ss.ar_mode = next(k for k, v in MODES.items() if v == cfg.mode)
    ss.ar_cap, ss.ar_conc, ss.ar_rpm = cfg.spend_cap_usd, cfg.max_concurrency, cfg.rpm
    ss.ar_repeats = cfg.repeats
    ss.ar_temperature = cfg.temperature
    known = set(_catalog(bool(ss.get("ar_live_catalog"))).ids())
    extra = [m for b in cfg.blocks for m in cfg.models_for(b) if m not in known]
    ss.ar_custom_models = list(dict.fromkeys([*ss.ar_custom_models, *extra]))
    ss.ar_models = list(cfg.default_models)
    ss.ar_blocks = [b.model_dump(mode="json") for b in cfg.blocks]
    for b in cfg.blocks:
        for suffix in ("domain", "preset", "n", "seed", "default", "models", "pick"):
            ss.pop(f"blk_{b.block_id}_{suffix}", None)


def _new_session() -> None:
    st.session_state.ar_selected = None


def _select(slug: str) -> None:
    st.session_state.ar_selected = slug


# ------------------------------------------------------------------ sidebar
def _sidebar() -> None:
    ss = st.session_state
    with st.sidebar:
        st.button("New session", type="primary", icon=":material/add:", width="stretch",
                  on_click=_new_session, key="ar_new", help=guide.tip("ar_new"))  # fmt: skip
        query = st.text_input("Search sessions", key="ar_q", placeholder="Name, client or model",
                              label_visibility="collapsed", icon=":material/search:",
                              help=guide.tip("ar_search"))  # fmt: skip
        chosen = st.selectbox("Status", FILTERS, key="ar_filter", label_visibility="collapsed",
                              help=guide.tip("ar_filter"))  # fmt: skip
        entries = WS.entries()
        shown = []
        for entry in entries:
            facts = _facts(entry.results_dir)
            text = " ".join([entry.name, entry.client, *entry.tags, *facts["models"]]).lower()
            if query and query.lower() not in text:
                continue
            if chosen and chosen != "All" and facts["state"] not in FILTER_STATES[str(chosen)]:
                continue
            shown.append((entry, facts))
        if not entries:
            st.caption("No saved sessions yet. Fill in the setup on the right and press Launch "
                       "campaign.")  # fmt: skip
        elif not shown:
            st.caption("No session matches this search.")
        for entry, facts in shown:
            selected = ss.ar_selected == entry.slug
            with st.container(border=True, key=f"ar_card_{entry.slug}"):
                st.button(entry.name, key=f"ar_open_{entry.slug}", width="stretch",
                          type="primary" if selected else "tertiary", on_click=_select,
                          args=(entry.slug,))  # fmt: skip
                with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                    _pill(facts["state"])
                    if entry.client:
                        st.badge(entry.client, color="gray", icon=":material/apartment:")
                best = "—" if facts["best"] is None else f"{facts['best']:.0%}"
                st.caption(f"{facts['done']:,} / {facts['total']:,} trials · best {best} · "
                           f"${facts['spend']:.2f} · {facts['updated'][:16].replace('T', ' ')}")  # fmt: skip
        if ss.ar_selected and WS.get(ss.ar_selected):
            _quick_actions(ss.ar_selected)
        with st.expander("Register existing folder", icon=":material/create_new_folder:"):
            folder = st.text_input("Results folder", key="ar_mount",
                                   placeholder="D:/evals/rack-audit/results",
                                   help=guide.tip("ar_mount"))  # fmt: skip
            if st.button("Add to sidebar", key="ar_mount_go", width="stretch"):
                entry = WS.register_existing(Path(folder)) if folder.strip() else None
                if entry is None:
                    st.error("That folder has no valid config.json from a Model Arena session.")
                else:
                    ss.ar_selected = entry.slug
                    st.rerun()


def _quick_actions(slug: str) -> None:
    entry = WS.get(slug)
    if entry is None:
        return
    cfg = storage.load_config(Path(entry.results_dir))
    st.caption(f"Actions for **{entry.name}**")
    with st.container(horizontal=True, gap="small"):
        if cfg is not None and st.button("Clone config", icon=":material/content_copy:",
                                         key="ar_clone", help=guide.tip("ar_clone")):  # fmt: skip
            _load_form(storage.clone_config(cfg, f"{cfg.session_name} copy", WS))
            st.rerun()
        with st.popover("Rename / tag", icon=":material/edit:", help=guide.tip("ar_rename")):
            name = st.text_input("Session name", entry.name, key=f"ar_rn_{slug}")
            client = st.text_input("Client / project", entry.client, key=f"ar_rc_{slug}")
            tags = st.text_input(
                "Tags (comma separated)", ", ".join(entry.tags), key=f"ar_rt_{slug}"
            )
            if st.button("Save", type="primary", key=f"ar_rs_{slug}", width="stretch"):
                WS.rename(slug, name.strip() or entry.name, client.strip(),
                          [t.strip() for t in tags.split(",") if t.strip()])  # fmt: skip
                st.rerun()
    others = {e.slug: e.name for e in WS.entries() if e.slug != slug}
    if others:
        st.selectbox("Compare against…", list(others), index=None, key="ar_compare",
                     format_func=others.__getitem__, placeholder="Pick a session to diff",
                     help=guide.tip("ar_compare"))  # fmt: skip
    with st.popover("Archive / remove from sidebar", icon=":material/archive:", width="stretch",
                    help=guide.tip("ar_archive")):  # fmt: skip
        st.caption("Removes the session from this list. Its folders on disk are kept, and "
                   "Register existing folder brings it back.")  # fmt: skip
        if st.button("Remove from sidebar", key=f"ar_arch_{slug}", width="stretch"):
            WS.archive(slug)
            st.session_state.ar_selected = None
            st.rerun()


# ------------------------------------------------------------------ new session: setup cards
def _setup_card() -> dict[str, Any]:
    ss = st.session_state
    with st.container(border=True):
        st.markdown("**Session setup & storage**")
        st.text_input("Session name", key="ar_name", on_change=_on_name, max_chars=120,
                      help=guide.tip("ar_name"))  # fmt: skip
        slug = WS.unique_slug(storage.slugify(str(ss.ar_name)))
        st.caption(f"Session id `{slug}`")
        a, b = st.columns(2)
        a.text_input("Organization / client", key="ar_client", placeholder="Optional; on reports",
                     help=guide.tip("ar_client"))  # fmt: skip
        b.text_input("Evaluator", key="ar_eval", placeholder="Optional", help=guide.tip("ar_eval"))
        st.text_area("Notes", key="ar_notes", height=68, placeholder="What this run is for",
                     help=guide.tip("ar_notes"))  # fmt: skip
        mode = st.segmented_control("Dataset folder", list(DATA_MODES), key="ar_data_mode",
                                    default="Generate new dataset", required=True,
                                    width="stretch", help=guide.tip("ar_data_mode"))  # fmt: skip
        reuse = DATA_MODES[str(mode)] == "reuse"
        items: list[ItemRecord] = []
        if reuse:
            options = _dataset_options()
            picked = st.selectbox("Certified dataset", options, index=0 if options else None,
                                  key="ar_reuse_dir", accept_new_options=True,
                                  placeholder="Pick a dataset or paste a folder path",
                                  help=guide.tip("ar_reuse_dir"))  # fmt: skip
            data_dir = str(picked or "")
            items = storage.dataset_items(storage.resolve(data_dir)) if data_dir else []
            if data_dir and not items:
                st.warning("This folder holds no certified dataset (no dataset_manifest.json "
                           "or index.jsonl).", icon=":material/folder_off:")  # fmt: skip
            elif items:
                sha = storage.dataset_fingerprint(items)
                certified = sum(i.certified for i in items)
                st.caption(f"{certified:,} certified items in "
                           f"{len(storage.dataset_blocks(items))} block(s) · sha256 `{sha[:16]}…`. "
                           "Frozen: the same images and answers for every model.")  # fmt: skip
        else:
            data_dir = st.text_input("Dataset folder (data_dir)", key="ar_data_dir",
                                     help=guide.tip("ar_data_dir"))  # fmt: skip
        results_dir = st.text_input("Results & reports folder (results_dir)", key="ar_results_dir",
                                    help=guide.tip("ar_results_dir"))  # fmt: skip
        if results_dir and (storage.resolve(results_dir) / storage.RESULTS).is_file():
            st.warning("This results folder already holds a session. Open it from the sidebar to "
                       "resume, or choose an empty folder.", icon=":material/warning:")  # fmt: skip
    return {"slug": slug, "reuse": reuse, "data_dir": data_dir, "results_dir": results_dir,
            "items": items}  # fmt: skip


def _dataset_options() -> list[str]:
    seen: list[str] = []
    for entry in WS.entries(include_archived=True):
        if (Path(entry.data_dir) / storage.MANIFEST).is_file():
            seen.append(relative(Path(entry.data_dir)))
    seen += [relative(p) for p in forensics.find_datasets()]
    return list(dict.fromkeys(seen))


def _mode_card() -> dict[str, Any]:
    ss = st.session_state
    with st.container(border=True):
        st.markdown("**Execution mode & guardrails**")
        mode = MODES[str(st.segmented_control("Execution mode", list(MODES), key="ar_mode",
                                              default="Offline Mock (No Key)", required=True,
                                              width="stretch", help=guide.tip("ar_mode")))]  # fmt: skip
        found, source = keys.detect()
        if mode == "live":
            if found:
                st.badge(f"Key loaded from {source} ({keys.mask(found)})", color="green",
                         icon=":material/key:")  # fmt: skip
            else:
                st.badge("No .env key detected", color="orange", icon=":material/key_off:")
            if st.toggle("Override with custom OpenRouter API key", key="ar_override",
                         help=guide.tip("ar_override")):  # fmt: skip
                st.text_input("OpenRouter API key", type="password", key=SECRET_KEY,
                              placeholder="sk-or-…", help=guide.tip("ar_key"))  # fmt: skip
        else:
            st.caption("Simulated models answer the certified items offline: no key, no network, "
                       "no charges. Costs shown are what a live run would cost.")  # fmt: skip
        a, b, c = st.columns(3)
        a.number_input("Spend cap ($)", 0.0, 250.0, 0.50, 0.25, key="ar_cap", format="%.2f",
                       help=guide.tip("ar_cap"))  # fmt: skip
        b.number_input("Max concurrency", 1, 8, 2, key="ar_conc", help=guide.tip("ar_conc"))
        c.number_input("Rate limit (RPM)", 1, 1000, 60, key="ar_rpm", help=guide.tip("ar_rpm"))
        k, t = st.columns(2)
        k.number_input("Repeats per item (k)", 1, MAX_REPEATS, 1, key="ar_repeats",
                       help=guide.tip("ar_repeats"))  # fmt: skip
        t.slider("Temperature", 0.0, 1.0, 0.0, 0.05, key="ar_temperature",
                 help=guide.tip("ar_temperature"))  # fmt: skip
        delay = 0.0
        if mode == "mock":
            delay = float(st.number_input("Simulated latency per call (s)", 0.0, 5.0, 0.2, 0.1,
                                          key="ar_delay", help=guide.tip("ar_delay")))  # fmt: skip
    return {"mode": mode, "cap": float(ss.ar_cap), "conc": int(ss.ar_conc),
            "rpm": int(ss.ar_rpm), "delay": delay, "repeats": int(ss.ar_repeats),
            "temperature": float(ss.ar_temperature)}  # fmt: skip


def _models_card(live_mode: bool) -> tuple[list[str], catalog.Catalog]:
    ss = st.session_state
    cat = _catalog(bool(ss.ar_live_catalog))
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown("**Vision models**", width="content")
            st.badge("Live catalog" if cat.source == "live" else "Curated offline catalog",
                     color="green" if cat.source == "live" else "gray",
                     icon=":material/cloud_done:" if cat.source == "live" else ":material/inventory_2:")  # fmt: skip
            if live_mode and st.button("Refresh live catalog", icon=":material/refresh:",
                                       key="ar_refresh", type="tertiary",
                                       help=guide.tip("ar_refresh")):  # fmt: skip
                ss.ar_live_catalog = True
                _catalog.clear()
                st.rerun()
        if cat.note:
            st.caption(cat.note)
        free_only = st.toggle("Free models only (:free)", key="ar_free_only",
                              help=guide.tip("ar_free_only"))  # fmt: skip
        options = list(dict.fromkeys([*cat.ids(free_only), *ss.ar_custom_models, *ss.ar_models]))
        labels = {m.id: m.label for m in cat.models}
        st.multiselect("Default models", options, key="ar_models",
                       format_func=lambda m: labels.get(m, f"{m} · custom"),
                       placeholder="Search vision models", help=guide.tip("ar_models"))  # fmt: skip
        with st.container(horizontal=True, vertical_alignment="bottom"):
            st.text_input("Custom model ID", key="ar_custom_input",
                          placeholder="vendor/model, or with :nitro / :floor / :exacto",
                          help=guide.tip("ar_custom_model"))  # fmt: skip
            st.button("Add model", icon=":material/add:", key="ar_custom_go", on_click=_add_custom,
                      help=guide.tip("ar_add_model"))  # fmt: skip
    return list(ss.ar_models), cat


def _add_custom() -> None:
    ss = st.session_state
    model = str(ss.get("ar_custom_input", "")).strip()
    if model and "/" in model and " " not in model:
        ss.ar_custom_models = list(dict.fromkeys([*ss.ar_custom_models, model]))
        ss.ar_models = list(dict.fromkeys([*ss.get("ar_models", []), model]))
        ss.ar_custom_input = ""


# ------------------------------------------------------------------ campaign builder
def _add_block() -> None:
    blocks = st.session_state.ar_blocks
    used = {b["block_id"] for b in blocks}
    k = len(blocks) + 1
    while f"b{k}" in used:
        k += 1
    seed = 1000 * k
    blocks.append({"block_id": f"b{k}", "domain": "tangle", "preset": "Clean", "n": 10,
                   "seed": seed, "models": None})  # fmt: skip


def _remove_block(block_id: str) -> None:
    st.session_state.ar_blocks = [
        b for b in st.session_state.ar_blocks if b["block_id"] != block_id
    ]


def _quick_n(block_id: str) -> None:
    pick = st.session_state.get(f"blk_{block_id}_pick")
    if pick:
        st.session_state[f"blk_{block_id}_n"] = int(pick)


def _block_editor(spec: dict[str, Any], models: list[str], frozen: bool) -> dict[str, Any]:
    ss, bid = st.session_state, spec["block_id"]
    with st.container(border=True, key=f"ar_block_{bid}"):
        head, remove = st.columns([12, 1], vertical_alignment="center")
        head.markdown(f"**Block {bid.removeprefix('b')}**")
        if not frozen and len(ss.ar_blocks) > 1:
            remove.button("", icon=":material/delete:", key=f"blk_{bid}_rm", type="tertiary",
                          on_click=_remove_block, args=(bid,), help="Remove block")  # fmt: skip
        cols = st.columns([1.3, 1.3, 1.6, 1], vertical_alignment="bottom")
        domains = list(DOMAIN_LABELS)
        domain = cols[0].selectbox(
            "Domain", domains, index=domains.index(spec["domain"]) if spec["domain"] in domains
            else 0, key=f"blk_{bid}_domain", format_func=DOMAIN_LABELS.__getitem__, disabled=frozen,
            help=guide.tip("blk_domain"),
        )  # fmt: skip
        presets = list(PRESETS) if spec["preset"] in PRESETS else [spec["preset"]]
        preset = cols[1].selectbox("Preset", presets, index=presets.index(spec["preset"]),
                                   key=f"blk_{bid}_preset", disabled=frozen,
                                   help=guide.tip("blk_preset"))  # fmt: skip
        ss.setdefault(f"blk_{bid}_n", int(spec["n"]))
        n = cols[2].number_input("Sample count", 1, 100_000, key=f"blk_{bid}_n", disabled=frozen,
                                 help=guide.tip("blk_n"))  # fmt: skip
        ss.setdefault(f"blk_{bid}_seed", int(spec["seed"]))
        seed = cols[3].number_input("Start seed", 0, 2**31 - 1, key=f"blk_{bid}_seed",
                                    disabled=frozen, help=guide.tip("blk_seed"))  # fmt: skip
        if not frozen:
            st.pills("Quick pick", list(planning.QUICK_N), key=f"blk_{bid}_pick",
                     on_change=_quick_n, args=(bid,), label_visibility="collapsed",
                     format_func=lambda v: f"{v:,}", help=guide.tip("blk_quick"))  # fmt: skip
        ss.setdefault(f"blk_{bid}_default", spec.get("models") is None)
        use_default = st.toggle("Use default models", key=f"blk_{bid}_default",
                                help=guide.tip("blk_default"))  # fmt: skip
        own: list[str] | None = None
        if not use_default:
            choices = list(
                dict.fromkeys([*models, *ss.ar_custom_models, *(spec.get("models") or [])])
            )
            ss.setdefault(f"blk_{bid}_models", list(spec.get("models") or models))
            own = list(st.multiselect("Models for this block", choices, key=f"blk_{bid}_models",
                                      help=guide.tip("blk_models")))  # fmt: skip
    return {"block_id": bid, "domain": domain, "preset": preset, "n": int(n), "seed": int(seed),
            "models": own}  # fmt: skip


def _campaign(models: list[str], setup: dict[str, Any]) -> list[dict[str, Any]]:
    ss = st.session_state
    frozen = setup["reuse"]
    st.markdown("#### Campaign builder")
    if frozen:
        st.caption("Blocks come from the frozen dataset: domains, presets and counts are fixed. "
                   "Choose which models answer each block.")  # fmt: skip
        found = storage.dataset_blocks(setup["items"])
        specs = [{**b, "n": b["certified"] or b["n"], "models": None} for b in found]
    else:
        specs = list(ss.ar_blocks)
    out = [_block_editor(spec, models, frozen) for spec in specs]
    if not frozen:
        ss.ar_blocks = out
        st.button("Add block", icon=":material/add:", key="ar_add_block", on_click=_add_block,
                  help=guide.tip("ar_add_block"))  # fmt: skip
    return out


# ------------------------------------------------------------------ pre-flight and launch
def _build_config(setup: dict[str, Any], mode: dict[str, Any], models: list[str],
                  blocks: list[dict[str, Any]]) -> SessionConfig:  # fmt: skip
    ss = st.session_state
    return SessionConfig(
        session_name=str(ss.ar_name).strip() or "Untitled session", slug=setup["slug"],
        client=str(ss.get("ar_client", "")).strip(), evaluator=str(ss.get("ar_eval", "")).strip(),
        notes=str(ss.get("ar_notes", "")).strip(), mode=mode["mode"],
        data_mode="reuse" if setup["reuse"] else "generate",
        data_dir=setup["data_dir"], results_dir=setup["results_dir"],
        default_models=models or ["(none)"],
        blocks=[BlockSpec.model_validate(b) for b in blocks],
        spend_cap_usd=mode["cap"], max_concurrency=mode["conc"], rpm=mode["rpm"],
        mock_delay_s=mode["delay"], repeats=mode["repeats"],
        temperature=mode.get("temperature", 0.0),
    )  # fmt: skip


def _launch_bar(setup: dict[str, Any], mode: dict[str, Any], models: list[str],
                blocks: list[dict[str, Any]], cat: catalog.Catalog) -> None:  # fmt: skip
    with st.container(border=True, key="ar_preflight"):
        st.markdown("**Pre-flight estimate**")
        try:
            cfg = _build_config(setup, mode, models, blocks)
        except ValidationError as exc:
            st.error("Check the setup: " + "; ".join(e["msg"] for e in exc.errors()[:3]))
            return
        reused = sum(i.certified for i in setup["items"])
        pf = planning.preflight(cfg, cat, reused)
        a, b, c, d = st.columns(4)
        if setup["reuse"]:
            a.metric("Certified images", f"{pf.images_reused:,}", "reused, frozen",
                     delta_color="off", delta_arrow="off", border=True,
                     help=guide.tip("pf_images"))  # fmt: skip
        else:
            a.metric("Images to generate", f"{pf.images_to_generate:,}",
                     f"~{planning.human_duration(pf.gen_seconds)}", delta_color="off",
                     delta_arrow="off", border=True, help=guide.tip("pf_images"))  # fmt: skip
        per = "Σ N × models" + (f" × k={cfg.repeats}" if cfg.repeats > 1 else " per block")
        b.metric("Model trials", f"{pf.trials:,}", per, delta_color="off",
                 delta_arrow="off", border=True, help=guide.tip("pf_trials"))  # fmt: skip
        c.metric("Est. evaluation time", planning.human_duration(pf.eval_seconds),
                 f"{cfg.max_concurrency} concurrent · {cfg.rpm} RPM", delta_color="off",
                 delta_arrow="off", border=True, help=guide.tip("pf_time"))  # fmt: skip
        live = cfg.mode == "live"
        d.metric("Est. max cost" if live else "Cost if run live", f"${pf.max_cost_usd:,.2f}",
                 f"cap ${cfg.spend_cap_usd:.2f}" if live else "$0.00 charged (mock)",
                 delta_color="off", delta_arrow="off", border=True,
                 help=guide.tip("pf_cost"))  # fmt: skip
        problems = []
        if not models:
            problems.append("Pick at least one default model.")
        if any(b.get("models") == [] for b in blocks):
            problems.append("A block with its own models needs at least one model.")
        key, _ = _api_key()
        if live and not key:
            problems.append("Live mode needs an OpenRouter API key: add it to .env or use the "
                            "override field.")  # fmt: skip
        if setup["reuse"] and not setup["items"]:
            problems.append("Pick a certified dataset to reuse.")
        for p in problems:
            st.warning(p, icon=":material/rule:")
        if pf.over_cap:
            st.info(f"The estimate exceeds the ${cfg.spend_cap_usd:.2f} cap: the run stops "
                    "cleanly when the cap is reached and can be resumed with a higher cap.",
                    icon=":material/savings:")  # fmt: skip
        if st.button("Launch campaign", type="primary", icon=":material/rocket_launch:",
                     width="stretch", disabled=bool(problems), key="ar_launch",
                     help=guide.tip("ar_launch")):  # fmt: skip
            found, process = planning.start(cfg, WS, cat, key if live else None)
            if found:
                for p in found:
                    st.error(p, icon=":material/error:")
                return
            st.session_state.ar_procs[cfg.slug] = process
            st.session_state.ar_selected = cfg.slug
            st.rerun()


def _new_session_view() -> None:
    st.markdown("### New evaluation session")
    st.caption("Certified items in their own dataset folder, every model's answers in a "
               "separate results folder. Runs resume after any interruption.")  # fmt: skip
    left, right = st.columns(2, gap="medium")
    with left:
        setup = _setup_card()
    with right:
        mode = _mode_card()
    models, cat = _models_card(mode["mode"] == "live")
    blocks = _campaign(models, setup)
    _launch_bar(setup, mode, models, blocks, cat)


# ------------------------------------------------------------------ session view
def _controls(entry_dir: Path, cfg: SessionConfig, state: str, running: bool) -> None:
    key, _ = _api_key()
    with st.container(horizontal=True, gap="small"):
        if st.button("Pause", icon=":material/pause:", disabled=not running, key="ar_pause",
                     help=guide.tip("ar_pause")):  # fmt: skip
            storage.send_control(entry_dir, "pause")
            st.toast("Pausing after the calls in flight.", icon=":material/pause:")
        if st.button("Stop", icon=":material/stop:", disabled=not running, key="ar_stop",
                     help=guide.tip("ar_stop")):  # fmt: skip
            storage.send_control(entry_dir, "stop")
            st.toast("Stopping after the calls in flight.", icon=":material/stop:")
        can_resume = not running and state != "done"
        label = "Resume" if state not in ("idle",) else "Start"
        if st.button(label, type="primary", icon=":material/play_arrow:", disabled=not can_resume,
                     key="ar_resume", help=guide.tip("ar_resume")):  # fmt: skip
            _relaunch(entry_dir, key, retry=False)
        failed = storage.read_status(entry_dir).failed_trials
        if st.button(f"Retry failed API calls only ({failed})", icon=":material/replay:",
                     disabled=running or failed == 0, key="ar_retry",
                     help=guide.tip("ar_retry")):  # fmt: skip
            _relaunch(entry_dir, key, retry=True)
    with st.expander("Copy headless CLI command", icon=":material/terminal:"):
        st.code(storage.cli_command(entry_dir), language="powershell", wrap_lines=True)
        st.caption("Runs or resumes this session from a terminal or CI job. Live runs read the key "
                   "from the OPENROUTER_API_KEY environment variable.")  # fmt: skip


def _relaunch(entry_dir: Path, key: str | None, retry: bool) -> None:
    problems, process = planning.resume(entry_dir, WS, key, retry_failed=retry)
    if problems:
        for p in problems:
            st.error(p, icon=":material/error:")
        return
    cfg = storage.load_config(entry_dir)
    if cfg is not None:
        st.session_state.ar_procs[cfg.slug] = process
    st.rerun()


def _kpis(cfg: SessionConfig, items: list[ItemRecord], trials: dict[str, TrialRecord],
          status: Any) -> None:  # fmt: skip
    lb = report.leaderboard(cfg, trials)
    best = next((r for r in lb["models"] if r["n"]), None)
    f = report.factory_telemetry(cfg, items)["overall"]
    scored = [t for t in trials.values() if t.status == "ok"]
    latency = report.summary([t.latency_ms for t in scored])["mean"]
    a, b, c, d, e = st.columns(5)
    a.metric("Completed trials", f"{status.completed_trials:,} / {status.total_trials:,}",
             f"{status.failed_trials} failed" if status.failed_trials else None,
             delta_color="off", delta_arrow="off", border=True,
             help=guide.tip("kpi_trials"))  # fmt: skip
    b.metric("Certification yield", report.pct(f["yield"]),
             f"{f['certified']:,} of {f['items']:,} items", delta_color="off", delta_arrow="off",
             border=True, help=guide.tip("kpi_yield_arena"))  # fmt: skip
    if best:
        c.metric("Best model accuracy", report.pct(best["accuracy"]),
                 f"{report.short(best['model'])} · CI {report.pct(best['ci_lo'], 0)}–"
                 f"{report.pct(best['ci_hi'], 0)}", delta_color="off", delta_arrow="off",
                 border=True, help=guide.tip("kpi_best"))  # fmt: skip
    else:
        c.metric("Best model accuracy", "—", "no scored trial yet", delta_color="off",
                 delta_arrow="off", border=True, help=guide.tip("kpi_best"))  # fmt: skip
    spend_note = f"cap ${cfg.spend_cap_usd:.2f}" if cfg.mode == "live" else "simulated estimate"
    d.metric("Total spend", report.usd(status.spend_usd), spend_note, delta_color="off",
             delta_arrow="off", border=True, help=guide.tip("kpi_spend"))  # fmt: skip
    gen, verify = f["gen_ms"]["mean"], f["verify_ms"]["mean"]
    pair = "—" if gen is None else f"{gen / 1000:.1f} / {(verify or 0) / 1000:.1f} s"
    e.metric("Mean gen / verify", pair, f"API {report.ms(latency)}", delta_color="off",
             delta_arrow="off", border=True, help=guide.tip("kpi_genverify"))  # fmt: skip


def _progress(status: Any, items: list[ItemRecord]) -> None:
    if status.phase == "generate" and status.total_items:
        frac = status.items_ready / status.total_items
        text = f"Generating and certifying: {status.items_ready:,} of {status.total_items:,} items"
    else:
        total = max(status.total_trials, 1)
        frac = status.completed_trials / total
        text = f"Model trials: {status.completed_trials:,} of {status.total_trials:,}"
    st.progress(min(max(frac, 0.0), 1.0), text=text)
    busy = sum(i.gen_ms + i.verify_ms for i in items)
    if busy:
        st.caption(f"Factory throughput {len(items) / (busy / 60000):.1f} items/min per worker · "
                   f"spend ${status.spend_usd:.4f}"
                   + (f" of ${status.spend_cap_usd:.2f}" if status.spend_cap_usd else ""))  # fmt: skip


def _leaderboard(cfg: SessionConfig, trials: dict[str, TrialRecord]) -> None:
    cells = report.leaderboard(cfg, trials)["cells"]
    rows = [{
        "Block": c["block"], "Model": report.short(c["model"]),
        "Accuracy": None if c["accuracy"] is None else 100 * c["accuracy"],
        "95% CI": f"{report.pct(c['ci_lo'])} – {report.pct(c['ci_hi'])}" if c["n"] else "—",
        "Correct / N": f"{c['correct']} / {c['n']}",
        "p50 / p95 latency": f"{report.ms(c['latency_p50_ms'])} / {report.ms(c['latency_p95_ms'])}",
        "Tokens": c["prompt_tokens"] + c["completion_tokens"], "Cost": c["cost_usd"],
        "$ / correct": c["cost_per_correct"], "API errors": c["api_errors"],
    } for c in cells]  # fmt: skip
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch", key="ar_matrix",
        alt="Block by model leaderboard with accuracy, confidence interval, latency and cost",
        column_config={
            "Block": st.column_config.TextColumn(help=guide.tip("col_block")),
            "Model": st.column_config.TextColumn(help=guide.tip("col_model")),
            "Accuracy": st.column_config.ProgressColumn(format="%.1f%%", min_value=0.0,
                                                        max_value=100.0,
                                                        help=guide.tip("col_accuracy")),
            "95% CI": st.column_config.TextColumn(help=guide.tip("col_ci")),
            "Correct / N": st.column_config.TextColumn(help=guide.tip("col_correct")),
            "p50 / p95 latency": st.column_config.TextColumn(help=guide.tip("col_latency")),
            "Cost": st.column_config.NumberColumn(format="$%.4f", help=guide.tip("col_cost")),
            "$ / correct": st.column_config.NumberColumn(
                format="$%.4f", help=guide.tip("col_cost_per_correct")
            ),
            "Tokens": st.column_config.NumberColumn(format="%d", help=guide.tip("col_tokens")),
            "API errors": st.column_config.NumberColumn(help=guide.tip("col_api_errors")),
        },
    )  # fmt: skip


def _live_body(entry_dir: Path, cfg: SessionConfig, was_running: bool) -> None:
    status = storage.read_status(entry_dir)
    child = st.session_state.ar_procs.get(cfg.slug)
    alive = (status.state == "running" and runner.pid_alive(status.pid)) or (
        child is not None and child.poll() is None
    )
    if was_running and not alive:
        st.rerun(scope="app")  # the run ended: redraw the page once, without polling
    items = storage.dataset_items(storage.resolve(cfg.data_dir))
    items = [i for i in items if i.block_id in {b.block_id for b in cfg.blocks}]
    trials = storage.load_trials(entry_dir)
    _progress(status, items)
    _kpis(cfg, items, trials, status)
    st.markdown("**Block × model leaderboard**")
    _leaderboard(cfg, trials)
    _reliability(cfg, trials)


def _reliability(cfg: SessionConfig, trials: dict[str, TrialRecord]) -> None:
    """pass@1 / pass@k / pass^k per (block, model) when each item is asked k > 1 times."""
    rel = report.reliability(cfg, trials)
    if rel is None:
        return
    st.markdown(f"**Reliability over k = {rel['k']} repeats**")
    rows = [{
        "Block": c["block"], "Model": report.short(c["model"]), "Items": c["items"],
        "pass@1": None if c["pass_at_1"] is None else 100 * c["pass_at_1"],
        "pass@k": None if c["pass_at_k"] is None else 100 * c["pass_at_k"],
        "pass^k": None if c["pass_hat_k"] is None else 100 * c["pass_hat_k"],
    } for c in rel["cells"]]  # fmt: skip
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch", key="ar_reliability",
        alt="pass@1, pass@k and pass^k per block and model",
        column_config={
            "Items": st.column_config.NumberColumn(help=guide.tip("col_rel_items")),
            "pass@1": st.column_config.NumberColumn(format="%.1f%%", help=guide.tip("col_pass1")),
            "pass@k": st.column_config.NumberColumn(format="%.1f%%", help=guide.tip("col_passk")),
            "pass^k": st.column_config.NumberColumn(format="%.1f%%", help=guide.tip("col_passhatk")),
        },
    )  # fmt: skip
    for finding in rel["findings"]:
        st.warning(finding, icon=":material/repeat:")


def _model_card(
    trial: TrialRecord | None, model: str, item: ItemRecord, cfg: SessionConfig
) -> None:
    with st.container(border=True):
        st.markdown(f"**{report.short(model)}**")
        if trial is None:
            st.badge("Not asked yet", color="gray", icon=":material/hourglass_empty:")
            return
        if trial.status == "api_error":
            st.badge("API error", color="orange", icon=":material/cloud_off:")
            st.caption(trial.error or "The call failed.")
            return
        if trial.correct:
            st.badge("Pass", color="green", icon=":material/check_circle:")
        else:
            st.badge("Fail", color="red", icon=":material/cancel:")
        st.markdown(f"Answered **{trial.extracted or '(nothing)'}**")
        if trial.error_category:
            st.caption(trial.error_category.replace("_", " "))
        st.caption(f"{report.ms(trial.latency_ms)} · {report.usd(trial.cost_usd)}"
                   + (" · simulated" if trial.simulated else ""))  # fmt: skip
        with st.expander("Raw reply", icon=":material/notes:"):
            st.text(trial.reply or "(empty)")
        if not trial.correct and st.button("Send failure to Forensics", icon=":material/troubleshoot:",
                                           key=f"ar_fx_{trial.trial_id}", width="stretch",
                                           help=guide.tip("ar_send_forensics")):  # fmt: skip
            _to_forensics(item, trial, cfg)


def _to_forensics(item: ItemRecord, trial: TrialRecord, cfg: SessionConfig) -> None:
    out = storage.resolve(cfg.results_dir) / "forensics" / item.bundle_id
    data = storage.resolve(cfg.data_dir)
    got = run_with_status(
        "Build counterfactual twins for this failure",
        lambda: arena_forensics.build_suite(item, {trial.model: trial.reply}, data, out),
    )
    if got is not None:
        st.session_state.cf_out = relative(out)
        st.session_state.fx_section = "Counterfactual twins"
        st.switch_page("app_pages/forensics.py")


def _inspector(cfg: SessionConfig) -> None:
    data = storage.resolve(cfg.data_dir)
    items = [i for i in storage.dataset_items(data) if i.certified]
    if not items:
        st.info("No certified item yet. Press Start or Resume above; items appear here as the "
                "factory certifies them.",
                icon=":material/image:")  # fmt: skip
        return
    trials = storage.load_trials(storage.resolve(cfg.results_dir))
    blocks = {b.block_id: b.label for b in cfg.blocks}
    a, b, c = st.columns([1.2, 2, 1], vertical_alignment="bottom")
    block = a.selectbox("Block", list(blocks), format_func=blocks.__getitem__, key="ar_ins_block",
                        help=guide.tip("ar_ins_block"))  # fmt: skip
    mine = sorted((i for i in items if i.block_id == block), key=lambda i: i.item_index)
    if not mine:
        st.caption("This block has no certified item yet.")
        return

    def label(i: ItemRecord) -> str:
        got = [
            trials.get(f"{i.block_id}:{i.item_index}:{m}")
            for m in cfg.models_for(cfg.block(i.block_id))
        ]
        done = [t for t in got if t and t.status == "ok"]
        right = sum(bool(t.correct) for t in done)
        return f"#{i.item_index:03d} · {i.family} · truth {i.truth} · {right}/{len(done)} correct"

    item = b.selectbox("Item", mine, format_func=label, key=f"ar_ins_item_{block}",
                       help=guide.tip("ar_ins_item"))  # fmt: skip
    witness = c.toggle("Witness overlay", key="ar_ins_witness", help=guide.tip("ar_ins_witness"))
    left, right = st.columns([1, 1.25], gap="medium")
    with left:
        bundle = data / item.dir
        if witness:
            cert = read_json(bundle / "task.json")["certificate"]
            st.image(factory.witness_overlay(bundle, cert), width=IMAGE_PX,
                     alt="Item with the kernel's witness evidence")  # fmt: skip
        else:
            st.image(factory.beauty(bundle), width=IMAGE_PX, alt="Certified item")
    with right:
        with st.container(border=True):
            st.markdown(item.question.split("\nEnd your reply")[0])
            st.markdown(f"Certified answer **{item.truth}** · `{item.bundle_id}`")
        models = cfg.models_for(cfg.block(item.block_id))
        for start in range(0, len(models), 3):
            row = models[start : start + 3]
            for col, model in zip(st.columns(len(row)), row, strict=True):
                with col:
                    _model_card(trials.get(f"{item.block_id}:{item.item_index}:{model}"), model,
                                item, cfg)  # fmt: skip


def _compare_cfg() -> tuple[SessionConfig, dict[str, TrialRecord]] | None:
    slug = st.session_state.get("ar_compare")
    entry = WS.get(slug) if slug else None
    if entry is None:
        return None
    cfg = storage.load_config(Path(entry.results_dir))
    if cfg is None:
        return None
    return cfg, storage.load_trials(Path(entry.results_dir))


def _report_tab(cfg: SessionConfig, entry_dir: Path) -> None:
    key, _ = _api_key()
    compare = _compare_cfg()
    st.caption("Deterministic statistics computed offline from this session's files: Wilson 95% "
               "intervals, cross-block z and Fisher tests, error taxonomy and timing.")  # fmt: skip
    with st.container(horizontal=True, vertical_alignment="bottom"):
        brief = st.checkbox("Include AI Executive Brief", key="ar_brief", disabled=not key,
                            help=guide.tip("ar_brief"))  # fmt: skip
        model = st.selectbox("Synthesis model", SYNTH_MODELS, key="ar_brief_model",
                             disabled=not (brief and key), help=guide.tip("ar_brief_model"))  # fmt: skip
        go = st.button("Generate report", type="primary", icon=":material/description:",
                       key="ar_report_go", help=guide.tip("ar_report_go"))  # fmt: skip
    if compare:
        st.caption(f"Includes a regression diff against **{compare[0].session_name}**.")
    if go:

        def build() -> dict[str, Path]:
            text = None
            if brief and key:
                items = storage.dataset_items(storage.resolve(cfg.data_dir))
                metrics = report.compute_metrics(cfg, items, storage.load_trials(entry_dir),
                                                 storage.read_status(entry_dir), compare)  # fmt: skip
                text = report.ai_brief(metrics, str(model), key)
            return report.export(cfg, compare, text)

        paths = run_with_status("Compute statistics and write the report", build)
        if paths:
            st.session_state[f"ar_report_{cfg.slug}"] = {k: str(v) for k, v in paths.items()}
    written = st.session_state.get(f"ar_report_{cfg.slug}")
    reports = entry_dir / "reports"
    if not written and (reports / "report.html").is_file():
        written = {p.name: str(p) for p in reports.iterdir() if p.is_file()}
    if not written:
        st.caption("No report yet. Press Generate report to write it.")
        return
    files = [Path(p) for p in written.values() if Path(p).is_file()]
    st.success(f"Report written to `{relative(reports)}`.", icon=":material/check_circle:")
    with st.container(horizontal=True, gap="small"):
        for path in sorted(files, key=lambda p: p.suffix):
            st.download_button(path.name, path.read_bytes(), path.name, key=f"ar_dl_{path.name}",
                               icon=":material/download:")  # fmt: skip
    html_path = reports / "report.html"
    if html_path.is_file():
        with st.expander("Preview report.html", icon=":material/visibility:"):
            components.html(html_path.read_text(encoding="utf-8"), height=720, scrolling=True)


def _compare_tab(cfg: SessionConfig, trials: dict[str, TrialRecord]) -> None:
    compare = _compare_cfg()
    if compare is None:
        st.info("Pick a session under Compare against… in the sidebar to see a regression diff.",
                icon=":material/compare_arrows:")  # fmt: skip
        return
    diff = report.regression(cfg, trials, compare[0], compare[1])
    if not diff["rows"]:
        st.caption("The two sessions share no (block, model) pair to compare.")
        return
    rows = [{"Block": r["block"], "Model": report.short(r["model"]),
             "This session": 100 * r["acc_a"], compare[0].session_name: 100 * r["acc_b"],
             "Δ pp": r["delta_pp"], "p (z)": report.pval(r["p_z"]),
             "Δ p50 latency": report.ms(r["latency_delta_ms"]), "Δ cost": r["cost_delta_usd"],
             "Verdict": r["verdict"]} for r in diff["rows"]]  # fmt: skip
    pctcol = st.column_config.NumberColumn(format="%.1f%%")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"This session": pctcol, compare[0].session_name: pctcol,
                                "Δ pp": st.column_config.NumberColumn(format="%+.1f"),
                                "Δ cost": st.column_config.NumberColumn(format="$%+.4f")})  # fmt: skip


def _session_view(slug: str) -> None:
    entry = WS.get(slug)
    if entry is None:
        st.session_state.ar_selected = None
        st.rerun()
        return
    entry_dir = Path(entry.results_dir)
    cfg = storage.load_config(entry_dir)
    if cfg is None:
        st.error(f"`{entry_dir}` no longer holds a valid config.json. Remove it from the sidebar "
                 "or register the folder again.", icon=":material/folder_off:")  # fmt: skip
        return
    status = storage.read_status(entry_dir)
    child = st.session_state.ar_procs.get(slug)
    launching = child is not None and child.poll() is None and status.state != "running"
    running = (status.state == "running" and runner.pid_alive(status.pid)) or launching
    state = "running" if launching else status.state
    if status.state == "running" and not running:
        state = "stopped"
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown(f"### {cfg.session_name}", width="content")
        _pill(state)
        st.badge("Live (OpenRouter)" if cfg.mode == "live" else "Offline mock",
                 color="violet" if cfg.mode == "live" else "gray",
                 icon=":material/cloud:" if cfg.mode == "live" else ":material/science:")  # fmt: skip
        if cfg.client:
            st.badge(cfg.client, color="gray", icon=":material/apartment:")
    st.caption(f"Dataset `{relative(storage.resolve(cfg.data_dir))}` · results "
               f"`{relative(entry_dir)}` · run `{cfg.run_id or '—'}`")  # fmt: skip
    if status.message:
        (st.warning if state in ("failed", "budget_stop") else st.caption)(status.message)
    if state == "stopped" and status.state == "running":
        st.warning("The runner stopped unexpectedly. Resume continues where it left off; no trial "
                   "is repeated.", icon=":material/restart_alt:")  # fmt: skip
    _controls(entry_dir, cfg, state, running)
    st.fragment(run_every=2.0 if running else None)(_live_body)(entry_dir, cfg, running)
    trials = storage.load_trials(entry_dir)
    inspect, rep, post, diff = st.tabs(
        ["Item inspector", "Report", "Post-Training Pack", "Compare"]
    )
    with inspect:
        _inspector(cfg)
    with rep:
        _report_tab(cfg, entry_dir)
    with post:
        posttrain_tab(cfg, entry_dir)
    with diff:
        _compare_tab(cfg, trials)


# ------------------------------------------------------------------ page
_init_state()
_sidebar()
if st.session_state.ar_selected:
    _session_view(str(st.session_state.ar_selected))
else:
    _new_session_view()
st.sidebar.caption(report.FOOTER)
