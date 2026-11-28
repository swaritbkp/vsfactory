"""Knotwork Visual Task Factory Studio: `uv run streamlit run apps/inspector.py`.

Entry point: page config, the compact status header and top navigation. Each page in `app_pages/`
only lays out widgets; the logic lives in `knotwork.studio` and the pipeline. Everything runs
offline and writes under `artifacts/studio/`. The theme is `.streamlit/config.toml`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

from knotwork.studio import demo_guard, guide, present

APP = Path(__file__).resolve().parent
#: left to right in the order of the real journey and of the Guided Mode steps (`guide.PAGES`)
PAGES = (
    ("studio", "studio.py", "Studio", ":material/science:"),
    ("model_arena", "arena.py", "Model Arena", ":material/leaderboard:"),
    ("forensics", "forensics.py", "Forensics", ":material/troubleshoot:"),
    ("human_eval", "human_eval.py", "Human Eval", ":material/person_check:"),
    ("export", "export.py", "Export", ":material/package_2:"),
)
HERO_STAGES = (
    (
        "studio",
        "1 · Studio",
        ":material/science:",
        "Generate & certify tasks, inspect Witness & vCoT, and curate the Working Set.",
    ),
    (
        "model_arena",
        "2 · Arena",
        ":material/leaderboard:",
        "Compare VLMs, explore the 531-call Showcase, or test the Reward Verifier Sandbox.",
    ),
    (
        "forensics",
        "3 · Forensics",
        ":material/troubleshoot:",
        "Inspect counterfactual twins, quarantine Pareto, and live fault proofs.",
    ),
    (
        "human_eval",
        "4 · Human Eval",
        ":material/fact_check:",
        "Answer blind certified items and compare your score against VLM baselines.",
    ),
    (
        "export",
        "5 · Export",
        ":material/download:",
        "Export Parquet/HF, Harbor, and RLVR datasets with SHA-256 lineage.",
    ),
)


@st.cache_data(ttl=20, show_spinner=False)
def _status() -> present.RepoStatus:
    return present.repo_status()


def _system_popover(page: str | None, status: present.RepoStatus) -> None:
    from studio_ui import guide_controls

    build = present.build_label(status)
    container = build.startswith("package ")
    verified = " · verified" if container and status["tcb_ok"] else ""
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.badge(
            build,
            icon=":material/commit:" if not container else ":material/inventory_2:",
            color="gray",
        )
        st.badge(
            f"{'v2 · ' if container else ''}kernel {status['kernel']}{verified}",
            icon=":material/fingerprint:",
            color="gray",
            help="First 12 characters of the trusted kernel's source SHA-256.",
        )
    if status["tcb_ok"]:
        st.badge(
            "TCB audit passed · Offline",
            icon=":material/verified_user:",
            color="green",
            help="The kernel passes its TCB audit (allow-listed imports, no network code) "
            "and the studio runs with the network policy set to deny.",
        )
    else:
        st.badge("TCB audit failed", icon=":material/gpp_bad:", color="red")
    if demo_guard.is_public_demo():
        st.info(demo_guard.banner(), icon=":material/shield:")
    st.divider()
    guide_controls(page)


def header(page: str | None) -> None:
    from studio_ui import guide_strip  # importable once APP is on sys.path
    from ui_chrome import switch_factory_page, ws_pill

    status = _status()
    with st.container(border=True, key="factory_hero"):
        col_brand, col_status, col_sys = st.columns([3.0, 5.2, 1.8], vertical_alignment="center")
        with col_brand:
            st.markdown(
                ":material/hub: **Knotwork** :small[:gray[· Proof-carrying visual task factory]]"
            )
        with col_status, st.container(horizontal=True, vertical_alignment="center", gap="small"):
            ws_pill()
            if demo_guard.is_public_demo():
                if demo_guard.byok_allowed():
                    st.badge(
                        "BYOK Sandbox · <=5 items · $0.50 cap · RAM-only key",
                        icon=":material/shield:",
                        color="blue",
                    )
                else:
                    st.badge(
                        "Offline Mock Sandbox · 10-item cap · /tmp",
                        icon=":material/shield:",
                        color="blue",
                    )
            else:
                st.badge("Local · Offline by default", icon=":material/verified:", color="green")
        with col_sys, st.popover("System & Guide", icon=":material/tune:", width="stretch"):
            _system_popover(page, status)

        cols = st.columns(5, gap="small")
        for col, (stage_key, label, icon, tip_text) in zip(cols, HERO_STAGES, strict=True):
            is_active = (page == stage_key) or (page == "arena" and stage_key == "model_arena")
            if (
                col.button(
                    label,
                    icon=icon,
                    type="primary" if is_active else "secondary",
                    width="stretch",
                    key=f"hero_nav_{stage_key}",
                    help=tip_text,
                )
                and not is_active
            ):
                if stage_key == "model_arena":
                    st.session_state.pop("arena_home_view", None)
                    st.session_state.pop("ar_selected", None)
                switch_factory_page(stage_key)
        guide_strip(page)


def main() -> None:
    os.environ["KNOTWORK_NETWORK"] = "deny"  # the studio never touches the network
    if demo_guard.is_public_demo():  # sessions, datasets and results live in the sandbox only
        os.environ["KNOTWORK_ARENA_ROOT"] = str(demo_guard.sandbox_root() / "arena")
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))  # pages import `studio_ui`
    st.set_page_config(page_title="Knotwork VTF Studio", page_icon=":material/hub:", layout="wide")
    pages = [
        st.Page(str(APP / "app_pages" / name), title=title, icon=icon, default=i == 0)
        for i, (_, name, title, icon) in enumerate(PAGES)
    ]
    st.session_state["_knotwork_pages"] = {
        key: page_obj for (key, _, _, _), page_obj in zip(PAGES, pages, strict=True)
    }
    from studio_ui import footer  # importable once APP is on sys.path
    from ui_chrome import working_set

    working_set()  # cold start: seed the shared Working Set before any page reads it
    current = st.navigation(pages, position="hidden")
    header(guide.page_key(current.title))
    current.run()
    footer()


if __name__ == "__main__":
    main()
