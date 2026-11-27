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
PAGES = (
    ("studio.py", "Studio", ":material/science:"),
    ("forensics.py", "Forensics", ":material/troubleshoot:"),
    ("human_eval.py", "Human Eval", ":material/person_check:"),
    ("export.py", "Export", ":material/package_2:"),
    ("arena.py", "Model Arena", ":material/leaderboard:"),
)


@st.cache_data(ttl=20, show_spinner=False)
def _status() -> present.RepoStatus:
    return present.repo_status()


def header(page: str | None) -> None:
    from studio_ui import guide_controls, guide_strip  # importable once APP is on sys.path

    status = _status()
    with st.container(
        horizontal=True, vertical_alignment="center", horizontal_alignment="distribute"
    ):
        st.markdown("#### :material/hub: Knotwork VTF v2", width="content")
        with st.container(horizontal=True, width="content", gap="small"):
            local = " + local changes" if status["dirty"] else ""
            st.badge(
                f"{status['branch']} @ {status['sha']}{local}",
                icon=":material/commit:",
                color="gray",
            )
            st.badge(
                f"kernel {status['kernel']}",
                icon=":material/fingerprint:",
                color="gray",
                help="First 12 characters of the trusted kernel's source SHA-256.",
            )
            if status["tcb_ok"]:
                st.badge(
                    "Certified offline",
                    icon=":material/verified_user:",
                    color="green",
                    help="The kernel passes its TCB audit (allow-listed imports, no network code) "
                    "and the studio runs with the network policy set to deny.",
                )
            else:
                st.badge("TCB audit failed", icon=":material/gpp_bad:", color="red")
            guide_controls(page)
    if demo_guard.is_public_demo():
        st.info(demo_guard.BANNER, icon=":material/science:")
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
        for i, (name, title, icon) in enumerate(PAGES)
    ]
    from studio_ui import footer  # importable once APP is on sys.path

    current = st.navigation(pages, position="top")
    header(guide.page_key(current.title))
    current.run()
    footer()


if __name__ == "__main__":
    main()
