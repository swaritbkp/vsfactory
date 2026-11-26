"""Streamlit helpers shared by the studio pages."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import streamlit as st

from knotwork.render.canvas import REPO_ROOT
from knotwork.studio import forensics

OUT = REPO_ROOT / "artifacts" / "studio"


def relative(path: Path) -> str:
    """Repository-relative text for a path (absolute when it lies elsewhere)."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve(text: str) -> Path:
    path = Path(text.strip())
    return path if path.is_absolute() else REPO_ROOT / path


def dataset_select(label: str, key: str, prefer: str | None = None) -> Path | None:
    """One searchable box over discovered datasets that also accepts a typed folder."""
    options = [relative(p) for p in forensics.find_datasets()]
    index = options.index(prefer) if prefer in options else (0 if options else None)
    chosen = st.selectbox(
        label,
        options,
        index=index,
        key=key,
        accept_new_options=True,
        placeholder="Pick a dataset or type a folder",
    )
    if not chosen:
        st.caption("No dataset yet. Build one under Studio, in Batch & OOD Build.")
        return None
    path = resolve(str(chosen))
    if not (path / "index.jsonl").is_file():
        st.warning(f"`{chosen}` has no index.jsonl.", icon=":material/folder_off:")
        return None
    return path


def run_with_status(label: str, action: Callable[[], Any]) -> Any:
    """Run a live action inside a status box; report failures in the page and return None."""
    with st.status(label, expanded=False) as status:
        try:
            result = action()
        except Exception as exc:  # the UI boundary: every failure is shown to the user
            status.update(label=f"{label} failed", state="error", expanded=True)
            st.error(f"{type(exc).__name__}: {exc}")
            return None
        status.update(label=f"{label} done", state="complete")
    return result


def verdict_badge(verdict: str) -> None:
    if verdict == "CERTIFIED":
        st.badge("CERTIFIED", icon=":material/verified:", color="green")
    else:
        st.badge("QUARANTINED", icon=":material/block:", color="red")
