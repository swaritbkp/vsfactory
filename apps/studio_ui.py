"""Streamlit helpers shared by the studio pages."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import streamlit as st

from knotwork.render.canvas import REPO_ROOT
from knotwork.studio import forensics

OUT = REPO_ROOT / "artifacts" / "studio"
FOOTER = (
    "© 2026 BilotaAI · Team Bilota (Swarit Sharma & Pankaj David) · Knotwork Visual Task Factory"
)
SOURCES = (  # folder prefix -> where the dataset came from, in the reader's words
    ("artifacts/studio/batch/", "Studio batch"),
    ("artifacts/phase3/bench_split/", "Benchmark split"),
    ("artifacts/selftest/data/", "Self-test"),
    ("artifacts/phase2/", "Phase 2 run"),
    ("artifacts/studio/", "Studio"),
    ("tests/", "Test fixture"),
)


def relative(path: Path) -> str:
    """Repository-relative text for a path (absolute when it lies elsewhere)."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve(text: str) -> Path:
    path = Path(text.strip())
    return path if path.is_absolute() else REPO_ROOT / path


def footer() -> None:
    """The attribution line every page ends with."""
    st.divider()
    st.caption(FOOTER)


@st.cache_data(ttl=30, show_spinner=False)
def _item_count(folder: str) -> int:
    index = resolve(folder) / "index.jsonl"
    if not index.is_file():
        return 0
    with index.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def dataset_label(folder: str) -> str:
    """`Tangle · Benchmark split · 600 items` for a repository-relative dataset folder."""
    source = next((name for prefix, name in SOURCES if folder.startswith(prefix)), "Folder")
    name = folder.rstrip("/").rsplit("/", 1)[-1].replace("data_", "").replace("_", " ")
    count = _item_count(folder)
    items = f"{count} item{'s' if count != 1 else ''}" if count else "not built yet"
    return f"{name[:1].upper()}{name[1:]} · {source} · {items}"


def dataset_select(label: str, key: str, prefer: str | None = None) -> Path | None:
    """One searchable box over discovered datasets that also accepts a typed folder."""
    options = [relative(p) for p in forensics.find_datasets()]
    index = options.index(prefer) if prefer in options else (0 if options else None)
    chosen = st.selectbox(
        label,
        options,
        index=index,
        key=key,
        format_func=dataset_label,
        accept_new_options=True,
        placeholder="Pick a dataset or type a folder path",
    )
    if not chosen:
        st.caption("No dataset yet. Build one in Studio under Batch & OOD Build.")
        return None
    path = resolve(str(chosen))
    if not (path / "index.jsonl").is_file():
        st.warning(
            f"`{chosen}` is not a dataset: it has no index.jsonl. Pick a folder from the list.",
            icon=":material/folder_off:",
        )
        return None
    st.caption(f"Folder: `{relative(path)}`")
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
        status.update(label=f"{label}: done", state="complete")
    return result


def verdict_badge(verdict: str) -> None:
    if verdict == "CERTIFIED":
        st.badge("Certified", icon=":material/verified:", color="green")
    else:
        st.badge("Quarantined", icon=":material/block:", color="red")
