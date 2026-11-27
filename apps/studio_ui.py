"""Streamlit helpers shared by the studio pages."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import streamlit as st

from knotwork.render.canvas import REPO_ROOT
from knotwork.studio import demo_guard, forensics, guide

#: where the Studio writes: the repository's artifacts/, or the sandbox in the public demo
OUT = (
    demo_guard.SANDBOX_ROOT / "studio"
    if demo_guard.is_public_demo()
    else REPO_ROOT / "artifacts" / "studio"
)
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
    """A typed folder as an absolute path. In the public demo it is confined to the sandbox
    (check `demo_guard.path_problem` first: this raises SandboxViolation outside it)."""
    if demo_guard.is_public_demo():
        return demo_guard.confine(text)
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


def dataset_select(
    label: str, key: str, prefer: str | None = None, help: str | None = None
) -> Path | None:
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
        help=help,
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


# ------------------------------------------------------------------ guided mode
GUIDED_KEY = "guided_mode"
TOUR_TITLE = "Knotwork Studio — 60-Second Guided Tour"


def guided() -> bool:
    """Whether the guide strip and tips are shown (on until the user turns it off)."""
    return bool(st.session_state.get(GUIDED_KEY, True))


def _hide_guide() -> None:
    st.session_state[GUIDED_KEY] = False


def _go(page: str) -> None:
    st.switch_page(guide.PAGES[page][1])


@st.dialog(TOUR_TITLE, width="large")
def tour_dialog() -> None:
    """All five tour steps, each with a button that opens its page."""
    st.caption("Five pages, in the order a reviewer usually wants them. Each takes about a minute.")
    for i, step in enumerate(guide.TOUR_STEPS, start=1):
        title = guide.PAGES[step["page"]][0]
        with st.container(border=True):
            text, go = st.columns([5, 1.4], vertical_alignment="center")
            text.markdown(f"**{i}. {title}: {step['goal']}**  \n{step['tip']}")
            if go.button(f"Open {title}", key=f"tour_go_{step['page']}", width="stretch"):
                _go(step["page"])


def guide_controls(page: str | None) -> None:
    """Guided mode toggle, Page guide popover and Quick tour button, for the header row."""
    st.session_state.setdefault(GUIDED_KEY, True)
    with st.container(horizontal=True, width="content", gap="small", vertical_alignment="center"):
        st.toggle("Guided mode", key=GUIDED_KEY, help=guide.tip("guided_mode"))
        if page is not None:
            info = guide.PAGE_GUIDES[page]
            with st.popover("Page guide", icon=":material/help:", help=guide.tip("page_guide")):
                st.markdown(f"**{info['title']}**: {info['summary']}")
                st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(info["steps"], 1)))
                st.caption(f"Try first: {info['try_first']}")
                st.markdown("**Terms**")
                terms = "\n".join(f"- **{t}**: {guide.GLOSSARY[t]}" for t in info["terms"])
                st.markdown(terms)
        if st.button("Quick tour", icon=":material/route:", type="tertiary", key="tour_open",
                     help=guide.tip("quick_tour")):  # fmt: skip
            tour_dialog()


def guide_strip(page: str | None) -> None:
    """One slim bordered strip: the tour step, a tip, Next step and Hide guide. Guided mode only."""
    if page is None or not guided():
        return
    i = guide.tour_index(page)
    step = guide.TOUR_STEPS[i]
    following = guide.next_page(page)
    with st.container(border=True, key="guide_strip"):
        text, actions = st.columns([3, 1.3], vertical_alignment="center")
        text.markdown(
            f"**Step {i + 1} of {len(guide.TOUR_STEPS)} · {step['goal']}**  \n:gray[{step['tip']}]"
        )
        with actions, st.container(horizontal=True, horizontal_alignment="right", gap="small"):
            st.button("Hide guide", type="tertiary", key="guide_hide", on_click=_hide_guide,
                      help=guide.tip("hide_guide"))  # fmt: skip
            target = following or guide.TOUR_STEPS[0]["page"]
            label = f"Next step: {guide.PAGES[target][0]} →" if following else "Back to Studio"
            if st.button(label, key="guide_next", help=guide.tip("next_step")):
                _go(target)


def review_controls(log: Path, item_id: str, domain: str, seed: int | None, key: str) -> str:
    """Human QA triage for one item: its current status badge, an optional one-line note and
    Approve / Flag buttons, appended to `log` (review_log.jsonl). Returns the current status."""
    from knotwork.studio import review

    status = review.status_of(review.load_review_states(log), item_id)
    text, color, icon = review.BADGES[status]
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.badge(text, color=color, icon=icon)  # type: ignore[arg-type]
        note = st.text_input("Review note", key=f"qa_note_{key}", placeholder="Optional note",
                             label_visibility="collapsed", max_chars=200,
                             help=guide.tip("qa_note"))  # fmt: skip
        clicked = None
        if st.button("Approve", icon=":material/verified:", key=f"qa_ok_{key}",
                     disabled=status == "approved", help=guide.tip("qa_approve")):  # fmt: skip
            clicked = "approved"
        if st.button("Flag", icon=":material/flag:", key=f"qa_flag_{key}",
                     disabled=status == "flagged", help=guide.tip("qa_flag")):  # fmt: skip
            clicked = "flagged"
    if clicked:
        review.record_review(log, item_id, clicked, domain, seed, note=str(note or ""))
        st.toast(f"{review.BADGES[clicked][0]}: {item_id}", icon=review.BADGES[clicked][2])
        st.rerun()
    return status
