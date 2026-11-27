"""Shared visual grammar of the five Studio pages: the Working Set (bootstrap, status pill, QA
review with tags) and the small status strips every page reuses. Layout only; the logic lives in
`knotwork.studio.working_set` and `knotwork.studio.review`."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from knotwork.studio import guide, review
from knotwork.studio.working_set import LABEL, WorkingSet

QA_CHOICES = {"Approve": "approved", "Flag": "flagged"}
QA_ICONS = {"Approve": ":material/verified:", "Flag": ":material/flag:"}
WS_ICON = ":material/stacks:"


@st.cache_resource(show_spinner=False)
def _bootstrap(root: str) -> bool:
    """Seed the Working Set from the committed starter once per server and folder."""
    return WorkingSet(Path(root)).ensure()


def working_set() -> WorkingSet:
    """The active Working Set, seeded on first use (a file copy, well under a second)."""
    ws = WorkingSet()
    _bootstrap(str(ws.root))
    if not ws.valid():  # the folder was removed while the server ran
        ws.ensure()
    return ws


def is_working_set(dataset: Path | None) -> bool:
    if dataset is None:
        return False
    try:
        return Path(dataset).resolve() == working_set().root.resolve()
    except OSError:
        return False


def review_log_for(dataset: Path | None) -> Path:
    """The QA log that belongs to a dataset: the Working Set's own, else the Studio log."""
    return working_set().review_log if is_working_set(dataset) else review.STUDIO_LOG


def ws_pill(ws: WorkingSet | None = None) -> None:
    """`Working Set: 5 certified · 2 approved`, the same badge on every page."""
    ws = ws or working_set()
    qa = ws.qa()
    total = sum(qa.values())
    st.badge(f"{LABEL}: {total} certified · {qa['approved']} approved", icon=WS_ICON,
             color="green" if total else "gray", help=guide.tip("ws_pill"))  # fmt: skip


def qa_strip(ws: WorkingSet, target: str | None = "export") -> None:
    """`Working Set QA: 2 approved · 1 flagged · 3 pending` with a jump to the next page."""
    qa = ws.qa()
    with st.container(horizontal=True, vertical_alignment="center", gap="small",
                      key="qa_strip"):  # fmt: skip
        st.badge(f"{qa['approved']} approved", icon=":material/verified:", color="green")
        st.badge(f"{qa['flagged']} flagged", icon=":material/flag:", color="red")
        st.badge(f"{qa['unreviewed']} pending", icon=":material/pending:", color="gray")
        if target is not None:
            title, page = guide.PAGES[target]
            st.page_link(page, label=f"Open {title}", icon=":material/arrow_forward:",
                         help=guide.tip("qa_jump"))  # fmt: skip


def _choice_of(status: str) -> str | None:
    return next((k for k, v in QA_CHOICES.items() if v == status), None)


def qa_review(ws: WorkingSet, item_id: str, key: str, bundle: Path | None = None,
              seed: int | None = None, attempts: int = 1) -> None:  # fmt: skip
    """Approve / Flag, tag pills and an optional note for one item, saved to the Working Set's
    append-only review log. With `bundle`, saving a review of an item that is not in the set yet
    adds it first. Runs as a fragment: saving redraws only this card."""
    st.fragment(_qa_review_body)(ws.root, item_id, key, bundle, seed, attempts)


def _qa_review_body(root: Path, item_id: str, key: str, bundle: Path | None, seed: int | None,
                    attempts: int) -> None:  # fmt: skip
    ws = WorkingSet(root)
    states = review.load_review_states(ws.review_log)
    status = review.status_of(states, item_id)
    saved_tags = review.tags_of(states, item_id)
    text, color, icon = review.BADGES[status]
    in_set = item_id in ws.ids()
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown("**Human QA**", width="content")
        st.badge(text, color=color, icon=icon)  # type: ignore[arg-type]
        if saved_tags:
            st.caption(" · ".join(saved_tags), width="content")
    if not in_set and bundle is None:
        st.caption("Add this item to the Working Set to review and tag it.")
        return
    left, right = st.columns([1, 2.4], gap="medium")
    with left:
        decision = st.segmented_control(
            "Decision", list(QA_CHOICES), default=_choice_of(status), key=f"qa_dec_{key}",
            help=guide.tip("qa_decision"),
        )  # fmt: skip
    with right:
        options = list(dict.fromkeys([*review.TAGS, *saved_tags]))
        st.pills("Tags", options, selection_mode="multi", default=saved_tags,
                 key=f"qa_tags_{key}", help=guide.tip("qa_tags"))  # fmt: skip
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        with st.popover("Note", icon=":material/edit_note:", help=guide.tip("qa_note")):
            st.text_input("Review note", key=f"qa_note_{key}", max_chars=200,
                          placeholder="Optional, one line")  # fmt: skip
        st.button("Save review", icon=":material/check:", key=f"qa_save_{key}",
                  disabled=decision is None, help=guide.tip("qa_save"), on_click=_save,
                  args=(root, item_id, key, bundle, seed, attempts))  # fmt: skip
    if st.session_state.pop(f"qa_added_{key}", False):
        st.rerun()  # the item joined the set: redraw the page so the header pill counts it


def _save(root: Path, item_id: str, key: str, bundle: Path | None, seed: int | None,
          attempts: int) -> None:  # fmt: skip
    """Save callback: runs before the redraw, so the card shows the new decision at once."""
    ss = st.session_state
    decision = ss.get(f"qa_dec_{key}")
    if decision is None:
        return
    ws = WorkingSet(root)
    chosen = QA_CHOICES[str(decision)]
    if item_id not in ws.ids() and bundle is not None:
        ss[f"qa_added_{key}"] = ws.add(bundle, "studio", seed, attempts).added
    ws.review(item_id, chosen, list(ss.get(f"qa_tags_{key}") or []),
              str(ss.get(f"qa_note_{key}") or ""))  # fmt: skip
    st.toast(f"{review.BADGES[chosen][0]}: {item_id}", icon=review.BADGES[chosen][2])


__all__ = ["is_working_set", "qa_review", "qa_strip", "review_log_for", "working_set", "ws_pill"]
