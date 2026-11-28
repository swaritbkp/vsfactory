"""Shared visual grammar of the five Studio pages: the Working Set (bootstrap, status pill, QA
review with tags) and the small status strips every page reuses. Layout only; the logic lives in
`knotwork.studio.working_set` and `knotwork.studio.review`."""

from __future__ import annotations

import contextlib
import shutil
import uuid
from pathlib import Path

import streamlit as st

from knotwork.studio import demo_guard, guide, review
from knotwork.studio.working_set import LABEL, WorkingSet

QA_CHOICES = {"Approve": "approved", "Flag": "flagged"}
QA_ICONS = {"Approve": ":material/verified:", "Flag": ":material/flag:"}
WS_ICON = ":material/stacks:"
_KEEP_SESSION_KEYS = frozenset(
    {"_knotwork_demo_sid", "_knotwork_demo_pruned", "_knotwork_pages", "guided_mode"}
)


def sync_demo_session() -> str | None:
    """Bind the ContextVar demo session id to this Streamlit browser session."""
    if not demo_guard.is_public_demo():
        demo_guard.set_demo_session_id(None)
        return None
    ss = st.session_state
    sid = ss.get("_knotwork_demo_sid")
    if not isinstance(sid, str) or not demo_guard.SESSION_ID_RE.match(sid):
        sid = f"s_{uuid.uuid4().hex[:8]}"
        ss["_knotwork_demo_sid"] = sid
        ss["_knotwork_demo_pruned"] = False
    demo_guard.set_demo_session_id(sid)
    if not ss.get("_knotwork_demo_pruned"):
        demo_guard.prune_old_sessions(max_sessions=15, keep_sid=sid)
        ss["_knotwork_demo_pruned"] = True
    return sid


def reset_demo_sandbox() -> None:
    """Rotate to a fresh isolated sandbox directory and clear visitor session state."""
    if not demo_guard.is_public_demo():
        return
    ss = st.session_state
    procs: list[object] = []
    batch = ss.get("batch")
    if isinstance(batch, dict):
        procs.append(batch.get("process"))
    ar_procs = ss.get("ar_procs")
    if isinstance(ar_procs, dict):
        procs.extend(ar_procs.values())
    for proc in procs:
        if proc is not None and hasattr(proc, "poll") and proc.poll() is None:
            with contextlib.suppress(OSError):
                proc.terminate()
    old_sid = ss.get("_knotwork_demo_sid")
    if isinstance(old_sid, str) and demo_guard.SESSION_ID_RE.match(old_sid):
        shutil.rmtree(demo_guard.base_sandbox_root() / old_sid, ignore_errors=True)
    new_sid = f"s_{uuid.uuid4().hex[:8]}"
    ss["_knotwork_demo_sid"] = new_sid
    ss["_knotwork_demo_pruned"] = False
    for key in [k for k in ss if k not in _KEEP_SESSION_KEYS]:
        del ss[key]
    sync_demo_session()


@st.cache_resource(show_spinner=False, max_entries=64)
def _bootstrap(root: str) -> bool:
    """Seed the Working Set from the committed starter once per server and folder."""
    return WorkingSet(Path(root)).ensure()


def working_set() -> WorkingSet:
    """The active Working Set, seeded on first use (a file copy, well under a second)."""
    sync_demo_session()
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
    return working_set().review_log if is_working_set(dataset) else review.active_studio_log()


def switch_factory_page(page_key: str) -> None:
    """Switch to a canonical factory page via the hidden navigation registry, or record the
    requested page when running an isolated single-page AppTest outside `inspector.py`."""
    canonical = "model_arena" if page_key == "arena" else page_key
    pages = st.session_state.get("_knotwork_pages") or {}
    page_obj = pages.get(canonical) or pages.get(page_key)
    if page_obj is not None:
        st.switch_page(page_obj)
    else:
        st.session_state["_knotwork_requested_page"] = canonical


def ws_pill(ws: WorkingSet | None = None) -> None:
    """`Working Set: 5 certified · 2 approved` (omitting approved count when 0)."""
    ws = ws or working_set()
    qa = ws.qa()
    total = sum(qa.values())
    approved = f" · {qa['approved']} approved" if qa["approved"] > 0 else ""
    st.badge(
        f"{LABEL}: {total} certified{approved}",
        icon=":material/inventory_2:",
        color="green" if total else "gray",
        help=guide.tip("ws_pill"),
    )


def qa_strip(ws: WorkingSet, target: str | None = "export") -> None:
    """`Working Set QA: 2 approved · 1 flagged · 3 pending` with a jump to the next page."""
    qa = ws.qa()
    with st.container(horizontal=True, vertical_alignment="center", gap="small",
                      key="qa_strip"):  # fmt: skip
        st.badge(f"{qa['approved']} approved", icon=":material/verified:", color="green")
        st.badge(f"{qa['flagged']} flagged", icon=":material/flag:", color="red")
        st.badge(f"{qa['unreviewed']} pending", icon=":material/pending:", color="gray")
        if target is not None:
            title, _ = guide.PAGES[target]
            if st.button(f"Open {title}", icon=":material/arrow_forward:", type="tertiary",
                         key=f"qa_jump_{target}", help=guide.tip("qa_jump")):  # fmt: skip
                switch_factory_page(target)


def _choice_of(status: str) -> str | None:
    return next((k for k, v in QA_CHOICES.items() if v == status), None)


def qa_review(ws: WorkingSet, item_id: str, key: str, bundle: Path | None = None,
              seed: int | None = None, attempts: int = 1) -> None:  # fmt: skip
    """Approve / Flag, tag pills and an optional note for one item, auto-saved to the Working
    Set's append-only review log. With `bundle`, saving a review of an item that is not in the set
    yet adds it first. Runs as a fragment: saving redraws only this card."""
    st.fragment(_qa_review_body)(ws.root, item_id, key, bundle, seed, attempts)


def _qa_review_body(root: Path, item_id: str, key: str, bundle: Path | None, seed: int | None,
                    attempts: int) -> None:  # fmt: skip
    sync_demo_session()
    ss = st.session_state
    pending_toast = ss.pop(f"_qa_toast_{key}", None)
    if pending_toast is not None:
        st.toast(pending_toast[0], icon=pending_toast[1])
    ws = WorkingSet(root)
    states = review.load_review_states(ws.review_log)
    rec = states.get(item_id)
    status = rec.status if rec else "unreviewed"
    saved_tags = list(rec.tags) if rec else []
    saved_note = rec.note if rec else ""
    text, color, icon = review.BADGES[status]
    in_set = item_id in ws.ids()
    dec_key = f"studio_qa_decision_{key}"
    tags_key = f"studio_qa_tags_{key}"
    note_key = f"studio_qa_note_{key}"
    ss.setdefault(dec_key, _choice_of(status))
    ss.setdefault(tags_key, list(saved_tags))
    ss.setdefault(note_key, saved_note)
    with st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="distribute",
                      gap="small"):  # fmt: skip
        with st.container(horizontal=True, vertical_alignment="center", gap="small",
                          width="content"):  # fmt: skip
            st.markdown("**Human QA**", width="content")
            st.badge(text, color=color, icon=icon)  # type: ignore[arg-type]
        if in_set or bundle is not None:
            with st.popover("Note", icon=":material/edit_note:", help=guide.tip("qa_note")):
                st.text_input("Review note", key=note_key, max_chars=200,
                              placeholder="Optional, one line")  # fmt: skip
                st.button("Save note", type="primary", icon=":material/check:",
                          key=f"studio_qa_save_note_{key}", width="stretch", on_click=_save,
                          args=(root, item_id, key, bundle, seed, attempts))  # fmt: skip
    if not in_set and bundle is None:
        st.caption("Add this item to the Working Set to review and tag it.")
        return
    args = (root, item_id, key, bundle, seed, attempts)
    st.segmented_control(
        "Decision", list(QA_CHOICES), key=dec_key, label_visibility="collapsed",
        width="stretch", on_change=_save, args=args, help=guide.tip("qa_decision"),
    )  # fmt: skip
    options = list(dict.fromkeys([*review.TAGS, *saved_tags]))
    st.pills("Tags", options, selection_mode="multi", key=tags_key, label_visibility="collapsed",
             on_change=_save, args=args, help=guide.tip("qa_tags"))  # fmt: skip
    if ss.pop(f"qa_added_{key}", False):
        st.rerun()  # the item joined the set: redraw the page so the header pill counts it


def _save(root: Path, item_id: str, key: str, bundle: Path | None, seed: int | None,
          attempts: int) -> None:  # fmt: skip
    """Auto-save callback: runs before redraw so status, tags and note persist immediately."""
    sync_demo_session()
    ss = st.session_state
    decision = ss.get(f"studio_qa_decision_{key}")
    chosen = QA_CHOICES[str(decision)] if decision in QA_CHOICES else "unreviewed"
    ws = WorkingSet(root)
    prev = review.status_of(review.load_review_states(ws.review_log), item_id)
    if item_id not in ws.ids() and bundle is not None:
        ss[f"qa_added_{key}"] = ws.add(bundle, "studio", seed, attempts).added
    ws.review(item_id, chosen, list(ss.get(f"studio_qa_tags_{key}") or []),
              str(ss.get(f"studio_qa_note_{key}") or ""))  # fmt: skip
    if chosen == "unreviewed" and prev != "unreviewed":
        ss[f"_qa_toast_{key}"] = ("QA cleared · Unreviewed", review.BADGES[chosen][2])
    else:
        ss[f"_qa_toast_{key}"] = (
            f"QA saved · {review.BADGES[chosen][0]}",
            review.BADGES[chosen][2],
        )


__all__ = [
    "is_working_set",
    "qa_review",
    "qa_strip",
    "reset_demo_sandbox",
    "review_log_for",
    "switch_factory_page",
    "sync_demo_session",
    "working_set",
    "ws_pill",
]
