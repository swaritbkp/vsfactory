"""Human Calibration Lab: one blind item at a time, scored live against the benchmarked models."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from knotwork.pipeline.io import read_json, read_jsonl
from knotwork.studio import calibration, factory, forensics, guide, present
from studio_ui import dataset_select, relative

IMAGE_PX = 520
MAX_BUTTON_OPTIONS = 8
EMPTY_ANSWER = "Enter an answer before submitting."


@st.cache_data(ttl=60, show_spinner=False)
def _preferred_dataset() -> str | None:
    """A dataset whose bundles the committed live run also answered, for head-to-head scores."""
    trials = calibration.VLM_TRIALS
    if not trials.is_file():
        return None
    asked = set(calibration.BUNDLE_RE.findall(trials.read_text(encoding="utf-8")))
    for dataset in forensics.find_datasets():
        if any(row["bundle_id"] in asked for row in forensics.bundles(dataset)):
            return relative(dataset)
    return None


def _session_rows(who: str) -> list[dict[str, Any]]:
    return [r for r in read_jsonl(calibration.LOG) if r.get("session") == who]


# ------------------------------------------------------------------ item card
def _question_text(item: dict[str, Any]) -> str:
    return str(item["question"]).split("\nEnd your reply")[0]


def _answer_input(item: dict[str, Any]) -> str:
    key = f"he_reply_{item['bundle_id']}"
    options = item["options"]
    if options and len(options) <= MAX_BUTTON_OPTIONS:
        return str(
            st.segmented_control("Your answer", options, key=key, help=guide.tip("he_answer")) or ""
        )
    if options:
        return str(
            st.selectbox("Your answer", options, index=None, key=key, help=guide.tip("he_answer"))
            or ""
        )
    return st.text_input("Your answer", placeholder="e.g. 42, or: yes; 42", key=key,
                         help=guide.tip("he_answer"))  # fmt: skip


def _picture(item: dict[str, Any], witness: bool) -> None:
    bundle = Path(item["dir"])
    if witness:
        certificate = read_json(bundle / "task.json")["certificate"]
        st.image(factory.witness_overlay(bundle, certificate), width=IMAGE_PX,
                 alt="The item with the kernel's witness evidence drawn on it")  # fmt: skip
        st.caption("Witness overlay: the evidence the kernel used to certify the answer.")
    else:
        st.image(factory.consumption(bundle), width=IMAGE_PX, alt="Task image")


def _verdict(row: dict[str, Any], item: dict[str, Any]) -> None:
    """The graded result for the item shown above it."""
    with st.container(horizontal=True, vertical_alignment="center"):
        if row["correct"]:
            st.badge("Correct", color="green", icon=":material/check_circle:")
        else:
            st.badge("Incorrect", color="red", icon=":material/cancel:")
        st.caption(f"{item['plugin'].title()} · {item['family']}", width="content")
    a, b, c = st.columns(3)
    a.metric("Your answer", row["response"])
    b.metric("Certified answer", row["truth"])
    c.metric("Response time", f"{row['elapsed_s']:.1f} s")


def _show_result(result: dict[str, Any]) -> None:
    """After a submit the item stays on screen and its verdict takes the Submit button's place,
    directly under the picture and question, so it is read without scrolling."""
    item, row = result["item"], result["row"]
    with st.container(horizontal_alignment="center"):
        _picture(item, bool(st.session_state.get("he_witness")))
    st.markdown(_question_text(item))
    with st.container(border=True, key="he_verdict"):
        _verdict(row, item)
    with st.container(horizontal=True, vertical_alignment="center"):
        st.toggle("Show witness overlay", key="he_witness", help=guide.tip("he_witness"))
        next_item = st.button("Next item", type="primary", icon=":material/skip_next:",
                              key="he_next", help=guide.tip("he_next"))  # fmt: skip
    if next_item:
        del st.session_state.last_result
        st.session_state.pop("he_witness", None)
        st.rerun()


def _trial(dataset: Path, who: str) -> None:
    if "last_result" in st.session_state:
        _show_result(st.session_state.last_result)
        return
    answered = calibration.done_ids(who)
    items = calibration.queue(dataset, who)
    left = [i for i in items if i["bundle_id"] not in answered]
    if not left:
        st.success(
            "You have answered every item in this set. Pick another item set under Session "
            "to keep going.",
            icon=":material/task_alt:",
        )
        return
    item = left[0]
    trial = st.session_state.get("trial")
    if trial is None or trial["bundle_id"] != item["bundle_id"]:
        trial = {"bundle_id": item["bundle_id"], "start": time.monotonic()}
        st.session_state.trial = trial
    st.caption(f"Item {len(items) - len(left) + 1} of {len(items)} · answer from the image alone")
    with st.container(horizontal_alignment="center"):
        _picture(item, witness=False)
    st.markdown(_question_text(item))
    reply = _answer_input(item)
    # a plain primary button, styled exactly like Next item (a form submit button flashes darker)
    sent = st.button(
        "Submit",
        type="primary",
        width="stretch",
        icon=":material/send:",
        key="he_submit",
        help=guide.tip("he_submit"),
    )
    if sent:
        if not reply.strip():
            st.warning(EMPTY_ANSWER, icon=":material/edit:")
            return
        elapsed = time.monotonic() - trial["start"]
        row = calibration.record(item, reply.strip(), elapsed, who)
        st.session_state.last_result = {"item": item, "row": row}
        st.rerun()


# ------------------------------------------------------------------ scoreboard
def _scoreboard(who: str) -> None:
    mine = _session_rows(who)
    right = sum(bool(r["correct"]) for r in mine)
    if mine:
        st.metric("Your session", f"{right / len(mine):.0%}", f"{right} of {len(mine)} correct",
                  delta_color="off", delta_arrow="off", help=guide.tip("he_session"))  # fmt: skip
    else:
        st.metric("Your session", "0 answered", "Answer the item on the left and press Submit",
                  delta_color="off", delta_arrow="off", help=guide.tip("he_session"))  # fmt: skip
    frame = baseline_frame()
    if frame.empty:
        st.caption("No model baseline yet: the committed live run is missing.")
        return

    def percent(key: str) -> Any:
        return st.column_config.ProgressColumn(
            format="%.1f%%", min_value=0.0, max_value=100.0, help=guide.tip(key)
        )

    st.dataframe(
        frame, hide_index=True, width="stretch", alt="Model accuracy on the live benchmark",
        column_config={
            "Model": st.column_config.TextColumn(help=guide.tip("col_model")),
            "Accuracy": percent("col_he_accuracy"), "Shared": percent("col_shared"),
            "Items": st.column_config.NumberColumn(help=guide.tip("col_items")),
        },
        height=36 * (len(frame) + 1) + 2,  # every row visible: no inner scroll, no blank space
    )  # fmt: skip
    st.caption(
        "VLM baselines from the committed live benchmark run. A Shared column appears once "
        "people and models have answered the same items."
    )


def baseline_frame() -> pd.DataFrame:
    """The committed live run's model accuracies (percent), best first. Not cached: the Shared
    column changes with every response a person submits."""
    rows = present.scoreboard(calibration.summary(), present.model_names())
    frame = pd.DataFrame([r for r in rows if r["who"] != "You"])
    if frame.empty:
        return frame
    frame = frame.rename(
        columns={"who": "Model", "accuracy": "Accuracy", "n": "Items", "shared": "Shared"}
    )
    frame[["Accuracy", "Shared"]] = frame[["Accuracy", "Shared"]].astype(float) * 100
    if frame["Shared"].isna().all():  # nothing answered by both a person and a model yet
        frame = frame.drop(columns="Shared")
    return frame.sort_values("Accuracy", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ page
if "participant" not in st.session_state:
    st.session_state.participant = f"p-{uuid.uuid4().hex[:6]}"
card, side = st.columns([2.2, 1], gap="large")
with side:
    with st.container(border=True, key="he_scoreboard"):
        st.markdown("**Scoreboard**")
        _scoreboard(str(st.session_state.participant))  # drawn first: above the fold on load
    with st.expander("Session", icon=":material/badge:"):
        participant = st.text_input("Participant id", key="participant",
                                    help=guide.tip("he_participant"))  # fmt: skip
        chosen = dataset_select("Item set", "he_dataset", _preferred_dataset(),
                                help=guide.tip("he_item_set"))  # fmt: skip
        if calibration.LOG.is_file():
            st.download_button(
                "Download responses", calibration.LOG.read_bytes(), "human_calibration.jsonl",
                icon=":material/download:", width="stretch", help=guide.tip("he_download"),
            )  # fmt: skip
with card, st.container(border=True):
    if chosen is None:
        st.info("Open Session on the right and pick an item set to start answering.",
                icon=":material/playlist_add:")  # fmt: skip
    else:
        _trial(chosen, participant)
