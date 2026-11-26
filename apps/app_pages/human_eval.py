"""Human Calibration Lab: one blind item at a time, with a running scoreboard against the models."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from knotwork.studio import calibration, factory, forensics, present
from studio_ui import dataset_select, relative

IMAGE_PX = 500
MAX_BUTTON_OPTIONS = 8


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


def _answer_input(item: dict[str, Any]) -> str:
    key = f"he_reply_{item['bundle_id']}"
    options = item["options"]
    if options and len(options) <= MAX_BUTTON_OPTIONS:
        return str(st.segmented_control("Your answer", options, key=key) or "")
    if options:
        return str(st.selectbox("Your answer", options, index=None, key=key) or "")
    return st.text_input("Your answer", placeholder="e.g. 42, or: yes; 42", key=key)


def _result(row: dict[str, Any]) -> None:
    if row["correct"]:
        st.badge("Correct", color="green", icon=":material/check_circle:")
    else:
        st.badge("Incorrect", color="red", icon=":material/cancel:")
    a, b, c = st.columns(3)
    a.metric("Your answer", row["response"])
    b.metric("Certified answer", row["truth"])
    c.metric("Time", f"{row['elapsed_s']:.1f} s")
    if st.button("Next item", type="primary", width="stretch", key="he_next"):
        del st.session_state.last_result
        st.rerun()


def _trial(dataset: Path, who: str) -> None:
    answered = calibration.done_ids(who)
    items = calibration.queue(dataset, who)
    left = [i for i in items if i["bundle_id"] not in answered]
    if "last_result" in st.session_state:
        _result(st.session_state.last_result)
        return
    if not left:
        st.success("Every item in this dataset is answered.", icon=":material/task_alt:")
        return
    item = left[0]
    trial = st.session_state.get("trial")
    if trial is None or trial["bundle_id"] != item["bundle_id"]:
        trial = {"bundle_id": item["bundle_id"], "start": time.monotonic()}
        st.session_state.trial = trial
    st.caption(f"Item {len(items) - len(left) + 1} of {len(items)}")
    with st.container(horizontal_alignment="center"):
        st.image(factory.consumption(Path(item["dir"])), width=IMAGE_PX, alt="Task image")
    st.markdown(item["question"].split("\nEnd your reply")[0])
    with st.form("he_form", border=False):
        reply = _answer_input(item)
        sent = st.form_submit_button("Submit", type="primary", width="stretch")
    if sent:
        if not reply.strip():
            st.warning("Enter an answer first.", icon=":material/edit:")
            return
        elapsed = time.monotonic() - trial["start"]
        st.session_state.last_result = calibration.record(item, reply.strip(), elapsed, who)
        st.rerun()


def _scoreboard() -> None:
    rows = present.scoreboard(calibration.summary(), present.model_names())
    frame = pd.DataFrame(rows).rename(
        columns={"who": "Solver", "accuracy": "Accuracy", "n": "Items", "shared": "Shared"}
    )
    if frame["Shared"].isna().all():  # nothing answered by both you and a model yet
        frame = frame.drop(columns="Shared")
    percent = st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0)
    st.dataframe(
        frame, hide_index=True, width="stretch", alt="Human versus model accuracy",
        column_config={"Accuracy": percent, "Shared": percent},
    )  # fmt: skip
    st.caption(
        "Model accuracy comes from the committed live run over its own items. "
        "A “Shared” column appears once you and a model have answered the same bundle."
    )


if "participant" not in st.session_state:
    st.session_state.participant = f"p-{uuid.uuid4().hex[:6]}"
card, side = st.columns([2.2, 1], gap="large")
with side:
    with st.container(border=True):
        st.markdown("**Scoreboard**")
        board = st.empty()
    with st.expander("Session", icon=":material/badge:"):
        participant = st.text_input("Participant id", key="participant")
        chosen = dataset_select("Item set", "he_dataset", _preferred_dataset())
        if calibration.LOG.is_file():
            st.download_button(
                "Download responses", calibration.LOG.read_bytes(), "human_calibration.jsonl",
                icon=":material/download:", width="stretch",
            )  # fmt: skip
with card, st.container(border=True):
    if chosen is not None:
        _trial(chosen, participant)
with board.container():
    _scoreboard()
