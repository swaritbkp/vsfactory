"""Model Arena's Post-Training Pack tab: frontier classification, a live reward verifier sandbox and
the one-click SFT / DPO / GRPO pack export. Layout only; the logic is `knotwork.posttrain`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from knotwork.arena import storage
from knotwork.arena.schemas import ItemRecord, SessionConfig
from knotwork.posttrain import curriculum, vcot
from knotwork.posttrain.reward import is_full_image, parse_boxes, score_rollout
from knotwork.studio import factory, guide, review
from studio_ui import relative, run_with_status

EXAMPLES = ("Grounded vCoT", "Lucky guess (no boxes)", "Wrong answer")
#: the always-on sandbox's one-click presets (Model Arena home, no campaign needed)
PRESETS = ("Grounded trace", "Ungrounded lucky guess", "Full-image box cheat [0,0,1000,1000]")
STYLE = {
    "Grounded vCoT": "grounded", "Grounded trace": "grounded",
    "Lucky guess (no boxes)": "lucky", "Ungrounded lucky guess": "lucky",
    "Wrong answer": "wrong", PRESETS[2]: "cheat",
}  # fmt: skip
CHEAT_BOX = "[0, 0, 1000, 1000]"
REGIME_NOTE = {
    "frontier": "GRPO Frontier",
    "cold_start": "SFT Cold-Start",
    "saturated": "Saturated",
}
IMAGE_PX = 420


def _stamp(entry_dir: Path) -> float:
    path = entry_dir / storage.RESULTS
    return path.stat().st_mtime if path.is_file() else 0.0


@st.cache_data(show_spinner=False, max_entries=16)
def _spec(results_dir: str, _stamp: float) -> dict[str, Any]:
    return curriculum.analyze(Path(results_dir))


@st.cache_data(show_spinner=False, max_entries=64)
def _trace(bundle: str) -> dict[str, Any] | None:
    trace = vcot.try_extract(Path(bundle))
    return None if trace is None else trace.model_dump()


def _wrong(item: ItemRecord) -> str:
    others = [o for o in item.options or [] if o != item.truth]
    if others:
        return others[0]
    try:
        return f"{float(item.truth) + 7:g}"
    except ValueError:
        return "yes; 0" if item.answer.get("type") == "premise" else "unknown"


def _example(kind: str, item: ItemRecord, trace: dict[str, Any]) -> str:
    style = STYLE.get(kind, "grounded")
    answer = str(trace["final_answer"])
    if style == "lucky":
        return f"<think>\nIt looks like {answer}.\n</think>\n\\boxed{{{answer}}}"
    if style == "cheat":
        return (f"<think>\nStep 1 (perception): the whole board <box>{CHEAT_BOX}</box>\n"
                f"</think>\n\\boxed{{{answer}}}")  # fmt: skip
    text = str(trace["formatted_cot"])
    if style == "wrong":
        return text.replace(f"\\boxed{{{answer}}}", f"\\boxed{{{_wrong(item)}}}")
    return text


def _frontier(spec: dict[str, Any]) -> None:
    cells = [c for c in spec["cells"] if c["n"]]
    counts = {k: sum(c["regime"] == k for c in cells) for k in REGIME_NOTE}
    seeds = spec["seeds"]
    a, b, c, d = st.columns(4)
    a.metric("GRPO frontier", counts["frontier"], border=True, help=guide.tip("pt_frontier"))
    b.metric("SFT cold-start", counts["cold_start"], border=True, help=guide.tip("pt_cold"))
    c.metric("Saturated", counts["saturated"], border=True, help=guide.tip("pt_saturated"))
    d.metric("Train seeds from", f"{seeds['train_seed_start']:,}", border=True,
             help=guide.tip("pt_seeds"))  # fmt: skip
    rows = [{"Block": x["block"], "Model": x["model"], "Accuracy": 100 * x["accuracy"],
             "Trials": x["n"], "Regime": x["badge"], "Mode": x["mode"] or "—",
             "Weight": 100 * x["weight"]} for x in cells]  # fmt: skip
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Accuracy": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100,
                                                    help=guide.tip("pt_accuracy")),
        "Regime": st.column_config.TextColumn(help=guide.tip("pt_regime")),
        "Mode": st.column_config.TextColumn(help=guide.tip("pt_mode")),
        "Weight": st.column_config.NumberColumn(format="%.1f%%", help=guide.tip("pt_weight")),
    })  # fmt: skip
    ok = bool(seeds["disjoint"])
    st.badge(
        f"Seed-disjoint: eval ≤ {seeds['max_eval_seed']:,}, train "
        f"{seeds['train_seed_start']:,}–{seeds['train_seed_end']:,}, fresh OOD eval from "
        f"{seeds['ood_eval_seed_start']:,}",
        color="green" if ok else "red",
        icon=":material/verified_user:" if ok else ":material/gpp_bad:",
    )


def _sandbox(cfg: SessionConfig) -> None:
    data = storage.resolve(cfg.data_dir)
    items = sorted((i for i in storage.dataset_items(data) if i.certified),
                   key=lambda i: (i.block_id, i.item_index))  # fmt: skip
    st.markdown("**Reward verifier sandbox**")
    st.caption("Score any rollout against a certified item with the same reward a trainer would "
               "call. Nothing leaves this machine.")  # fmt: skip
    if not items:
        st.caption("No certified item yet. Run the session first.")
        return
    reward_sandbox(items, data, "pt", EXAMPLES)


def reward_sandbox(items: list[ItemRecord], data: Path, key: str,
                   examples: tuple[str, ...] = PRESETS) -> None:  # fmt: skip
    """Item picker, example rollouts and the live reward breakdown. A fragment: picking an item or
    a preset, or editing the rollout, redraws only the sandbox."""
    st.fragment(_sandbox_body)(items, data, key, examples)


def _item_label(i: ItemRecord) -> str:
    if i.preset == "Studio batch":
        return f"{i.domain.replace('_', ' ').title()} · {i.family} · truth {i.truth}"
    return f"{i.block_id} #{i.item_index:03d} · {i.family} · truth {i.truth}"


def _sandbox_body(items: list[ItemRecord], data: Path, key: str,
                  examples: tuple[str, ...]) -> None:  # fmt: skip
    left, right = st.columns([1, 1.35], gap="medium")
    with left:
        item = st.selectbox("Item", items, key=f"{key}_item", help=guide.tip("pt_item"),
                            format_func=_item_label)  # fmt: skip
        trace = _trace(str(data / item.dir))
        if trace is None:
            st.caption("This item has no grounded trace, so only outcome and format are scored.")
            st.image(factory.beauty(data / item.dir), width=IMAGE_PX)
        else:
            st.image(vcot.overlay(factory.beauty(data / item.dir),
                                  vcot.VCoTTrace.model_validate(trace)), width=IMAGE_PX,
                     alt="Item with its grounded vCoT boxes")  # fmt: skip
    with right:
        tip = "sb_preset" if examples == PRESETS else "pt_example"
        kind = st.segmented_control("Example rollout", examples, default=examples[0],
                                    key=f"{key}_example_{item.bundle_id}", required=True,
                                    help=guide.tip(tip))  # fmt: skip
        start = _example(str(kind), item, trace) if trace else f"\\boxed{{{item.truth}}}"
        text = st.text_area("Rollout", start, height=220,
                            key=f"{key}_roll_{item.bundle_id}_{kind}",
                            help=guide.tip("pt_rollout"))  # fmt: skip
        extra = {"domain": item.domain, "vcot_boxes_1000": [s["bbox_norm_1000"] for s in
                                                            trace["steps"]] if trace else []}  # fmt: skip
        r = score_rollout(text, json.dumps(item.answer), extra)
        a, b, c, d = st.columns(4)
        a.metric("Composite", f"{r['score']:.3f}", help=guide.tip("pt_score"))
        b.metric("Outcome", f"{r['outcome_score']:.0f}", help=guide.tip("pt_outcome"))
        c.metric("Grounding IoU", f"{r['grounding_iou']:.2f}", help=guide.tip("pt_iou"))
        d.metric("Format", f"{r['format_score']:.1f}", help=guide.tip("pt_format"))
        if r["reward_hack_flag"]:
            st.badge("Reward-hack flag: right answer, no box matched (IoU ≥ 0.5) · gated to "
                     f"≤ {r['weights']['format']:.2f}", color="red",
                     icon=":material/report:")  # fmt: skip
            if any(is_full_image(b) for b in parse_boxes(text)):
                st.caption("A box covering at least 85% of the canvas is never matched, so "
                           "boxing the whole image earns no grounding credit.")  # fmt: skip
        else:
            st.badge(f"{r['matched_boxes']} of {r['ref_boxes']} certified boxes matched "
                     f"(IoU ≥ 0.5) · abstention {r['abstention_score']:.0f}",
                     color="green" if r["outcome_score"] else "orange",
                     icon=":material/rule:")  # fmt: skip


def _export(cfg: SessionConfig, entry_dir: Path) -> None:
    st.markdown("**Post-training pack**")
    st.caption("SFT traces for failed and cold-start items, DPO pairs from real failures, a verl "
               "GRPO parquet, the curriculum spec, the RLVR kit and a provenance attestation.")  # fmt: skip
    only = st.toggle("Approved items only", key=f"pt_approved_{cfg.slug}",
                     help=guide.tip("qa_approved_only"))  # fmt: skip
    blocked = False
    if only:
        states = review.load_review_states(entry_dir / review.LOG_NAME)
        ids = [i.bundle_id for i in storage.dataset_items(storage.resolve(cfg.data_dir))
               if i.certified and i.block_id in {b.block_id for b in cfg.blocks}]  # fmt: skip
        approved = review.counts(states, ids)["approved"]
        if approved:
            st.caption(review.filter_note(approved, len(ids)) + ": only these go into the pack.")
        else:
            blocked = True
            st.warning(f"{review.filter_note(0, len(ids))}. Approve items in the Item inspector, "
                       "or turn this off to export every item.", icon=":material/rule:")  # fmt: skip
    if st.button("Export post-training pack", type="primary", icon=":material/package_2:",
                 key="pt_export", disabled=blocked, help=guide.tip("pt_export")):  # fmt: skip
        done = run_with_status("Export the post-training pack",
                               lambda: curriculum.export_pack(entry_dir, approved_only=only))  # fmt: skip
        if done:
            st.session_state[f"pt_done_{cfg.slug}"] = {k: str(v) for k, v in done.items()}
    done = st.session_state.get(f"pt_done_{cfg.slug}")
    archive = entry_dir / f"{curriculum.PACK_DIR}.zip"
    if not archive.is_file():
        st.caption("No pack yet. Press Export post-training pack to write it.")
        return
    if done:
        a, b, c = st.columns(3)
        a.metric("SFT traces", done["sft"], help=guide.tip("pt_sft"))
        b.metric("DPO pairs", done["dpo"], help=guide.tip("pt_dpo"))
        c.metric("GRPO prompts", done["grpo"], help=guide.tip("pt_grpo"))
    st.success(f"Pack written to `{relative(entry_dir / curriculum.PACK_DIR)}`.",
               icon=":material/check_circle:")  # fmt: skip
    st.download_button("Download posttrain_pack.zip", archive.read_bytes(), archive.name,
                       key="pt_download", icon=":material/download:",
                       help=guide.tip("pt_download"))  # fmt: skip
    att = entry_dir / curriculum.PACK_DIR / "provenance_attestation.json"
    if att.is_file():
        with st.expander("Clean-IP provenance attestation", icon=":material/verified_user:"):
            st.caption(guide.tip("pt_provenance"))
            body = json.loads(att.read_text(encoding="utf-8"))
            body.pop("files_sha256", None)
            st.json(body, expanded=1)


def posttrain_tab(cfg: SessionConfig, entry_dir: Path) -> None:
    st.caption(guide.tip("pt_pack"))
    spec = _spec(str(entry_dir), _stamp(entry_dir))
    if not any(c["n"] for c in spec["cells"]):
        st.info("No scored trial yet. The frontier table fills in as models answer.",
                icon=":material/insights:")  # fmt: skip
    else:
        _frontier(spec)
    with st.container(border=True):
        _sandbox(cfg)
    with st.container(border=True):
        _export(cfg, entry_dir)
