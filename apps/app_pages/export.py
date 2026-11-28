"""Export: three steps on the left, the dataset and what the export wrote on the right."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

from knotwork.export.manifest import check_manifest
from knotwork.export.run import run_export
from knotwork.freshness import lineage
from knotwork.pipeline.io import read_json
from knotwork.render.canvas import REPO_ROOT
from knotwork.studio import demo_guard, factory, forensics, guide, present, review, working_set
from studio_ui import OUT, dataset_label, dataset_select, relative, resolve
from ui_chrome import is_working_set, review_log_for

PREVIEW_PX = 220
FILE_TABLE_HEIGHT = 260
QA_LABELS = {"approved": "Approved", "flagged": "Flagged", "unreviewed": "Pending"}


def _preferred() -> str | None:
    """The batch just built in Studio, else the Working Set (the selector's first option)."""
    last = st.session_state.get("last_batch_dir")
    if last and (REPO_ROOT / last / "index.jsonl").is_file():
        return str(last)
    return None


def _export(dataset: Path, formats: list[str], out: Path, shard: float, limit: int,
            only_ids: set[str] | None = None,
            filters: dict[str, object] | None = None) -> None:  # fmt: skip
    chosen = [f for f in formats if f != "manifest"]
    with st.status("Exporting dataset…", expanded=True) as status:
        try:
            st.write(f"Reading {len(forensics.bundles(dataset))} bundles")
            code = run_export(
                dataset,
                ",".join(chosen) or "manifest",
                out,
                limit or None,
                shard or None,
                only_ids,
                filters,
            )
            st.write("Checking every file against the manifest hashes")
            problems = check_manifest(out) if code == 0 else ["the exporter wrote nothing"]
        except Exception as exc:  # the UI boundary: every failure is shown to the user
            status.update(label="Export failed", state="error")
            st.error(f"{type(exc).__name__}: {exc}")
            return
        if problems:
            status.update(label="Export needs attention", state="error")
            st.warning("Some files do not match the manifest: " + "; ".join(problems[:5]))
            return
        status.update(label="Dataset exported", state="complete", expanded=False)
    st.session_state.exported = str(out)


# ------------------------------------------------------------------ right column
def _ws_table(dataset: Path) -> None:
    """Every Working Set item with its domain, certified answer, QA status and tags."""
    rows = [{"Domain": r["domain"].replace("_", " ").title(),
             "Certified answer": r["answer"], "QA status": QA_LABELS[r["status"]],
             "Tags": r["tags"], "Bundle": r["bundle_id"]}
            for r in working_set.WorkingSet(dataset).table()]  # fmt: skip
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", key="ex_ws_table",
                 column_config={
                     "Domain": st.column_config.TextColumn(width="small"),
                     "Certified answer": st.column_config.TextColumn(width="small"),
                     "QA status": st.column_config.TextColumn(width="small"),
                     "Tags": st.column_config.ListColumn(),
                     "Bundle": st.column_config.TextColumn(width="medium"),
                 })  # fmt: skip


def _preview(dataset: Path) -> None:
    rows = forensics.bundles(dataset)
    if not rows:
        st.caption("This dataset has no bundles yet.")
        return
    bases = [r for r in rows if r.get("role", "base") == "base"]
    families = Counter(str(r.get("family", "?")) for r in rows)
    picture, facts = st.columns([1, 1.6], gap="medium", vertical_alignment="center")
    with picture:
        first = dataset / bases[0]["dir"] if bases else None
        if first is not None and (first / "beauty.png").is_file():
            st.image(factory.beauty(first, PREVIEW_PX), width=PREVIEW_PX, alt="First item")
    with facts:
        a, b = st.columns(2)
        a.metric("Bundles", len(rows), help=guide.tip("ex_bundles"))
        b.metric("Base items", len(bases), help=guide.tip("ex_bases"))
        frame = pd.DataFrame([{"Family": k, "Bundles": v} for k, v in sorted(families.items())])
        st.dataframe(frame, hide_index=True, width="stretch", alt="Bundles per question family")


def _result(out: Path) -> None:
    manifest = out / "dataset_manifest.json"
    if not manifest.is_file():
        return
    data = read_json(manifest)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    rows = present.export_summary(out)
    total = sum(r["bytes"] for r in rows)
    n_bundles = data.get("n_bundles", "?")
    item_word = "item" if n_bundles == 1 else "items"
    msg_col, dl_col = st.columns([3, 1.5], vertical_alignment="center")
    msg_col.success(
        f"**Dataset exported.** {n_bundles} {item_word}, "
        f"{sum(r['files'] for r in rows)} files, {present.human_bytes(total)} in "
        f"`{demo_guard.display_path(out)}`.  \nManifest SHA-256 `{digest}`",
        icon=":material/check_circle:",
    )
    dl_col.download_button(
        "Download export (.zip)", lambda: working_set.folder_zip(out, out.name),
        f"{out.name}_export.zip", mime="application/zip", type="primary",
        icon=":material/download:", key="ex_zip", on_click="ignore", width="stretch",
        help=guide.tip("ex_download_zip"),
    )  # fmt: skip
    files = pd.DataFrame(present.export_rows(out, data))
    st.dataframe(files, hide_index=True, width="stretch", height=FILE_TABLE_HEIGHT,
                 alt="Every exported file and its size")  # fmt: skip
    st.download_button(
        "dataset_manifest.json", manifest.read_bytes(), "dataset_manifest.json",
        icon=":material/data_object:", key="ex_manifest", on_click="ignore",
        help=guide.tip("ex_manifest"),
    )  # fmt: skip


def _lineage(dataset: Path) -> None:
    rows = forensics.bundles(dataset)
    if not rows:
        return
    nodes = forensics.lineage_nodes(dataset)
    st.caption(f"{len(nodes)} nodes: scene → render → certificate → twin → eval")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        bundle = st.selectbox("Bundle", [r["bundle_id"] for r in rows], key="ln_bundle",
                              help=guide.tip("ln_bundle"))  # fmt: skip
        check = st.button("Verify against files", icon=":material/fact_check:", key="ln_check",
                          help=guide.tip("ln_check"))  # fmt: skip
    if check:
        stored = (dataset / lineage.FILENAME).is_file()
        found = lineage.check_dataset(dataset) if stored else lineage.verify(nodes)
        if found:
            st.error("Lineage does not match the files: " + "; ".join(found[:5]))
        else:
            st.success("Lineage matches the files on disk.", icon=":material/check_circle:")
    part = forensics.lineage_slice(nodes, str(bundle))
    if part:
        st.graphviz_chart(forensics.lineage_dot(part), alt="Lineage graph of the bundle")


# ------------------------------------------------------------------ page
controls, output = st.columns([45, 55], gap="large")
with controls:
    with st.container(border=True):
        st.markdown("**1. Choose a dataset**")
        dataset = dataset_select(
            "Dataset", "ex_dataset", _preferred(), help=guide.tip("ex_dataset")
        )
    with st.container(border=True):
        st.markdown("**2. Choose formats**")
        formats = list(
            st.pills(
                "Formats", list(present.FORMAT_LABELS), selection_mode="multi",
                default=list(present.FORMAT_LABELS), format_func=present.FORMAT_LABELS.__getitem__,
                key="ex_formats", label_visibility="collapsed", help=guide.tip("ex_formats"),
            )
        )  # fmt: skip
        st.caption("Every export also writes a manifest with the SHA-256 of each file.")
        with st.expander("Options", icon=":material/tune:"):
            shard = float(st.number_input("Max parquet shard (MB, 0 = one file)", 0.0, 4096.0, 0.0,
                                          help=guide.tip("ex_shard")))  # fmt: skip
            limit = int(st.number_input("Limit bundles (0 = all)", 0, 10_000_000, 0,
                                        help=guide.tip("ex_limit")))  # fmt: skip
            default = (
                f"studio/export/{dataset.name if dataset else 'dataset'}"
                if demo_guard.is_public_demo()
                else relative(OUT / "export" / (dataset.name if dataset else "dataset"))
            )
            folder = st.text_input("Output folder", default, key=f"ex_out_{default}",
                                   help=guide.tip("ex_out"))  # fmt: skip
    with st.container(border=True):
        st.markdown("**3. Export**")
        refused = demo_guard.path_problem(folder)
        if refused:
            st.error(refused, icon=":material/folder_off:")
        out = OUT / "export" / "sandbox" if refused else resolve(folder)
        only = st.toggle("Approved items only", key="ex_approved",
                         help=guide.tip("qa_approved_only"))  # fmt: skip
        only_ids: set[str] | None = None
        filters: dict[str, object] | None = None
        states = review.load_review_states(review_log_for(dataset))
        ids = [r["bundle_id"] for r in forensics.bundles(dataset)] if dataset else []
        used = sorted({t for i in ids for t in review.tags_of(states, i)})
        tags = list(st.pills("Filter by tag", list(dict.fromkeys([*review.TAGS, *used])),
                             selection_mode="multi", key="ex_tags",
                             help=guide.tip("ex_tags")))  # fmt: skip
        if (only or tags) and dataset is not None:
            keep = set(ids)
            if only:
                keep = {i for i in keep if review.status_of(states, i) == "approved"}
            if tags:
                keep = {i for i in keep if set(review.tags_of(states, i)) & set(tags)}
            only_ids = keep
            filters = {"qa_filter": review.QA_FILTER if only else "none", "tag_filter": tags}
            where = "Working Set reviews" if is_working_set(dataset) else "Human Eval reviews"
            if keep:
                st.caption(f"{len(keep)} of {len(ids)} items match ({where}).")
            else:
                st.warning(f"0 of {len(ids)} items match these filters. Review or tag items in "
                           "Studio or Human Eval, or clear a filter.", icon=":material/rule:")  # fmt: skip
        st.caption(f"Writes to `{demo_guard.display_path(out)}`.")
        go = st.button(
            "Export dataset", type="primary", width="stretch", icon=":material/package_2:",
            disabled=dataset is None or not formats or only_ids == set() or bool(refused),
            key="ex_go",
            help=guide.tip("ex_go"),
        )  # fmt: skip
        if not formats:
            st.caption("Choose at least one format to export.")
with output:
    exported_here = st.session_state.get("exported") == str(out)
    if go and dataset is not None:
        _export(dataset, formats, out, shard, limit, only_ids, filters)
        exported_here = st.session_state.get("exported") == str(out)
    if exported_here:
        with st.container(border=True):
            st.markdown("**Export output**")
            _result(out)
    with st.container(border=True):
        if dataset is None:
            st.info("Choose a dataset on the left to preview it.", icon=":material/dataset:")
        else:
            st.markdown(f"**{dataset_label(relative(dataset))}**")
            _preview(dataset)
            if is_working_set(dataset):
                _ws_table(dataset)
    if not exported_here:
        with st.container(border=True):
            st.markdown("**Export output**")
            st.caption("Nothing exported yet. Press Export dataset to write the files here.")
    if dataset is not None:
        with st.expander("Manifest & lineage", icon=":material/account_tree:"):
            _lineage(dataset)
