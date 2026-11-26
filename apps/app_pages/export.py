"""Export: three steps from a dataset folder to hashed deliverables, then its lineage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from knotwork.export.manifest import check_manifest
from knotwork.export.run import run_export
from knotwork.freshness import lineage
from knotwork.studio import forensics, present
from studio_ui import OUT, dataset_select, relative, resolve


def _export(dataset: Path, formats: list[str], out: Path, shard: float, limit: int) -> None:
    chosen = [f for f in formats if f != "manifest"]
    with st.status("Exporting", expanded=True) as status:
        try:
            st.write(f"Reading {len(forensics.bundles(dataset))} bundles")
            code = run_export(
                dataset, ",".join(chosen) or "manifest", out, limit or None, shard or None
            )
            st.write("Checking every file against the manifest hashes")
            problems = check_manifest(out) if code == 0 else ["nothing to export"]
        except Exception as exc:  # the UI boundary: every failure is shown to the user
            status.update(label="Export failed", state="error")
            st.error(f"{type(exc).__name__}: {exc}")
            return
        if problems:
            status.update(label="Export needs attention", state="error")
            st.warning("; ".join(problems[:5]))
            return
        status.update(label="Export complete", state="complete", expanded=False)
    st.session_state.exported = str(out)


def _summary(out: Path) -> None:
    rows = present.export_summary(out)
    if not rows:
        return
    a, b, c = st.columns(3)
    a.metric("Formats", len(rows))
    b.metric("Files", sum(r["files"] for r in rows))
    c.metric("Size", present.human_bytes(sum(r["bytes"] for r in rows)))
    table = pd.DataFrame(
        [{"Format": r["format"], "Files": r["files"], "Size": present.human_bytes(r["bytes"])}
         for r in rows]
    )  # fmt: skip
    st.dataframe(table, hide_index=True, width="stretch", alt="Exported files by format")
    st.caption(f"Everything in `{relative(out)}`, hashed in dataset_manifest.json.")
    st.download_button(
        "Download dataset_manifest.json", (out / "dataset_manifest.json").read_bytes(),
        "dataset_manifest.json", icon=":material/download:", key="ex_manifest",
    )  # fmt: skip


def _lineage(dataset: Path) -> None:
    rows = forensics.bundles(dataset)
    if not rows:
        return
    nodes = forensics.lineage_nodes(dataset)
    st.caption(f"{len(nodes)} nodes: scene → render → certificate → twin → eval")
    bundle = st.selectbox("Bundle", [r["bundle_id"] for r in rows], key="ln_bundle")
    if st.button("Verify against files", icon=":material/fact_check:", key="ln_check"):
        stored = (dataset / lineage.FILENAME).is_file()
        found = lineage.check_dataset(dataset) if stored else lineage.verify(nodes)
        if found:
            st.error("; ".join(found[:5]))
        else:
            st.success("Lineage is consistent.", icon=":material/check_circle:")
    part = forensics.lineage_slice(nodes, str(bundle))
    if part:
        st.graphviz_chart(forensics.lineage_dot(part), alt="Lineage graph of the bundle")


_, middle, _ = st.columns([1, 3, 1])
with middle:
    with st.container(border=True):
        st.markdown("**1 · Select dataset folder**")
        dataset = dataset_select("Dataset", "ex_dataset")
    with st.container(border=True):
        st.markdown("**2 · Pick formats**")
        formats = list(
            st.pills(
                "Formats", list(present.FORMAT_LABELS), selection_mode="multi",
                default=list(present.FORMAT_LABELS), format_func=present.FORMAT_LABELS.__getitem__,
                key="ex_formats", label_visibility="collapsed",
            )
        )  # fmt: skip
        st.caption("A manifest with the SHA-256 of every file is written with any export.")
        with st.popover("Options", icon=":material/tune:"):
            shard = float(st.number_input("Max parquet shard (MB, 0 = one file)", 0.0, 4096.0, 0.0))
            limit = int(st.number_input("Limit bundles (0 = all)", 0, 10_000_000, 0))
            default = relative(OUT / "export" / (dataset.name if dataset else "dataset"))
            folder = st.text_input("Output folder", default, key=f"ex_out_{default}")
    with st.container(border=True):
        st.markdown("**3 · Export**")
        out = resolve(folder)
        if st.button(
            "Export dataset", type="primary", width="stretch", icon=":material/package_2:",
            disabled=dataset is None or not formats, key="ex_go",
        ) and dataset is not None:  # fmt: skip
            _export(dataset, formats, out, shard, limit)
        if st.session_state.get("exported") == str(out):
            _summary(out)
    if dataset is not None:
        with st.expander("Lineage", icon=":material/account_tree:"):
            _lineage(dataset)
