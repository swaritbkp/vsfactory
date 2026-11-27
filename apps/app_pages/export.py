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
from knotwork.studio import factory, forensics, guide, present, review
from studio_ui import OUT, dataset_label, dataset_select, relative, resolve

FALLBACKS = ("artifacts/selftest/data/tangle", "artifacts/phase3/bench_split/tangle")
PREVIEW_PX = 220
FILE_TABLE_HEIGHT = 260


def _preferred() -> str | None:
    """The batch just built in Studio, else a tangle dataset."""
    last = st.session_state.get("last_batch_dir")
    if last and (REPO_ROOT / last / "index.jsonl").is_file():
        return str(last)
    return next((f for f in FALLBACKS if (REPO_ROOT / f / "index.jsonl").is_file()), None)


def _export(dataset: Path, formats: list[str], out: Path, shard: float, limit: int,
            only_ids: set[str] | None = None) -> None:  # fmt: skip
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
    st.success(
        f"**Dataset exported.** {data.get('n_bundles', '?')} items, "
        f"{sum(r['files'] for r in rows)} files, {present.human_bytes(total)} in "
        f"`{relative(out)}`.  \nManifest SHA-256 `{digest}`",
        icon=":material/check_circle:",
    )
    files = pd.DataFrame(present.export_rows(out, data))
    st.dataframe(files, hide_index=True, width="stretch", height=FILE_TABLE_HEIGHT,
                 alt="Every exported file and its size")  # fmt: skip
    st.download_button(
        "Download dataset_manifest.json", manifest.read_bytes(), "dataset_manifest.json",
        icon=":material/download:", key="ex_manifest", width="stretch",
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
            default = relative(OUT / "export" / (dataset.name if dataset else "dataset"))
            folder = st.text_input("Output folder", default, key=f"ex_out_{default}",
                                   help=guide.tip("ex_out"))  # fmt: skip
    with st.container(border=True):
        st.markdown("**3. Export**")
        out = resolve(folder)
        only = st.toggle("Approved items only", key="ex_approved",
                         help=guide.tip("qa_approved_only"))  # fmt: skip
        only_ids: set[str] | None = None
        if only and dataset is not None:
            states = review.load_review_states(review.STUDIO_LOG)
            ids = [r["bundle_id"] for r in forensics.bundles(dataset)]
            only_ids = {i for i in ids if review.status_of(states, i) == "approved"}
            if only_ids:
                st.caption(review.filter_note(len(only_ids), len(ids)) + " (Human Eval reviews).")
            else:
                st.warning(f"{review.filter_note(0, len(ids))}. Approve items in Human Eval, or "
                           "turn this off to export every item.", icon=":material/rule:")  # fmt: skip
        st.caption(f"Writes to `{relative(out)}`.")
        go = st.button(
            "Export dataset", type="primary", width="stretch", icon=":material/package_2:",
            disabled=dataset is None or not formats or only_ids == set(), key="ex_go",
            help=guide.tip("ex_go"),
        )  # fmt: skip
        if not formats:
            st.caption("Choose at least one format to export.")
with output:
    with st.container(border=True):
        if dataset is None:
            st.info("Choose a dataset on the left to preview it.", icon=":material/dataset:")
        else:
            st.markdown(f"**{dataset_label(relative(dataset))}**")
            _preview(dataset)
    with st.container(border=True):
        st.markdown("**Export output**")
        if go and dataset is not None:
            _export(dataset, formats, out, shard, limit, only_ids)
        if st.session_state.get("exported") == str(out):
            _result(out)
        else:
            st.caption("Nothing exported yet. Press Export dataset to write the files here.")
    if dataset is not None:
        with st.expander("Manifest & lineage", icon=":material/account_tree:"):
            _lineage(dataset)
