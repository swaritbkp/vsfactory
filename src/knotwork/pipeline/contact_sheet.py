"""Contact sheet with Pixel-Witness Proof strips (SPEC 9.1 "Contact sheet").

    python -m knotwork.pipeline.contact_sheet --data artifacts/phase2 --out artifacts/phase2/contact_sheet.png

Rows: tangle / gauge / panel (base, causal twin, nuisance twin) from ``DATA/data_<plugin>``; Pixel-Witness
Proof strips ``beauty | id_buffer | kernel_overlay`` whose overlays are computed by calling knotcore on
the bundle files (never plugin geometry); and a fault row M1-M10 from a ``mutate --n 1`` run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from knotwork.knotcore import gauge as kgauge
from knotwork.knotcore import ocr, topo
from knotwork.knotcore.contracts import FAULT_EXPECTED, FaultCode
from knotwork.knotcore.io import Bundle, KernelFault, load_bundle
from knotwork.pipeline.generate import load_profile
from knotwork.pipeline.io import read_json, read_jsonl
from knotwork.render.canvas import REPO_ROOT

CELL, CAP, GAP, LEFT, TOP, COLS = 400, 40, 10, 230, 70, 5
TAB20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120), (44, 160, 44), (152, 223, 138),
         (214, 39, 40), (255, 152, 150), (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
         (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199), (188, 189, 34), (219, 219, 141),
         (23, 190, 207), (158, 218, 229)]  # fmt: skip
FONT = REPO_ROOT / "assets" / "fonts" / "DejaVuSans.ttf"


def font(px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT), px)


def id_colormap(ids: np.ndarray) -> Image.Image:
    lut = np.array([(0, 0, 0)] + [TAB20[(k - 1) % 20] for k in range(1, 65536)], dtype=np.uint8)
    return Image.fromarray(lut[ids.astype(np.int64)], mode="RGB")


def _starts(b: Bundle, task: dict[str, Any] | None) -> list[str]:
    if task:
        for st in task["question"]["steps"]:
            if st["op"] in ("TRACE", "CROSS", "ONTOP") and str(st["args"][0]).startswith(
                ("tangle/", "panel/")
            ):
                return [str(st["args"][0])]
    return [t.terminal_id for t in b.manifest.terminals if t.role == "start"][:1]


def kernel_overlay(b: Bundle, cfg: dict[str, Any], task: dict[str, Any] | None) -> Image.Image:
    """Beauty dimmed 50 % plus kernel evidence: TRACE component (cyan), crossing centres with halo
    circles coloured by the ONTOP winner, needle PCA axis + dial centre, OCR text at each label box."""
    base = (b.beauty.astype(np.float64) * 0.5 + 127.5).astype(np.uint8)
    for start in _starts(b, task):
        try:
            _, comp = topo.trace_component(b, start)
            base[comp] = (0, 200, 230)
        except KernelFault:
            continue
    img = Image.fromarray(base, mode="RGB")
    d = ImageDraw.Draw(img)
    f18 = font(18)
    seen: set[tuple[int, int]] = set()
    for t in (t for t in b.manifest.terminals if t.role == "start"):
        try:
            cs = topo.crossings(b, t.terminal_id, "*")
        except KernelFault:
            continue
        for c in cs:
            x, y = c["center"]
            key = (round(x), round(y))
            if key in seen:
                continue
            seen.add(key)
            top = int(b.ids[round(y), round(x)])
            col = TAB20[(top - 1) % 20] if top else (255, 0, 0)
            d.ellipse([x - 22, y - 22, x + 22, y + 22], outline=col, width=3)
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 0, 255))
    for g in b.manifest.gauges:
        try:
            ev = kgauge.needle(b, g, cfg["kernel"])
        except KernelFault as exc:
            d.text((20, 20), f"{g.gauge_id}: {exc.code.value}", fill=(255, 0, 0), font=f18)
            continue
        (cx, cy), (tx, ty) = ev["center"], ev["tip"]
        d.line([cx - (tx - cx) * 0.2, cy - (ty - cy) * 0.2, tx, ty], fill=(255, 0, 255), width=4)
        d.line([cx - 12, cy, cx + 12, cy], fill=(0, 160, 255), width=3)
        d.line([cx, cy - 12, cx, cy + 12], fill=(0, 160, 255), width=3)
    for o in (o for o in b.manifest.objects if o.kind.value == "label" and o.box is not None):
        try:
            text, ok = ocr.label_text(b, o.obj_id, cfg["kernel"])[0], True
        except KernelFault as exc:
            text, ok = exc.code.value.removeprefix("E_"), False
        box = o.box
        assert box is not None
        d.rectangle(
            [box.x0, box.y0, box.x1, box.y1], outline=(0, 170, 0) if ok else (230, 0, 0), width=2
        )
        d.text((box.x1 + 2, box.y0 - 16), text, fill=(0, 120, 0) if ok else (230, 0, 0), font=f18)
    return img


def _thumb(img: Image.Image) -> Image.Image:
    return img.convert("RGB").resize((CELL, CELL), Image.Resampling.LANCZOS)


def _beauty(d: Path) -> Image.Image:
    return Image.open(d / "beauty.png")


def _plugin_row(
    data: Path, plugin: str
) -> tuple[list[tuple[str, Image.Image, str]], dict[str, Any], Path | None]:
    ds = data / f"data_{plugin}"
    index = read_jsonl(ds / "index.jsonl")
    base = next((r for r in index if r["role"] == "base"), None)
    if base is None:
        return [], {"name": plugin, "panels": []}, None
    by_id = {r["bundle_id"]: r for r in index}
    task = read_json(ds / base["dir"] / "task.json")
    cells, panels = [], []
    for kind, bid in (("base", base["bundle_id"]), ("causal", task["twins"]["causal_id"]),
                      ("nuisance", task["twins"]["nuisance_id"])):  # fmt: skip
        row = by_id.get(bid)
        if row is None:
            continue
        t = read_json(ds / row["dir"] / "task.json")
        diff = t["twins"].get("pixel_diff_frac")
        cap = f"{kind} | {t['family']} | {t['answer']['canonical']}" + (
            f" | dpx {diff:.3f}" if kind != "base" else ""
        )
        cells.append((kind, _beauty(ds / row["dir"]), cap))
        panels.append({"kind": kind, "bundle_id": bid, "caption": cap})
    return cells, {"name": plugin, "panels": panels}, ds / base["dir"]


def build(
    data: Path, out: Path, profile_path: Path, fault_report: Path | None = None
) -> dict[str, Any]:
    """Compose the sheet; ``fault_report`` reuses a kept ``mutate --n 1`` run (else one is run)."""
    from knotwork.pipeline.mutate import run_mutate

    profile = load_profile(profile_path)
    grid: list[tuple[str, list[tuple[str, Image.Image, str]]]] = []
    meta: dict[str, Any] = {"rows": [], "faults": []}
    witness_dirs: dict[str, Path] = {}
    for plugin in ("tangle", "gauge", "panel"):
        cells, row, base_dir = _plugin_row(data, plugin)
        grid.append((plugin, cells))
        meta["rows"].append(row)
        if base_dir is not None:
            witness_dirs[plugin] = base_dir
    for plugin in ("tangle", "panel", "gauge"):
        d = witness_dirs.get(plugin)
        if d is None:
            continue
        b = load_bundle(d)
        task = read_json(d / "task.json")
        bid = task["bundle_id"]
        cells = [("beauty", Image.fromarray(b.beauty), "beauty.png (what the VLM sees)"),
                 ("id_buffer", id_colormap(b.ids), "id_buffer.png (modal object ids)"),
                 ("kernel_overlay", kernel_overlay(b, profile, task), "kernel overlay (knotcore on files)")]  # fmt: skip
        grid.append((f"pixel_witness:{plugin}", cells))
        meta["rows"].append({"name": f"pixel_witness:{plugin}",
                             "panels": [{"kind": k, "bundle_id": bid, "caption": c} for k, _, c in cells]})  # fmt: skip
    if fault_report is None:
        fault_report = data / "contact_faults.json"
        run_mutate("all", 1, 0, fault_report, profile_path, keep_dir=data / "contact_faults")
    rep = read_json(fault_report)
    fcells = []
    for f in FaultCode:
        row = rep["faults"][f.value]
        ex = (row.get("examples") or [None])[0]
        exp = FAULT_EXPECTED[f].value
        code = (
            exp
            if ex and exp in ex["codes"]
            else (ex["codes"][0] if ex and ex["codes"] else "MISSED")
        )
        img = _beauty(Path(ex["dir"])) if ex else Image.new("RGB", (CELL, CELL), (60, 0, 0))
        fcells.append((f.value, img, f"{f.value} -> {code}"))
        meta["faults"].append(
            {"fault": f.value, "code": code, "bundle_id": ex["bundle_id"] if ex else None}
        )
    grid.append(("faults M1-M5", fcells[:5]))
    grid.append(("faults M6-M10", fcells[5:]))
    _compose(grid, out)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def _compose(grid: list[tuple[str, list[tuple[str, Image.Image, str]]]], out: Path) -> None:
    w = LEFT + COLS * (CELL + GAP) + GAP
    h = TOP + len(grid) * (CELL + CAP + GAP) + GAP
    sheet = Image.new("RGB", (w, h), (250, 250, 250))
    d = ImageDraw.Draw(sheet)
    d.text((GAP, 18), "Knotwork contact sheet - every caption and overlay is computed from the bundle files",
           fill=(20, 20, 20), font=font(30))  # fmt: skip
    f20, f16 = font(22), font(17)
    for r, (name, cells) in enumerate(grid):
        y = TOP + r * (CELL + CAP + GAP)
        d.text((GAP, y + CELL // 2 - 12), name, fill=(20, 20, 20), font=f20)
        for c, (_, img, cap) in enumerate(cells):
            x = LEFT + GAP + c * (CELL + GAP)
            sheet.paste(_thumb(img), (x, y))
            d.text((x + 2, y + CELL + 6), cap[:52], fill=(20, 20, 20), font=f16)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--profile", type=Path, default=REPO_ROOT / "configs" / "default_profile.yaml")
    args = ap.parse_args(argv)
    meta = build(args.data, args.out, args.profile)
    print(f"contact sheet -> {args.out} ({len(meta['rows'])} rows, {len(meta['faults'])} faults)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
