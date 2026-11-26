"""`knot counterfactual`: for each failed item, a certified Diagnostic Twin Suite.

* ``perceptual_clean``    -- same logic and topology; no realism, clutter, noise or jitter, high contrast
* ``palette_swap``        -- the same scene under a rotated colour-vision-safe palette
* ``reasoning_step_down`` -- one crossing fewer on the queried path (cable families only)
* ``text_ir_twin``        -- the task as structured text (separates seeing from reasoning)

Every visual twin is rendered and certified by `knotcore.engine.certify`; an uncertified twin is
reported with its codes and left out of `prompts.jsonl`. Reading the results: solving
``perceptual_clean`` after failing the original points at perception under clutter; solving only
``reasoning_step_down`` points at reasoning depth; solving ``text_ir_twin`` but no image points at
the vision encoder.
"""

from __future__ import annotations

import copy
import html
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from knotwork.diversity import rebuild_scene
from knotwork.knotcore.contracts import QuestionProgram, SceneIR
from knotwork.knotcore.grading import committed_answers
from knotwork.knotcore.seeds import SeedTree
from knotwork.pipeline import text_ir
from knotwork.pipeline.certify import run_candidate
from knotwork.pipeline.generate import load_profile
from knotwork.pipeline.io import append_jsonl, read_json, write_json
from knotwork.plugins.gauge.plugin import BEZELS, FACES, TINTS
from knotwork.plugins.payloads import validate_payload
from knotwork.plugins.registry import get_plugin
from knotwork.plugins.tangle import braid, geom
from knotwork.plugins.tangle.plugin import MONO, PALETTE, lay_of
from knotwork.plugins.tangle.render import hex_rgb
from knotwork.render.canvas import REPO_ROOT

VISUAL = ("perceptual_clean", "palette_swap", "reasoning_step_down")
KINDS = (*VISUAL, "text_ir_twin")
DEFAULT_PROFILE = REPO_ROOT / "configs" / "default_profile.yaml"
BUNDLE_RE = re.compile(r"kw-[0-9a-f]{20}")
PLAIN_BACKGROUND = [246, 246, 243]
THUMB_PX = 480
PALETTE_SHIFT = 3
#: braid-word cable plugins (the rack patch bay is a tangle drawn as a rack)
TANGLES = ("tangle", "rack_patch")


class Unavailable(RuntimeError):
    """This twin cannot be built for this item; the message says why."""


@dataclass
class Failure:
    bundle_id: str
    replies: dict[str, str] = field(default_factory=dict)
    note: str = ""
    key: str = ""
    question: str = ""
    truth: str = ""

    @property
    def read_sub_question(self) -> bool:
        """A bench composition failure on "what does this gauge read", not on the bundle's own
        question."""
        return self.key.startswith("read:") and bool(self.question and self.truth)


# ------------------------------------------------------------------ inputs
def _records(path: Path) -> Iterable[dict[str, Any]]:
    if path.is_dir():
        files = sorted(path.rglob("failure.json"))
        if not files:
            raise ValueError(f"no failure.json under {path}")
        for f in files:
            yield read_json(f)
        return
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
        yield from data if isinstance(data, list) else [data]
        return
    for line in text.splitlines():
        if line.strip():
            yield json.loads(line)


def load_failures(path: Path) -> list[Failure]:
    """Failed items from a `failures.jsonl`, a bench `trials.jsonl` (correct rows are skipped), one
    JSON file, or a directory of `failure.json` files; merged per bundle."""
    found: dict[str, Failure] = {}
    for row in _records(Path(path)):
        if row.get("correct") is True or row.get("missing") is True:
            continue
        match = BUNDLE_RE.search(f"{row.get('bundle_id', '')} {row.get('key', '')}")
        if not match:
            continue
        item = found.setdefault(match.group(0), Failure(match.group(0)))
        item.replies.update({str(k): str(v) for k, v in (row.get("replies") or {}).items()})
        model = row.get("model") or row.get("slot")
        reply = row.get("reply", row.get("prediction"))
        if model is not None and reply is not None:
            item.replies[str(model)] = str(reply)
        item.note = item.note or str(row.get("block", ""))
        item.key = item.key or str(row.get("key", ""))
        item.question = item.question or str(row.get("question", ""))
        item.truth = item.truth or str(row.get("truth", row.get("answer", "")) or "")
    return list(found.values())


def index_bundles(data: Path) -> dict[str, Path]:
    """bundle id -> directory holding its `task.json`, anywhere under `data`."""
    out: dict[str, Path] = {}
    for task in sorted(Path(data).rglob("task.json")):
        bid = read_json(task).get("bundle_id")
        if bid:
            out.setdefault(str(bid), task.parent)
    return out


def dataset_of(bundle: Path, root: Path) -> Path | None:
    for parent in [bundle, *bundle.parents]:
        if (parent / "freshness_receipt.json").is_file() and (parent / "index.jsonl").is_file():
            return parent
        if parent == root:
            break
    return None


# ------------------------------------------------------------------ transforms
def _surfaces(pay: dict[str, Any]) -> list[dict[str, Any]]:
    dials = pay.get("dials", [pay["dial"]] if "dial" in pay else [])
    return [pay["surface"], *(d["surface"] for d in dials)]


def perceptual_clean(plugin: str, pay: dict[str, Any]) -> None:
    pay["plain"] = True
    pay["relief"] = []
    for s in _surfaces(pay):
        s.update(light=0.0, grain=0.0, brush=0.0, smudges=0, dust=0, noise=0.0, dpi_scale=1.0)
    pay["render"]["background_rgb"] = list(PLAIN_BACKGROUND)
    if plugin == "gauge":
        pay["stem"] = pay["plate"] = None
        return
    pay["style"] = "flat"
    if pay["color_regime"] == "mono":
        pay["colors"] = [[20, 20, 20] for _ in pay["colors"]]
    if plugin in TANGLES:
        trial = copy.deepcopy(pay)
        trial["layout"]["jitter_amps"] = [0.0, 0.0, 0.0]
        if geom.validate(lay_of(trial), trial["n"], float(trial["stroke"])) is None:
            pay["layout"] = trial["layout"]


def palette_swap(plugin: str, pay: dict[str, Any]) -> None:
    """Rotate the cable palette (distinct colours stay distinct); a gauge has no semantic cable
    colour, so its housing, bezel and face tones rotate instead and the zone colours stay."""
    if plugin == "gauge":
        d = pay["dial"]
        for target, key, options in ((d, "bezel_rgb", BEZELS), (d, "face_rgb", FACES),
                                     (pay["render"], "background_rgb", TINTS)):  # fmt: skip
            tones = [list(t) for t in options]
            at = tones.index(target[key]) if target[key] in tones else -1
            target[key] = tones[(at + 1) % len(tones)]
        return
    wheel = [list(hex_rgb(h)) for h in PALETTE]
    mono = list(hex_rgb(MONO))

    def rotate(color: list[int]) -> list[int]:
        if color == mono or color not in wheel:
            return wheel[6]
        return wheel[(wheel.index(color) + PALETTE_SHIFT) % len(wheel)]

    pay["colors"] = [rotate(list(c)) for c in pay["colors"]]


def _queried_strands(plugin: str, pay: dict[str, Any], q: QuestionProgram) -> set[int]:
    labels = [str(a).rsplit("_", 1)[-1] for s in q.steps for a in s.args if "start_" in str(a)]
    if plugin in TANGLES:
        return {pay["start_labels"].index(x) for x in labels if x in pay["start_labels"]}
    return {int(s) for s, lab in pay["sockets"].items() if lab in labels}


def step_down_candidates(plugin: str, pay: dict[str, Any], q: QuestionProgram) -> list[int]:
    """Word positions whose removal takes one crossing off the queried path and keeps the task
    well formed: never a crossing between the two queried cables, and on a panel only an exchange
    of two cables of the same destination kind (so every socket still reaches a gauge)."""
    if plugin == "gauge":
        raise Unavailable("a single-gauge reading has no reasoning step to remove")
    queried = _queried_strands(plugin, pay, q)
    if not queried:
        raise Unavailable("the question names no single cable to shorten")
    word = [None if x is None else tuple(int(v) for v in x) for x in pay["word"]]
    sockets = {int(s) for s in pay.get("sockets", {})}
    out = []
    for c in braid.compose(pay["n"], word)[1]:
        pair = {c.a, c.b}
        if not pair & queried or pair <= queried:
            continue
        if plugin == "panel" and len(pair & sockets) == 1:
            continue
        out.append(c.letter)
    if not out:
        raise Unavailable("no removable crossing on the queried path")
    return out


def remove_crossing(plugin: str, pay: dict[str, Any], letter: int) -> None:
    if plugin in TANGLES:
        pay["word"][letter] = None  # an identity slot keeps the column layout
    else:
        del pay["word"][letter]
    word = [None if x is None else tuple(int(v) for v in x) for x in pay["word"]]
    pay["perm"] = [int(v) for v in braid.compose(pay["n"], word)[0]]


def _rewire_panel(pay: dict[str, Any]) -> None:
    from knotwork.plugins.panel import compose

    word = [None if x is None else tuple(int(v) for v in x) for x in pay["word"]]
    plan = compose.braid_plan(pay["n"], word, float(pay["stroke"]))
    if plan is None:
        raise Unavailable("the shortened panel braid has no legible layout")
    plan["y_top"] = pay["braid"]["y_top"]
    pay["braid"] = plan


# ------------------------------------------------------------------ suite
def _certify_twin(
    scene: SceneIR, question: QuestionProgram, out: Path, profile: dict[str, Any]
) -> dict[str, Any]:
    plugin = get_plugin(scene.plugin)
    validate_payload(scene.plugin, scene.payload)
    if scene.plugin == "panel":
        question = plugin.twin_question(scene, question)  # type: ignore[attr-defined]
    cand = run_candidate(plugin, scene, question, out, profile)
    codes = [c.value for c in cand.certificate.codes]
    return {
        "status": "certified" if cand.certified else "quarantined",
        "codes": codes,
        "answer": cand.answer.model_dump(mode="json"),
        "question": cand.question.text,
        "image": str(out / "beauty.png"),
    }


def build_visual_twin(
    kind: str, base: SceneIR, question: QuestionProgram, out: Path, profile: dict[str, Any]
) -> dict[str, Any]:
    """One certified visual twin; `reasoning_step_down` tries removable crossings in a seeded order
    until one certifies."""
    pay = copy.deepcopy(base.payload)
    if kind == "perceptual_clean":
        perceptual_clean(base.plugin, pay)
        return _certify_twin(base.model_copy(update={"payload": pay}), question, out, profile)
    if kind == "palette_swap":
        palette_swap(base.plugin, pay)
        return _certify_twin(base.model_copy(update={"payload": pay}), question, out, profile)
    letters = step_down_candidates(base.plugin, pay, question)
    rng = SeedTree(0).child("counterfactual", base.seed_path).rng()
    last = "no candidate gave a legible scene"
    for attempt, k in enumerate(int(letters[i]) for i in rng.permutation(len(letters))):
        trial = copy.deepcopy(base.payload)
        remove_crossing(base.plugin, trial, k)
        try:
            if base.plugin == "panel":
                _rewire_panel(trial)
            elif geom.validate(lay_of(trial), trial["n"], float(trial["stroke"])) is not None:
                continue
            got = _certify_twin(
                base.model_copy(update={"payload": trial}), question, out / f"try{attempt}", profile
            )
        except (Unavailable, ValueError, RuntimeError) as exc:
            last = str(exc)[:200]
            continue
        if got["status"] == "certified" or attempt >= 3:
            return got | {"removed_crossing": k}
    raise Unavailable(last)


def build_suite(
    failure: Failure, data: Path, bundle: Path, out: Path, profile: dict[str, Any]
) -> dict[str, Any]:
    """Every twin of one failed item, each either built or marked unavailable with the reason."""
    task = read_json(bundle / "task.json")
    suite: dict[str, Any] = {
        "bundle_id": failure.bundle_id,
        "plugin": task["plugin"],
        "family": task["family"],
        "question": task["question"]["text"],
        "answer": task["answer"],
        "replies": failure.replies,
        "twins": {},
    }
    original = next((p for p in (bundle / "beauty.png", bundle / "image.jpg") if p.is_file()), None)
    suite["image"] = str(original) if original else None
    scene, why = None, "no dataset directory (index.jsonl + receipt) holds this bundle"
    ds = dataset_of(bundle, data)
    if ds is not None:
        try:
            scene = rebuild_scene(ds, task, profile)
        except (ValueError, KeyError, RuntimeError, OSError) as exc:
            why = f"the scene cannot be re-sampled by this version: {str(exc)[:160]}"
    for kind in VISUAL:
        if scene is None:
            suite["twins"][kind] = {"status": "unavailable", "reason": why}
            continue
        try:
            question = QuestionProgram.model_validate(task["question"])
            suite["twins"][kind] = build_visual_twin(kind, scene, question, out / kind, profile)
        except (Unavailable, ValueError, RuntimeError) as exc:
            suite["twins"][kind] = {"status": "unavailable", "reason": str(exc)[:200]}
    asked, answer = str(task["question"]["text"]), task["answer"]
    sub_question = None
    if failure.read_sub_question:
        sub_question = asked = failure.question
        answer = {"type": "float", "value": float(failure.truth), "canonical": failure.truth,
                  "tolerance": float(task["answer"].get("tolerance") or 0.0)}  # fmt: skip
        suite["failed_question"] = asked
    text = (
        text_ir.from_scene(scene, asked)
        if scene is not None
        else text_ir.from_certificate(task, sub_question)
    )
    if text is None:
        suite["twins"]["text_ir_twin"] = {"status": "unavailable", "reason": why}
    else:
        out.mkdir(parents=True, exist_ok=True)
        (out / "text_ir.txt").write_text(text, encoding="utf-8")
        source = "scene" if scene is not None else "kernel evidence"
        suite["twins"]["text_ir_twin"] = {
            "status": "text", "source": source, "prompt": text, "answer": answer,
        }  # fmt: skip
    return suite


# ------------------------------------------------------------------ report
def _thumb(source: str | None, out: Path, name: str) -> str | None:
    if not source or not Path(source).is_file():
        return None
    target = out / "images" / f"{name}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as im:
        rgb = im.convert("RGB")
        rgb.thumbnail((THUMB_PX, THUMB_PX), Image.Resampling.LANCZOS)
        rgb.save(target, quality=90)
    return target.relative_to(out).as_posix()


def _cell(twin: dict[str, Any]) -> tuple[str, str]:
    """(status line, answer) of a twin for the report tables."""
    if twin["status"] == "unavailable":
        return f"unavailable: {twin['reason']}", ""
    codes = ", ".join(twin.get("codes", []))
    return twin["status"] + (f" ({codes})" if codes else ""), str(twin["answer"]["canonical"])


def render_report(suites: list[dict[str, Any]], out: Path) -> None:
    """`report.md` and a self-describing `report.html` with the twins side by side."""
    md = ["# Diagnostic twin suites", ""]
    page = [
        "<!doctype html><meta charset='utf-8'><title>Diagnostic twin suites</title><style>",
        "body{font:14px system-ui,sans-serif;margin:24px;color:#1c1c1c}",
        ".row{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:32px}",
        ".card{width:300px;border:1px solid #ccc;border-radius:6px;padding:8px}",
        ".card img{width:100%}pre{white-space:pre-wrap;font-size:12px}</style>",
        "<h1>Diagnostic twin suites</h1>",
    ]
    for s in suites:
        replies = "; ".join(f"{m}: {r}" for m, r in sorted(s["replies"].items())) or "none recorded"
        if s.get("failed_question"):
            replies += f" (to the sub-question: {s['failed_question']})"
        md += [f"## {s['bundle_id']} ({s['family']})", "", s["question"], "",
               f"* certified answer: `{s['answer']['canonical']}`", f"* failed replies: {replies}", "",
               "| twin | status | answer |", "|---|---|---|"]  # fmt: skip
        page += [
            f"<h2>{html.escape(s['bundle_id'])} &middot; {html.escape(s['family'])}</h2>",
            f"<p>{html.escape(s['question'])}</p><p>Certified answer "
            f"<b>{html.escape(str(s['answer']['canonical']))}</b>. Failed replies: "
            f"{html.escape(replies)}</p><div class='row'>",
        ]
        cards = [("original", {"status": "certified", "answer": s["answer"], "image": s["image"]})]
        cards += [(k, s["twins"][k]) for k in KINDS]
        for kind, twin in cards:
            status, answer = _cell(twin)
            if kind != "original":
                md.append(f"| {kind} | {status.replace('|', '/')} | {answer} |")
            body = f"<div class='card'><h3>{kind}</h3><p>{html.escape(status)}</p>"
            thumb = _thumb(twin.get("image"), out, f"{s['bundle_id']}_{kind}")
            if thumb:
                body += f"<img src='{thumb}' alt='{kind}'>"
            if twin.get("prompt"):
                body += f"<pre>{html.escape(twin['prompt'])}</pre>"
            page.append(
                body + (f"<p>answer: <b>{html.escape(answer)}</b></p>" if answer else "") + "</div>"
            )
        md.append("")
        page.append("</div>")
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "report.html").write_text("\n".join(page) + "\n", encoding="utf-8")


def prompt_rows(suites: list[dict[str, Any]], out: Path) -> list[dict[str, Any]]:
    """Evaluation rows for every usable twin (image path relative to `out`, or the text prompt)."""
    rows = []
    for s in suites:
        for kind in KINDS:
            twin = s["twins"][kind]
            if twin["status"] not in ("certified", "text"):
                continue
            image = twin.get("image")
            rows.append(
                {
                    "bundle_id": s["bundle_id"],
                    "kind": kind,
                    "prompt": twin.get("prompt", twin.get("question")),
                    "image": Path(image).relative_to(out).as_posix() if image else None,
                    "answer": twin["answer"],
                }
            )
    return rows


def run_counterfactual(
    data: Path, failures: Path, out: Path, profile: Path = DEFAULT_PROFILE, limit: int | None = None
) -> int:
    data, out = Path(data).resolve(), Path(out).resolve()
    prof = load_profile(profile)
    items = load_failures(Path(failures))[:limit]
    if not items:
        print(f"counterfactual: no failed items in {failures}")
        return 1
    bundles = index_bundles(data)
    out.mkdir(parents=True, exist_ok=True)
    suites, missing = [], []
    for item in items:
        bundle = bundles.get(item.bundle_id)
        if bundle is None:
            missing.append(item.bundle_id)
            continue
        suites.append(build_suite(item, data, bundle, out / "twins" / item.bundle_id, prof))
        for model, reply in item.replies.items():
            if len(committed_answers(reply)) > 1:
                suites[-1]["replies"][model] = f"{reply} [hedged]"
    write_json(out / "suites.json", {"suites": suites, "missing_bundles": missing})
    (out / "prompts.jsonl").unlink(missing_ok=True)
    rows = prompt_rows(suites, out)
    append_jsonl(out / "prompts.jsonl", rows)
    render_report(suites, out)
    print(
        f"counterfactual: {len(suites)} suites, {len(rows)} usable twins, "
        f"{len(missing)} bundles not found under {data}; report {out / 'report.html'}"
    )
    return 0 if suites else 1
