#!/usr/bin/env python3
"""Rebuild `samples/demo_forensics/`: the sample the Studio's Forensics page opens with.

The images and twins are real: two tangle items are generated, and their diagnostic twins are built
and certified by the kernel. The model replies are NOT real. They are written by this script to
illustrate the two diagnoses, and every file says so. No network, no API key.

    uv run python scripts/build_demo_forensics.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "samples" / "demo_forensics"
PROFILE = REPO / "configs" / "default_profile.yaml"
SEED = 17
CANDIDATES = 6
SAMPLE_MODEL = "sample-model (written for this demo, not a model output)"
#: which twins each illustrative model answers correctly
PATTERNS: tuple[tuple[str, dict[str, bool]], ...] = (
    (
        "Vision failure",
        {"perceptual_clean": True, "palette_swap": False, "reasoning_step_down": False,
         "text_ir_twin": True},
    ),
    (
        "Reasoning failure",
        {"perceptual_clean": False, "palette_swap": False, "reasoning_step_down": True,
         "text_ir_twin": False},
    ),
)  # fmt: skip
README = """# Sample forensics data (not a model evaluation)

Opened by default on the Studio's Forensics page. The tangle items and their diagnostic twins are
real renders certified by the Knotwork kernel. **The model replies in `results.jsonl` and
`suites.json` were written by `scripts/build_demo_forensics.py` to illustrate a vision failure and
a reasoning failure. No model produced them, and they must not be reported as results.**

Rebuild with `uv run python scripts/build_demo_forensics.py`.
"""


def wrong_label(answer: str, options: list[str]) -> str:
    """The next option after the certified one (deterministically wrong)."""
    return options[(options.index(answer) + 1) % len(options)]


def main() -> int:
    from knotwork.pipeline.counterfactual import run_counterfactual
    from knotwork.pipeline.generate import run_generate
    from knotwork.pipeline.io import append_jsonl, read_json, read_jsonl, write_json
    from knotwork.studio import present

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        data, report = work / "data", work / "report"
        rc = run_generate(
            "tangle", CANDIDATES, data, SEED, "offline", PROFILE, "dev", False, ["T1_TRACE"], 0,
            difficulty={"reasoning_load": 0.4},
        )  # fmt: skip
        if rc != 0:
            return rc
        rows = read_jsonl(data / "index.jsonl")
        append_jsonl(work / "failures.jsonl", [{"bundle_id": r["bundle_id"]} for r in rows])
        if run_counterfactual(data, work / "failures.jsonl", report) != 0:
            return 1
        complete = [
            s
            for s in read_json(report / "suites.json")["suites"]
            if all(t["status"] in ("certified", "text") for t in s["twins"].values())
        ]
        if len(complete) < len(PATTERNS):
            print(f"only {len(complete)} items have every twin certified")
            return 1
        shutil.rmtree(OUT, ignore_errors=True)
        (OUT / "images").mkdir(parents=True)
        options = {
            r["bundle_id"]: read_json(data / r["dir"] / "task.json")["question"]["options"]
            for r in rows
        }
        suites, results = [], []
        for number, (suite, (verdict, pattern)) in enumerate(
            zip(complete, PATTERNS, strict=False), 1
        ):
            bid = suite["bundle_id"]

            def keep(kind: str, bundle: str = bid) -> str:
                name = f"{bundle}_{kind}.jpg"
                shutil.copyfile(report / "images" / name, OUT / "images" / name)
                return (OUT / "images" / name).relative_to(REPO).as_posix()

            suite["image"] = keep("original")
            for kind, twin in suite["twins"].items():
                if twin.get("image"):
                    twin["image"] = keep(kind)
            replies: dict[str, str] = {}
            for kind, solved in pattern.items():
                truth = str(suite["twins"][kind]["answer"]["canonical"])
                replies[kind] = truth if solved else wrong_label(truth, options[bid])
                results.append(
                    {"bundle_id": bid, "kind": kind, "reply": replies[kind], "sample": True,
                     "model": SAMPLE_MODEL}
                )  # fmt: skip
            original = wrong_label(str(suite["answer"]["canonical"]), options[bid])
            suite |= {
                "sample": True,
                "sample_title": f"Sample {number}",
                "replies": {SAMPLE_MODEL: original},
            }
            got = present.diagnose(present.grade_twins(suite, replies))[0]
            if got != verdict:
                print(f"{bid}: expected {verdict}, graded {got}")
                return 1
            suites.append(suite)
        write_json(OUT / "suites.json", {"sample": True, "suites": suites, "missing_bundles": []})
        append_jsonl(OUT / "results.jsonl", results)
        (OUT / "README.md").write_text(README, encoding="utf-8", newline="\n")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"demo forensics: {len(suites)} sample items, {size / 1024:.0f} KB in {OUT}")
    print(json.dumps([s["bundle_id"] for s in suites]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
