"""Grounded vCoT traces on real certified bundles of all four domains."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from PIL import Image

from knotwork.knotcore.contracts import RenderManifest
from knotwork.posttrain import vcot
from posttrain_fixtures import DOMAINS, FIXTURES, bundle_copy, task

COT_RE = re.compile(r"^<think>\n(Step \d+ \((perception|transition|conclusion)\): .+ "
                    r"<box>\[\d+, \d+, \d+, \d+\]</box>\n)+</think>\n\\boxed\{[^{}]+\}$")  # fmt: skip


@pytest.mark.parametrize("domain", DOMAINS)
def test_trace_is_grounded_in_bounds_and_well_formed(domain: str) -> None:
    trace = vcot.extract(FIXTURES / domain)
    t = task(domain)
    assert trace.domain == domain and trace.item_id == t["bundle_id"]
    assert trace.image_size == [1024, 1024] and len(trace.steps) >= 3
    assert [s.step_idx for s in trace.steps] == list(range(1, len(trace.steps) + 1))
    assert trace.steps[0].phase == "perception" and trace.steps[-1].phase == "conclusion"
    assert all(s.phase != "conclusion" for s in trace.steps[:-1])
    for s in trace.steps:
        x0, y0, x1, y1 = s.bbox_px
        assert 0 <= x0 < x1 <= 1024 and 0 <= y0 < y1 <= 1024, s
        assert all(0 <= v <= 1000 for v in s.bbox_norm_1000)
        assert s.bbox_norm_1000[0] < s.bbox_norm_1000[2] and s.text.endswith(".")
    assert trace.final_answer == t["answer"]["canonical"]
    assert f"Final answer: {trace.final_answer}." in trace.steps[-1].text
    assert COT_RE.match(trace.formatted_cot), trace.formatted_cot
    assert trace.formatted_cot.endswith("\\boxed{" + trace.final_answer + "}")
    assert vcot.extract(FIXTURES / domain) == trace  # deterministic


@pytest.mark.parametrize("domain", ["rack_patch", "tangle", "panel"])
def test_trace_starts_on_the_start_terminal_and_ends_on_the_end(domain: str) -> None:
    trace = vcot.extract(FIXTURES / domain)
    step = task(domain)["certificate"]["steps"][0]
    man = RenderManifest.model_validate_json((FIXTURES / domain / "manifest.json").read_text())
    boxes = {t.terminal_id: t.box for t in man.terminals}

    def inside(term: str, box: list[int]) -> bool:
        b = boxes[term]
        return box[0] <= b.x0 and box[1] <= b.y0 and box[2] >= b.x1 and box[3] >= b.y1

    assert inside(step["inputs"][0], trace.steps[0].bbox_px)
    end = next(s for s in trace.steps if s.evidence_label.startswith("end "))
    assert inside(step["evidence"]["end_terminal"], end.bbox_px)
    hops = [s for s in trace.steps if s.evidence_label.startswith("crossing")]
    assert hops and all(re.search(r"passes (over|under)", s.text) for s in hops)


def test_gauge_trace_reads_bezel_scale_needle_then_value() -> None:
    trace = vcot.extract(FIXTURES / "gauge")
    assert [s.evidence_label.split()[0] for s in trace.steps] == ["dial", "scale", "needle",
                                                                  "reading"]  # fmt: skip
    dial = trace.steps[0].bbox_px
    for s in trace.steps[1:]:  # scale, needle and reading all sit on the dial
        assert dial[0] <= s.bbox_px[0] and s.bbox_px[2] <= dial[2] + 2


def test_overlay_draws_on_any_image_size(tmp_path: Path) -> None:
    bundle = bundle_copy("panel", tmp_path / "b")
    trace = vcot.extract(bundle)
    small = Image.open(bundle / "beauty.png").resize((512, 512))
    out = vcot.overlay(small, trace)
    assert out.size == (512, 512) and out.getpixel((0, 0)) != (0, 0, 0)
    assert out.tobytes() != small.convert("RGB").tobytes()


def test_bundle_without_certificate_steps_is_refused(tmp_path: Path) -> None:
    bundle = bundle_copy("gauge", tmp_path / "b")
    t = task("gauge")
    t["certificate"]["steps"] = []
    with pytest.raises(vcot.VCoTError):
        vcot.extract(bundle, t)
    assert vcot.try_extract(bundle, t) is None
