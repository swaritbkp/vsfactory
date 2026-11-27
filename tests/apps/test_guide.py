"""Guided mode microcopy: complete for all five pages, short, factual, and every key the pages use
exists. Pure checks, no rendering (well under a second)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from knotwork.studio import guide

REPO = Path(__file__).resolve().parents[2]
PAGE_KEYS = ["studio", "arena", "forensics", "human_eval", "export"]
REQUIRED_TERMS = {
    "TCB (Trusted Computing Base)",
    "D1 = D3 Verification",
    "Beauty-Only Oracle",
    "CVD Gate",
    "Stray-Ink Witness",
    "Quarantine (E_* codes)",
    "Causal vs. Nuisance Twins",
    "OOD Holdout Split",
    "Wilson 95% CI",
    "Cost per Correct ($/correct)",
}
OVERCLAIMS = re.compile(
    r"rendered image alone|from the image alone|pixels alone|850 ?LOC|≤ ?850|<= ?850", re.I
)


def _all_text() -> list[str]:
    texts = [*guide.TOOLTIPS.values(), *guide.GLOSSARY.values(), guide.KERNEL_CLAIM]
    for page in guide.PAGE_GUIDES.values():
        texts += [page["summary"], page["try_first"], *page["steps"]]
    for step in guide.TOUR_STEPS:
        texts += [step["goal"], step["tip"]]
    return texts


def test_every_page_has_a_guide_and_one_tour_step_in_judge_order() -> None:
    assert list(guide.PAGES) == PAGE_KEYS
    assert [s["page"] for s in guide.TOUR_STEPS] == PAGE_KEYS
    for key in PAGE_KEYS:
        info = guide.PAGE_GUIDES[key]
        assert info["title"] == guide.PAGES[key][0]
        assert 3 <= len(info["steps"]) <= 4 and info["summary"] and info["try_first"]
        assert info["terms"] and set(info["terms"]) <= set(guide.GLOSSARY)
        assert (REPO / "apps" / guide.PAGES[key][1]).is_file()
    assert guide.next_page("studio") == "arena" and guide.next_page("export") is None
    assert guide.page_key("Model Arena") == "arena" and guide.page_key("Nope") is None


def test_glossary_and_tooltips_are_complete_and_short() -> None:
    assert set(guide.GLOSSARY) >= REQUIRED_TERMS
    long = {k: len(v) for k, v in guide.TOOLTIPS.items() if len(v) > guide.MAX_TOOLTIP}
    assert not long, long
    assert all(v == v.strip() and v.endswith(".") for v in guide.TOOLTIPS.values())


def test_microcopy_is_factual() -> None:
    hits = [t for t in _all_text() if OVERCLAIMS.search(t)]
    assert not hits, hits
    assert (
        "renderer object masks" in guide.KERNEL_CLAIM and "stray-ink witness" in guide.KERNEL_CLAIM
    )
    tour = " ".join(s["tip"] for s in guide.TOUR_STEPS)
    assert "56 / 56" in tour and "14 / 14" in tour


def test_every_tooltip_key_used_by_the_pages_exists() -> None:
    sources = [REPO / "apps" / "studio_ui.py", *sorted((REPO / "apps" / "app_pages").glob("*.py"))]
    used = {
        key
        for path in sources
        for key in re.findall(r"""guide\.tip\(\s*['"]([a-z_0-9]+)['"]""", path.read_text("utf-8"))
    }
    assert len(used) > 100
    assert not used - set(guide.TOOLTIPS), used - set(guide.TOOLTIPS)


def test_p6_docs_live_documentation_set_is_complete() -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    import verify_phase

    assert verify_phase.docs_problems(REPO) == []
    assert "docs/SUBMISSION_CHECKLIST.md" in verify_phase.LIVE_DOCS
