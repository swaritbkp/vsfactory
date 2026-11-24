# PROGRESS — Knotwork build log

Repository: https://github.com/swaritbkp/vsfactory (public). Environment: Windows 11, `uv` 0.11,
Python 3.11.15 (`.venv`), offline only (no `OPENROUTER_API_KEY` set; paid runs capped at $5 by the owner
until the Phase 5 budget review).

| Phase | Scope | Gate | Commit |
|---|---|---|---|
| 0 | bootstrap, layered renderer, contracts, isolation, seeds | PASSED | `28991f3` |
| 1 | knotcore trusted pixel kernel (819 / 850 executable LOC, 1218 / 1350 total) | PASSED | `8a18d93` |
| 2 | tangle / gauge / panel plugins, generate, twins, mutate M1–M10, contact sheet | PASSED | see git log |

## Phase 0
* `render/canvas.py`: 2× supersampled beauty (Lanczos to 1024), non-AA amodal masks, modal uint16 id
  buffer, per-crossing halo layering, `E_FONT` check. 7 smoke tests.
* Gate script portability fix (UTF-8 subprocess + file I/O on Windows), recorded as D-10.

## Phase 1
* `io`, `ocr` (768 px, crop-hash LRU, template fallback), `topo` (TRACE / CROSS / ONTOP), `gauge`
  (READ / ZONE / GAUGE_OF / READ_ALL), `logic`, `witness`, `legibility`, `engine`, `grading`.
* 98 kernel tests over synthetic bundles drawn with the renderer only (no plugin imports).
* Decisions D-11 … D-14 (φ wrap, KernelFault convention, OCR template details, TRACE/GAUGE_OF composition).

## Phase 2
Gate `artifacts/gates/phase2.json`: PASSED.
* Tests: 254 passed, 1 skipped (POSIX-only socket test) across conformance, kernel, plugins, pipeline.
* Generate (seed 11, twins on): tangle 30 base (false-quarantine 0.032), gauge 20 (0.000), panel 40 (0.024);
  90 base items each with a valid causal twin (answer flips, < 5 % pixels) and nuisance twin (answer kept).
* Determinism: identical bundle ids across runs and worker counts; disjoint ids for a new seed.
* Panel: 6 gaslight + 6 true-premise controls of 40 (exact round_half_up(0.15 n)).
* Mutate M1–M10: 100 % caught with the exact expected code (4 injections each); clean false-quarantine 0.05.
* Contact sheet `artifacts/phase2/contact_sheet.png` (2290x3680) -> [HUMAN-CHECKPOINT-1].
* Decisions D-15 … D-23 in SPEC section 16.
