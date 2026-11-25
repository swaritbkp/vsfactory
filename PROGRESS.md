# PROGRESS — Knotwork build log

Repository: https://github.com/swaritbkp/vsfactory (public). Environment: Windows 11, `uv` 0.11,
Python 3.11.15 (`.venv`), offline only (no `OPENROUTER_API_KEY` set; paid runs capped at $5 by the owner
until the Phase 5 budget review).

| Phase | Scope | Gate | Commit |
|---|---|---|---|
| 0 | bootstrap, layered renderer, contracts, isolation, seeds | PASSED | `28991f3` |
| 1 | knotcore trusted pixel kernel (819 / 850 executable LOC, 1218 / 1350 total) | PASSED | `8a18d93` |
| 2 | tangle / gauge / panel plugins, generate, twins, mutate M1–M10, contact sheet | PASSED | see git log |
| 3 | freshness, probes, diversity, golden selftest, file-only verify | FAILED on `p3.probes` only (BLOCKERS 3.6) | see git log |
| 4 | Harbor / HF / RLVR exporters, `new-plugin` scaffold, CI | all p4 gates PASS; pytest fix re-verified in Phase 5 run (4.5 not re-run as a whole) | see git log |
| 5 | staircase, mock / replay / OpenRouter clients, budget, stats, `knot bench` (offline) | FAILED on `p5.bench_mock` staircase tolerance only (BLOCKERS 5.6) | see git log |

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

## Phase 2 visual-realism upgrade (pre-Checkpoint 1)
Gate re-run: PASSED, knotcore untouched (819 / 850 executable LOC).
* Renders: seeded photographic finish (lighting, powder-coat/brushed grain, smudges, dust, sensor grain);
  bevelled metal bezels, aged enamel faces, dense minor/medium ticks, counterweighted pointer with
  contact shadow and pivot cap, curved-glass glare and rim grime; cylindrical cable jackets with drop
  shadows, heat-shrink sleeves and ferrules; panel enclosure with screws, placards and a DIN rail.
* Gate: 254 tests; first-try rejection tangle 0.000 / gauge 0.000 / panel 0.024; 90 valid twin triples;
  determinism across worker counts; M1-M10 caught with exact codes (clean controls 0.000 rejected).
* Contact sheet 2314x3780 with wrapped captions. Decisions D-24 ... D-26.

## Phase 3 (gate: all PASS except `p3.probes`)
* pytest (phases 0-3) 299 passed; golden selftest 10/10 checks, byte-identical reports; verify 66/66 clean, pixel and
  answer tampering -> E_DIGEST_MISMATCH; freshness receipt + canary propagation; Vendi ratio tangle 3.73x, gauge 2.13x,
  panel 2.95x; pHash dup rate tangle 0.002, gauge 0.010, panel 0.000; knotcore untouched (819 / 850).
* Sampler fixes in untrusted code only: layout variety (D-27), decile-balanced reads (D-28), exact answer balance
  from a per-family slot ordinal (D-34). Remaining probe failures: premise-family SPEC conflict + tangle T1/T3 residual
  (BLOCKERS 3.6). Decisions D-27 ... D-34.

## Phase 4
* Harbor: 66 tasks, oracle 1.0 / wrong 0.0 / missing 0.0 on 20 via the real solve.sh / test.sh; HF parquet 66 rows,
  RLVR oracle 1.0 / wrong 0.0; scaffold compiles, passes its 3 tests, knotcore sha unchanged; ci.yml steps present
  (job timeout raised to 150 min for the larger suite).

## Phase 5 (offline only; no API key, zero network calls)
* pytest (phases 0-5) 344 passed, 1 skipped. Mock bench on the 600/plugin split: per-crossing MLE p_hat 0.047 / 0.076 /
  0.173 for true 0.04 / 0.08 / 0.15 (CIs contain truth); composition gap CIs contain 0; panel staircases within
  tolerance; tangle staircases for mock_a / mock_b outside +-1.5 levels (BLOCKERS 5.6). Replay of committed fixtures:
  0 network calls. Preflight (default plan, 50 trials): $5.06 for 1260 calls; $5 option (`--max-trials 40`): $4.05
  for 1008 calls. Stopped at [HUMAN-CHECKPOINT-2].

## Phase 5 live run ([HUMAN-CHECKPOINT-2] approved by the owner)
* Plan: `--max-trials 40`, three slots (gpt-5.4 with reasoning off after the canary, gemini-2.5-flash,
  qwen3-vl-235b). Two canaries $0.0501 + full run $0.9992 = **$1.0493** of the $5.00 cap; 871 paid calls,
  997 trials, 0 HTTP errors, 2 parse failures. Offline replay reproduces every per-model result exactly
  with 0 network calls. Summary: `README.md` "Live benchmark results"; figures `docs/figures/fig01-06`.

## Phase 6 (partial)
* CI fix for Ubuntu (D-35): platform-neutral replay keys, shipped derived fixture renders,
  platform-tagged golden. `report/summary.py`, `report/figures.py` (6 of the 11 required figures),
  `knot report` (self-contained HTML). Not done yet: 6.1 (11 figures), 6.2 inspector, 6.4-6.6 docs/deck.
