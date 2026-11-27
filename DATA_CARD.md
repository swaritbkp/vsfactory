# Data card: Knotwork synthetic visual reasoning tasks

**What it is.** Procedurally generated `(image, question, answer, evidence)` bundles in four families of
scenes: **rack_patch** (a 2U switch / patch-panel loom of 8–14 cables, added in v2), **tangle** (cables
through braided crossings), **gauge** (analog dials) and **panel** (cables from sockets to gauges). Eleven question families: trace (T1), count crossings (T2), which cable is on
top (T3), read a gauge (G1), gauge zone (G2), and panel compositions (P1 read-after-trace, P2
zone-after-trace, P3 argmax, P4 count, P5 true-premise control, T_GASLIGHT false premise).

**How labels are guaranteed.** Each answer is computed twice and independently: symbolically by the
scene generator (D1), and by an import-isolated trusted kernel (`knotcore`) that never reads the
generator's internal state (D3). The kernel re-derives every answer from renderer object masks,
cross-checked against the visible image via a beauty-only sanity oracle, colourblindness (CVD) gate,
and stray-ink witness. A bundle is emitted only if
D1 = D3, every plugin claim re-checks against the pixels, and every witness and legibility check passes
(strokes, crossing angles and labels checked at the 768-px consumption resolution). Everything else is
quarantined with an exact error code.

**Contents (as generated in this repository).**

| Set | Size | Notes |
|---|---|---|
| Benchmark pool (`split=test`, seed 101) | 600 base items per plugin | monochrome cables; rejection rate 0.5 % tangle, 0.7 % gauge, 6.1 % panel |
| Golden selftest (seed 1234) | 2 items per family + twins = 66 bundles | regenerated and re-verified on every `knot selftest` |
| Twins | causal (minimal edit, answer changes) and nuisance (restyle, answer kept) per base item | causal < 5 % of pixels, nuisance ≥ 2 % |

**Formats.** Native bundles (PNG beauty, 16-bit id buffer, masks, manifest, claims, `task.json` with the
kernel certificate); Harbor task folders with a stdlib-only verifier (oracle 1.0, wrong 0.0, missing
0.0); HuggingFace parquet; RLVR jsonl with the kernel's step trace and a `grade_prediction` reward.

**Freshness and contamination controls.** Generation is seeded from a public beacon (drand / NIST) or an
offline counter and recorded in a freshness receipt; bundles are content-addressed by SHA-256; 20 % of
`test` / `sealed` answers are replaced by salted commitments on export; items sent to a paid API are
logged in a burn ledger and excluded from future test splits; every record carries a
`KNOTWORK-CANARY-…` string.

**Live evaluation so far.** Three VLMs, 943 paid calls, 1,069 trials, $1.12 total (details and
caveats in `README.md`). The premise questions now have a matched control: 24 `T_GASLIGHT` scenes
re-certified as `P5_PREMISE_CONTROL` with only the claimed zone word changed. No model's "yes" rate
differed detectably between true and false premises (GPT-5.4 0 vs 0 of 24, Gemini 2.5 Flash 15 vs 12,
Qwen3-VL-235B 23 vs 24; exact McNemar p = 1.00 / 0.45 / 1.00). A false-premise acceptance rate on its
own cannot separate response bias from premise checking: report it beside a true-premise control.
Abstention is not measured.

**Intended use.** Evaluating and diagnosing vision-language models on visual tracing, gauge reading and
compositional / premise-checking questions, and as verifiable-reward data for RL fine-tuning.

**Not intended for.** Claims about natural-image perception, reading real industrial instruments, or
safety-critical decisions.

**Limitations.**
- Synthetic 2-D scenes with a photographic finish; no perspective, occlusion by real objects or
  real-world clutter. English prompts only.
- Shortcut audit (`BLOCKERS.md` 3.6): the v1 leaks in tangle T1/T3 and in the premise families pass in
  v2 (randomised zone order, balanced deciles, shuffled slots, D-39). One row still beats chance: panel
  P1_READ_TRACE image statistics, 0.173 vs a 0.164 limit. All other probe, balance and duplicate
  checks pass.
- `rack_patch` was added after the live run, so it has no live VLM results yet (offline mock sessions
  only).
- Raster bits are reproducible per platform; across platforms answers and verdicts reproduce but bundle
  ids may differ (SPEC D-35).

**License.** Code and generated data: Apache-2.0. Bundled font: DejaVu Sans (Bitstream Vera license).

**Canary.** Do not train on any document containing a `KNOTWORK-CANARY-` string.
