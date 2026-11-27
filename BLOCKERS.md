# BLOCKERS and documented limitations

**Summary.** No blocker stops the factory. Generation, certification, verification, export, CI and the
live benchmark all work. The entries below are documented engineering limitations: each names
the check affected, what was tried, the root cause, and the decision it needs. No gate threshold was
relaxed to hide any of them (CLAUDE.md law 8).

| # | Limitation | Affects | Status |
|---|---|---|---|
| 3.6 | Shortcut-probe residuals | gate `p3.probes` only | v1 residuals pass in v2; open on one new row (panel P1 image stats) |
| 5.6 | Mock staircase and error recovery | gate `p5.bench_mock` | passes in v2; one panel staircase sits at the 1.5-level limit |
| L-1 | Live staircases at the floor | interpretation of live thresholds | documented in README, not a gate |

Claude Code appends here when a task fails verification 3 times (see CLAUDE.md section 4).
Format per entry:

```
## <task id> <task title>
- Verification command: `<exact command>`
- Last failure (tail): <10 lines max>
- Attempts: 1) <hypothesis + change> 2) <...> 3) <...>
- Suspected root cause: <one sentence>
- Proposed human decision: <one concrete question>
- Status: OPEN
```


## 3.6 Shortcut-probe residuals (gate `p3.probes`)
- Verification command: `uv run python scripts/verify_phase.py --phase 3 --keep-going`
- Last failure (tail):
  `tangle/gbdt_image_stats: wilson_upper 0.328 > chance 0.273+0.05; tangle/text_only: wilson_upper 0.329 > chance 0.273+0.05;
  tangle/T1_TRACE/text_only: acc 0.240 > chance 0.180+0.053; tangle/T3_ONTOP/gbdt_image_stats: acc 0.596 > chance 0.500+0.070;
  panel/text_only: wilson_upper 0.276 > chance 0.206+0.05; panel/P5_PREMISE_CONTROL/text_only: acc 0.411 > chance 0.128+0.069;
  balance panel/P5_PREMISE_CONTROL/bins: p=2e-06 < 0.00125; balance panel/T_GASLIGHT/bins: p=0.000309 < 0.00125`
  Every other Phase 3 gate passes (pytest 299 passed, selftest, verify + tamper, freshness, Vendi 3.7x / 2.1x / 2.9x,
  dedup tangle 0.002 / gauge 0.010 / panel 0.000).
- Attempts: 1) pHash near-duplicates (gauge 1.36, panel 0.22 pairs/item) -> layout variety in gauge/panel/tangle
  render code (D-27); dedup now passes. 2) G1 decile imbalance (p = 4.5e-5) -> decile-balanced READ values (D-28);
  G1/P1 balance now passes. 3) Sampling-noise imbalance exploited by text_only / constant predictors -> exact
  answer balancing from a per-family slot ordinal (D-34); gauge and all zone/T2/T3 balance tests now pass.
  Stopped here per the owner's instruction not to loop on re-runs (each split regeneration is ~35 min).
- Suspected root cause: two separate things.
  (a) Premise families (P5_PREMISE_CONTROL, T_GASLIGHT): a SPEC conflict, not a sampler bug. SPEC 8.2 fixes the
      zone geometry (red always in the top 1.5-3 deciles) and SPEC 8.3 states the zone in the question text, so a
      true premise reveals the value's decile from text alone (P5 text_only 0.411). Making value deciles uniform
      (needed for the SPEC 11.2 numeric chi-square) forces non-uniform zones, and then the claimed zone leaks
      premise truth instead; both cannot hold under the fixed zone ranges.
  (b) Tangle: T1 text_only (0.240 vs 0.180) and T3 gbdt_image_stats (0.596 vs 0.500) still clear chance after
      exact balancing; not yet diagnosed (possible residual correlation between the ordinal-driven choices and
      n / image density, or probe over-fit within small per-n groups).
- Proposed human decision: for (a), choose one of: (i) exclude the premise families' value decile from the
  numeric balance test and score premise probes on the yes/no token only; (ii) define premise chance as
  1/2 x 1/(deciles spanned by the claimed zone) (D-29 amendment); (iii) relax SPEC 8.2 zone ranges to equal
  thirds (still leaves a text hint for true premises). For (b), may I spend one more ~35-min split regeneration
  investigating the tangle residual after Checkpoint 2?
- v2 change (D-39, 2026-11-26): (a) premise scenes now randomise zone colour order and balance value
  deciles; premise families are scored as one yes/no task; trivial baselines added. (b) T1 / T3 slot
  order is now shuffled by a run-seeded permutation.
- v2 measurement (2026-11-26, gate function `gate_probes` on a fresh 600-per-plugin `test` split,
  seed 101, artifacts under `artifacts/v2_gates/`): the residuals listed above now pass.
  tangle/T1 text_only 0.176 vs chance 0.180 (limit 0.232); tangle/T3 gbdt_image_stats 0.495 vs 0.500
  (limit 0.570); premise yes/no text_only 0.444 and gbdt_image_stats 0.494 vs 0.500 (limit 0.573;
  the premise target and chance changed in D-39, so these do not compare with the v1 numbers);
  pooled Wilson upper bounds: tangle text_only 0.293 <= 0.323, panel text_only 0.316 <= 0.370;
  all 9 balance tests pass (premise bins p = 0.9999 / 0.9945); trivial baselines: no violation;
  dedup rates tangle 0.0017, gauge 0.0033, panel 0.0100 (the limit is 0.01).
- v2 remaining failure: panel/P1_READ_TRACE gbdt_image_stats 0.173 (26 of 150) > chance 0.113 + 0.051.
  This family was not part of the v1 blocker. Not diagnosed.
- Status: OPEN. Gate `p3.probes` FAILS on that single row. Probe thresholds are unchanged.

## 5.6 Mock staircase tolerance (gate `p5.bench_mock`)
- Verification command: `uv run python scripts/verify_phase.py --phase 5 --keep-going`
- Last failure (tail): `mock_a/tangle: staircase 6.33 vs 12.00; mock_b/tangle: staircase 7.00 vs 4.99`
  Everything else in the Phase 5 gate passes: pytest 344 passed / 1 skipped; per-crossing MLE recovers p for all
  three mocks (0.047 / 0.076 / 0.173 vs 0.04 / 0.08 / 0.15, true p inside the CI); composition gap CI contains 0
  for all mocks; panel staircases within 1.5 levels; replay offline (0 network calls); preflight $5.06 <= $8.
- Attempts: none on the gate itself (by the owner's instruction: log borderline spec tolerances and pause).
  Pre-implementation simulation of the exact SPEC 12.2 rule (start 3, 1-up-2-down, stop at 10 reversals or 50
  trials, mean of last 6 reversals) over 400 seeded runs: within 1.5 levels of k* in 26 % (p = 0.04, m = 6),
  ~65 % (p = 0.08, m = 6) and ~92 % (p = 0.15) of runs -- the rule is too short / too coarse for shallow
  psychometric curves, and it stops early while still climbing from level 3, biasing thresholds low.
- Suspected root cause: SPEC 12.2 staircase parameters vs. the mock_a / mock_b `p_cross` values in
  `configs/models.yaml` and the 1.5-level tolerance of gate `p5.bench_mock`; not an implementation bug (task 5.1's
  calibration test with a steeper observer passes in >= 90 % of 200 runs).
- Proposed human decision: pick one: (i) more staircase trials for mocks (e.g. `max_trials` 150 / 20 reversals);
  (ii) widen `STAIRCASE_LEVEL_TOL` for the tangle track; (iii) steeper mock `p_cross` (e.g. 0.10 / 0.15 / 0.25);
  (iv) accept the staircase as a reported (not gated) diagnostic for mocks.
- v2 change (D-39, 2026-11-26): coarse-then-fine steps, up to 75 trials, logistic maximum-likelihood
  `k*`. Replay fixtures rebuilt.
- v2 measurement (2026-11-26, gate function `gate_bench_mock` on the same split, seed 5): all 6 mock
  staircases are within 1.5 levels (estimate vs expected): mock_a tangle 12.00 / 11.58, panel
  5.68 / 7.00; mock_b tangle 4.92 / 4.92, panel 5.65 / 6.94; mock_c tangle 3.09 / 2.56, panel
  2.45 / 3.41 (v1 failed 2 of 6).
- v2 remaining failure: mock_a per-crossing error estimate 0.013, 95 % CI 0.007-0.020, true 0.04.
  Cause: items are re-asked when a level has few of them. mock_a's tangle staircase spent 75 trials
  on 18 distinct items (one 16 times, at level 12), and repeats of a deterministic model are counted
  as independent trials. The mock_a tangle staircase pass is therefore weak evidence (it sat at the
  top level).
- v2 fix (D-46, 2026-11-26): every staircase trial asks an item not asked before (nearest level with
  an unused item; the level shown is what the fit uses) and the per-crossing estimate counts each
  distinct request once.
- v2 re-measurement (same split, gate function `gate_bench_mock`): PASS. All staircase trials are
  distinct. Per-crossing estimates: mock_a 0.034 (CI 0.022-0.047, true 0.04), mock_b 0.071
  (0.050-0.098, true 0.08), mock_c 0.155 (0.123-0.192, true 0.15). Staircase estimate / expected:
  mock_a tangle 12.00 / 11.58, panel 5.50 / 7.00; mock_b tangle 4.92 / 4.92, panel 5.50 / 6.94;
  mock_c tangle 3.11 / 2.56, panel 2.45 / 3.41.
- Caveat: mock_a panel passes exactly at the 1.5-level limit (mock_b panel at 1.44). The split holds
  only 3 panel items with 6 crossings and 1 with 7, so shallow panel curves are data-limited.
- Status: RESOLVED for gate `p5.bench_mock` (the full `--phase 5` run was not repeated).
  `STAIRCASE_LEVEL_TOL` is unchanged.

## L-1 Live staircases sit at the floor (not a gate)
- Observation: on the live run every model fails at 1-2 crossings; Gemini and Qwen dropped from the start
  level (3) to level 1 on the tangle track and never climbed back (no reversals), so thresholds of 1.1-1.7 crossings bound rather than measure ability.
  The per-crossing error MLE (0.80 to ~1.0) and the composition product law are uninformative at this
  floor. Per-model twin drops are directional (n = 24, exact McNemar p = 0.125 / 0.25 / 0.375).
- Handling: stated as caveats in README ("Full live results table and caveats"), VERIFICATION.md and the figure subtitles.
  An easier level ladder (0-crossing items) would be needed to place thresholds for current models.
- Status: documented; no code change.
