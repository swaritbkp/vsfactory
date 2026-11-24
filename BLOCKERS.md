# BLOCKERS

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


## 3.6 Benchmark split and shortcut tuning (gate `p3.probes`)
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
- Status: OPEN

## 5.6 Phase 5 gate (gate `p5.bench_mock`, staircase tolerance only)
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
- Status: OPEN
