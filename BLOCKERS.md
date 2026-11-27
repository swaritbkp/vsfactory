# Open items and engineering defects

This register lists every known defect, failing gate and interpretive limit on `main`. Each entry
gives its status, the check it affects, the root cause as far as it is known, what was tried, and
the decision it still needs. No gate threshold has been relaxed to clear any entry (CLAUDE.md law 8).

None of these items stops generation, certification, file-only verification, export, the Studio,
Model Arena, the post-training pack or CI.

| ID | Item | Affects | Status |
|---|---|---|---|
| 3.6 | Shortcut-probe residual on `panel` P1 | gate `p3.probes` | Open |
| 5.6 | Mock staircase tolerance | gate `p5.bench_mock` | Resolved for the gate function; full Phase 5 run not repeated |
| G-1 | Phase 3–5 gate records predate current code | gate `p6.docs` | Open |
| L-1 | Live staircases sit at the accuracy floor | interpretation of live thresholds | Documented; not a gate |
| L-2 | One failed API call in the v2 live audit | 1 of 531 v2 live trials | Documented; not retried |

## 3.6 Shortcut-probe residual (gate `p3.probes`)
- Status: OPEN. Gate `p3.probes` fails on one row. Probe thresholds are unchanged.
- Verification command: `uv run python scripts/verify_phase.py --phase 3 --keep-going`
- Current failure: `panel/P1_READ_TRACE gbdt_image_stats` accuracy 0.173 (26 of 150) against a limit
  of 0.164 (chance 0.113 + 0.051). Measured 2026-11-26 with gate function `gate_probes` on a fresh
  600-per-plugin `test` split, seed 101. This family was not part of the v1 failure.
- Everything else in the probe gate passes on that split:
  - tangle T1 `text_only` 0.176 against chance 0.180 (limit 0.232);
  - tangle T3 `gbdt_image_stats` 0.495 against 0.500 (limit 0.570);
  - premise yes/no `text_only` 0.444 and `gbdt_image_stats` 0.494 against 0.500 (limit 0.573);
  - pooled Wilson upper bounds: tangle `text_only` 0.293 ≤ 0.323, panel `text_only` 0.316 ≤ 0.370;
  - all 9 balance tests (premise bins p = 0.9999 / 0.9945);
  - trivial baselines;
  - near-duplicate rates: tangle 0.0017, gauge 0.0033, panel 0.0100 (limit 0.01).
- Root cause: not diagnosed. A GBDT on global image statistics predicts the P1 reading slightly
  above chance. The likely channel is a correlation between the traced gauge's needle position and
  coarse image statistics, but this has not been confirmed.
- History:
  - v1 failed on eight rows: tangle T1/T3 and pooled `text_only`, panel `text_only`, the premise
    families, and two balance tests.
  - Three v1 attempts fixed the dedup and decile-balance checks: layout variety (D-27), decile-balanced
    READ values (D-28) and exact answer balancing from a per-family slot ordinal (D-34).
  - The premise leak came from a SPEC conflict. Fixed zone geometry (SPEC 8.2) plus the zone named in
    the question (SPEC 8.3) let the text reveal the value decile.
  - v2 (D-39) fixed that leak: zone colour order is randomised, value deciles are balanced, premise
    families are scored as one yes/no task, trivial baselines were added, and T1/T3 slot order is
    shuffled by a run-seeded permutation.
- Impact: the panel P1 family can be answered slightly above chance from image statistics alone. P1
  accuracies in reports should be read against that baseline, not against uniform chance.
- Decision needed: run one diagnostic split regeneration (about 35 minutes) to find out which image
  statistic carries the signal, then decide between a renderer fix and a family-specific baseline.

## 5.6 Mock staircase tolerance (gate `p5.bench_mock`)
- Status: RESOLVED for gate `p5.bench_mock`. The full `--phase 5` run was not repeated.
  `STAIRCASE_LEVEL_TOL` is unchanged.
- Verification command: `uv run python scripts/verify_phase.py --phase 5 --keep-going`
- Original failure (v1): `mock_a/tangle: staircase 6.33 vs 12.00; mock_b/tangle: staircase 7.00 vs
  4.99`. The SPEC 12.2 rule was too short and too coarse for shallow psychometric curves. It was
  start at 3, 1-up-2-down, stop at 10 reversals or 50 trials. In 400 seeded simulations it landed
  within 1.5 levels in only 26 % of runs at p = 0.04.
- Fixes:
  - D-39: coarse-then-fine steps, up to 75 trials, a logistic maximum-likelihood `k*`.
  - D-46: every staircase trial asks an item not asked before, and the per-crossing estimate counts
    each distinct request once. Before D-46, `mock_a` spent 75 trials on 18 distinct items and
    counted repeats of a deterministic model as independent.
- Current measurement (gate function `gate_bench_mock`, PASS):
  - Per-crossing error estimates with 95 % CI: `mock_a` 0.034 (0.022–0.047, true 0.04), `mock_b`
    0.071 (0.050–0.098, true 0.08), `mock_c` 0.155 (0.123–0.192, true 0.15).
  - Staircase estimate against expected level:
    - `mock_a`: tangle 12.00 / 11.58, panel 5.50 / 7.00
    - `mock_b`: tangle 4.92 / 4.92, panel 5.50 / 6.94
    - `mock_c`: tangle 3.11 / 2.56, panel 2.45 / 3.41
- Residual: `mock_a` panel passes exactly at the 1.5-level limit, and `mock_b` panel at 1.44. The
  split holds only 3 panel items with 6 crossings and 1 with 7, so shallow panel curves are
  data-limited.
- Impact: none on certified data. The staircase is a measurement instrument, and its panel track
  has little headroom at high crossing counts.

## G-1 Phase 3–5 gate records predate current code (gate `p6.docs`)
- Status: OPEN.
- Verification command: `uv run python scripts/verify_phase.py --phase 6 --keep-going`
- Observation:
  - `artifacts/gates/phase3.json` is `ok: false` on `core.ruff` and `p3.probes`. `core.ruff` is
    clean on current code; `p3.probes` is entry 3.6.
  - `phase4.json` is `ok: false` on `core.pytest` and `p4.scaffold`. The scaffold bug, an underscore
    in a generated object name, was fixed afterwards and is covered by `tests/scaffold`.
  - `phase5.json` is `ok: false` on `p5.bench_mock`, which is resolved in entry 5.6.
- Root cause: the records were written by v1 runs. The full Phase 3, 4 and 5 gates have not been
  re-run since the fixes.
- Impact: gate `p6.docs` requires every phase record to be `ok` (D-47), so it fails even though
  its documentation checks (`docs_problems`) pass. That requirement is deliberate and stays.
- Decision needed: schedule a full Phase 3–5 re-run (the Phase 3 gate regenerates 600 items per
  plugin). Phase 3 will still fail until 3.6 is closed.

## L-1 Live staircases sit at the floor (not a gate)
- Status: DOCUMENTED. No code change.
- Observation: in the live run every model fails at 1–2 crossings. Gemini 2.5 Flash and
  Qwen3-VL-235B dropped from the start level (3) to level 1 on the tangle track and never climbed
  back. With no reversals, the thresholds of 1.1–1.7 crossings bound ability rather than measure it.
  At this floor the per-crossing error estimate (0.80 to about 1.0) and the composition product law
  carry no information. Per-model twin drops are directional only (n = 24, exact McNemar p = 0.125 /
  0.25 / 0.375).
- Handling: stated as caveats in `README.md`, `VERIFICATION.md` and the figure subtitles.
- Next step: an easier level ladder, including 0-crossing items, would be needed to place
  thresholds for current models.

## L-2 One failed API call in the v2 live audit (not a gate)
- Status: DOCUMENTED. No code change.
- Observation: the 2026-11-27 live run of `docs/examples/full_v2_live_campaign.yaml` issued 531
  calls; 530 were scored and 1 (Gemini 2.5 Flash, `rack_patch` monochrome) ended as an API error.
  The runner recorded it, and `Retry failed API calls only` would re-ask just that call.
- Handling: not retried, so the published numbers use 530 scored trials and Gemini's
  `rack_patch` monochrome cell reports 10 complete items plus 1 partial item.
- Impact: none on certification; one fewer trial in one cell.

---

**Entry template.**

Agents append entries in this form when a task fails verification three times (CLAUDE.md section 4).

```
## <task id> <task title>
- Status: OPEN
- Verification command: `<exact command>`
- Last failure (tail): <10 lines max>
- Attempts: 1) <hypothesis + change> 2) <...> 3) <...>
- Suspected root cause: <one sentence>
- Proposed human decision: <one concrete question>
```
