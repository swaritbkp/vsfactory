# Verification

What was checked, how, and what is still open. Gate reports are written by
`uv run python scripts/verify_phase.py --phase N` to `artifacts/gates/phase<N>.json`. `artifacts/` is
not committed, so this page records their results.

## Automated checks

| Check | Result | Evidence |
|---|---|---|
| GitHub Actions `ci` (Ubuntu, offline, no API keys), every push and PR | **green** on `feat/vtf-v2`: `tier0` (ruff, ruff format, import-linter, mypy strict kernel, seed known-answer test, tier0 pytest) and `tier1` (plugins, pipeline, faults, Studio and Model Arena on shared fixtures) run in parallel, under 2 minutes wall time | `.github/workflows/ci.yml` |
| GitHub Actions `nightly` (scheduled on `main`, or manual) | `tier2` pytest (bulk generation, probes, bench, 10-render `rack_patch` yield gate, background-runner resume) and the phase gates (`verify_phase.py --phase 0`, `knot mutate`, `knot selftest`, Phase 4 gate) | `.github/workflows/nightly.yml` |
| Kernel isolation (TCB contract, SPEC D-38) | import-linter 3 contracts kept; TCB audit clean (allow-listed imports, no network / process / dynamic-code modules, no `SceneIR.payload` access, deterministic `kernel_sha256`). The 850 / 1350 LOC budgets are informational telemetry since D-38 (currently 1153 executable / 1576 total) | gate `core.ast_isolation`, `tests/conformance/test_kernel_isolation.py` |
| Phase 2 gate | PASSED: exact base counts, rejection ≤ 10 %, determinism across worker counts, valid twins, exact gaslight counts | `phase2.json` |
| Fault injection M1–M14 (v2, 2026-11-26) | 56 / 56 injections caught with the exact expected code (4 per fault, M11–M14 are the panel faults of D-39), 0 / 56 clean controls rejected. Re-run on current code by `tests/pipeline/test_mutate.py` (tier2) and the nightly mutate step | `knot mutate --fault all --n 4` → `artifacts/mutate/report.json` |
| Real render defect caught by the kernel | crowded panel crossings clipped the top cable's halo: 4 / 120 clean panels failed the crossing-colour check; after the renderer fix 0 / 400 | commit `aaf2528` (measurement in its message) |
| `rack_patch` yield (unmodified kernel) | 10 / 10 certified on the 10-render gate set, 8 / 10 on the first attempt (gate ≥ 80 %); tier1 keeps a 3-render first-attempt check | `tests/arena/test_rack_patch.py` (tier2), `tests/arena/test_rack_patch_fast.py` (tier1) |
| Golden selftest | 66 bundles regenerated, answers + verdicts + kernel hash match the golden; report byte-identical across runs | gate `p3.selftest` |
| File-only `knot verify` | 66 / 66 clean; a 1-bit id-buffer change and an edited answer are both caught (`E_DIGEST_MISMATCH`); imports no plugin code | gate `p3.verify` |
| Diversity (Vendi, 256 items) | 3.73× tangle, 2.95× panel, 2.13× gauge over a recolour-only baseline (gate ≥ 2×) | gate `p3.diversity` |
| Exporters | Harbor: 66 tasks, oracle 1.0 / wrong 0.0 / missing 0.0 on 20; HF parquet 66 rows; RLVR oracle 1.0 / wrong 0.0 | gates `p4.export_harbor`, `p4.hf_rlvr` |
| Plugin scaffold | `knot new-plugin` output compiles, passes its 3 tests, kernel hash unchanged | `tests/scaffold` |
| Bench harness (offline) | mock VLMs: per-crossing error recovered (0.047 / 0.076 / 0.173 vs true 0.04 / 0.08 / 0.15); composition-law CI contains 0; committed fixtures replay with 0 network calls | gate `p5.*`, `tests/bench` |

## Live benchmark (human-approved, Checkpoint 2)

Three VLMs via OpenRouter. Totals: **$1.12, 943 paid calls (canary calls not counted), 1,069 trials,
0 HTTP errors, 2 unparsable replies**.

| Run | Cap | Paid calls | Trials | Cost |
|---|---|---|---|---|
| Two canaries | $5.00 (shared) | not counted above | none | $0.0501 |
| Main run | $5.00 (shared) | 871 | 997 | $0.9992 |
| True-premise control (SPEC D-37, owner-run) | $0.50 | 72 | 72 | $0.0756 |

An offline replay of the cached responses reproduces every per-model statistic of the main run exactly
with 0 network calls. Results and caveats are in `README.md`. Raw outputs: `samples/live_run/`.

### True-premise control

| Check | Result |
|---|---|
| Matched pairs certified (source re-verifies from files, original premise false, control `CERTIFIED` as `P5`, kernel ZONE = claimed zone, kernel says "yes", identical pixels) | 24 / 24 on every check |
| Request parity between arms (temperature, max tokens, reasoning setting, seed, model slug) | identical for all three models |
| Control trials / missing / HTTP errors / unparsable | 72 / 0 / 0 / 0 |
| Offline replay of the control | same replies, grades and statistics, 0 network calls |
| "Yes" on true vs false premise (of 24; exact McNemar p) | GPT-5.4 0 vs 0 (1.00); Gemini 15 vs 12 (0.45); Qwen 23 vs 24 (1.00) |

Finding: no model detectably discriminated true from false premises. The earlier false-premise rates
(0 / 50 / 100 %) are therefore not evidence of premise grounding. Abstention is not measured. The
control's code was checked locally with ruff, mypy, import-linter and `tests/bench` plus
`tests/apps/test_report.py`; the full suite, phase gates and CI above predate it.

## Open items (documented limitations, `BLOCKERS.md`)

- **Phase 3 `p3.probes`** (open): the v1 residuals in tangle T1/T3 and the premise families pass in
  v2; one new row fails: panel P1_READ_TRACE image statistics 0.173 vs limit 0.164 (chance 0.113 +
  0.051). Thresholds unchanged.
- **Phase 5 `p5.bench_mock`** (resolved for the gate function): passes with a distinct item on every
  staircase trial (D-46); mock_a panel sits exactly at the 1.5-level limit because the split has few
  panel items with 6 or more crossings. The full `--phase 5` run was not repeated.
- **Phase 4 gate record**: `phase4.json` still shows a scaffold bug (underscore in an object name) that
  was fixed afterwards. The fix is covered by `tests/scaffold` and the full suite; the Phase 4 gate
  itself was not re-run.
- **Live staircases** sit at the floor (every model fails at 1–2 crossings), so they bound rather than
  measure thresholds; per-model twin drops are directional at n = 24.
