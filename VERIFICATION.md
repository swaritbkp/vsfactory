# Verification

What was checked, how, and what is still open. Gate reports are written by
`uv run python scripts/verify_phase.py --phase N` to `artifacts/gates/phase<N>.json`. `artifacts/` is
not committed, so this page records their results.

## Automated checks

| Check | Result | Evidence |
|---|---|---|
| GitHub Actions `ci / gates` (Ubuntu, offline, no API keys) | **green** on `f83c8a7`: ruff, mypy (strict kernel), import isolation, full pytest, seed known-answer test, Phase 0 gate, `knot mutate --fault all --n 2` | `.github/workflows/ci.yml` |
| Full offline test suite (local, Windows) | 346 passed, 1 skipped (POSIX-only socket test) | `pytest -q` |
| Kernel isolation | 819 / 850 executable LOC; import-linter 3 contracts kept; AST scan clean | gate `core.*` |
| Phase 2 gate | PASSED: exact base counts, rejection ≤ 10 %, determinism across worker counts, valid twins, exact gaslight counts | `phase2.json` |
| Fault injection M1–M10 | 40 / 40 injections caught with the exact expected code (4 per fault), 0 / 40 clean controls rejected. That report predates the Phase 3 sampler changes; on current code the matrix is re-run by `tests/pipeline/test_mutate.py` and the CI mutate step | `mutate_report.json` |
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

- **Phase 3 `p3.probes`**: shortcut-probe residuals in tangle T1/T3 and the premise families (cause
  analysed; human decision pending).
- **Phase 5 `p5.bench_mock`**: tangle staircases for two mock models miss the ±1.5-level tolerance.
  The fixed staircase rule is too coarse for shallow psychometric curves; all other Phase 5 checks
  pass.
- **Phase 4 gate record**: `phase4.json` still shows a scaffold bug (underscore in an object name) that
  was fixed afterwards. The fix is covered by `tests/scaffold` and the full suite; the Phase 4 gate
  itself was not re-run.
- **Live staircases** sit at the floor (every model fails at 1–2 crossings), so they bound rather than
  measure thresholds; per-model twin drops are directional at n = 24.
