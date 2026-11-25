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

Three VLMs via OpenRouter, $5.00 cap: two canaries ($0.0501) and one full run ($0.9992), **$1.0493
total, 871 paid calls, 997 trials, 0 HTTP errors, 2 unparsable replies**. An offline replay of the
cached responses reproduces every per-model statistic exactly with 0 network calls. Results and caveats
are in `README.md`. Raw outputs: `samples/live_run/`.

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
