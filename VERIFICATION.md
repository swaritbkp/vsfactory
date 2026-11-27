# Verification ledger

This ledger records what has been checked on `main`, how, when, and what remains open. Phase gates
are run with `uv run python scripts/verify_phase.py --phase N` and write their reports to
`artifacts/gates/phase<N>.json`. `artifacts/` is not committed, so this page records the results.
Every row gives the command or test that reproduces it.

## 1. Kernel invariants (`knotcore`)

| Invariant | Result | How it is checked |
|---|---|---|
| Import isolation | 3 / 3 import-linter contracts kept: `knotcore` imports no other Knotwork package; `knotcore` imports no network, UI, ML-probe or scene-geometry library; `tangle` and `gauge` are mutually independent | `uv run lint-imports` (CI `tier0`); `tests/conformance/test_kernel_isolation.py` |
| TCB contract (SPEC D-38) | 0 violations: allow-listed imports only, no network / process / dynamic-code modules, no `SceneIR.payload` access | `kernel_audit.tcb_report()`; gate `core.ast_isolation` |
| Kernel hash | `kernel_sha256 = cc915c558ec021270fb01b0baab9726af53c21735be08512ed06d36bceadf42f` on `main` | `kernel_audit.tcb_report()`; pinned in the golden |
| Size (telemetry since D-38) | 1,153 executable LOC (excluding `contracts.py`), 1,576 total, across 14 modules. The original 850 / 1,350 budgets are reported, not gated. | `kernel_audit.loc_report()` |
| Types | `mypy` strict on `knotcore` | `uv run mypy` (CI `tier0`) |
| Determinism | Seed and content-address known-answer vector | `uv run knot selftest --seeds-only` (CI `tier0`) |

## 2. Label validity

| Check | Result | Reproduce |
|---|---|---|
| Fault-injection matrix M1–M14, 4 injections per fault | **56 / 56 caught** with the exact expected code; **0 / 56** clean controls quarantined. Re-run 2026-11-27 on `b92915e` in 71 s. | `uv run knot mutate --fault all --n 4` (also `tests/pipeline/test_mutate.py`, tier2, and the nightly mutate step) |
| Golden self test | **66 / 66** bundles regenerated (18 `tangle`, 12 `gauge`, 36 `panel`; seed 1234) and re-verified; answers, verdicts, digests and kernel hash match. Re-run 2026-11-27 on `b92915e` in 4 min 25 s. | `uv run knot selftest` |
| File-only verification | 66 / 66 clean. A 1-bit id-buffer change and an edited answer are both caught (`E_DIGEST_MISMATCH`). No plugin code is imported. | `uv run knot verify <dir>`; gate `p3.verify` |
| Real render defect caught by the kernel | Crowded panel crossings clipped the top cable's halo: 4 / 120 clean panels failed the crossing-colour check; 0 / 400 after the renderer fix | commit `aaf2528` (measurement in its message) |
| `rack_patch` yield with the unmodified kernel | 10 / 10 certified on the 10-render gate set, 8 / 10 on the first attempt (gate ≥ 80 %). `tier1` keeps a 3-render first-attempt check. | `tests/arena/test_rack_patch.py` (tier2), `tests/arena/test_rack_patch_fast.py` (tier1) |
| Phase 2 gate | PASSED: exact base counts, rejection ≤ 10 %, determinism across worker counts, valid twins, exact gaslight counts | `phase2.json` |

## 3. Data quality

| Check | Result | Reproduce |
|---|---|---|
| Structural diversity (Vendi, 256 certified items per plugin) | 3.73× `tangle`, 2.95× `panel`, 2.13× `gauge` over a recolour-only baseline (gate ≥ 2×) | `uv run knot diversity`; gate `p3.diversity` |
| Shortcut probes, balance, near-duplicates | All rows pass except `panel` P1_READ_TRACE GBDT image statistics, 0.173 against a 0.164 limit (`BLOCKERS.md` 3.6) | `uv run knot probe`; gate `p3.probes` |
| Exporters | Harbor: 66 tasks, oracle 1.0 / wrong 0.0 / missing 0.0 on 20; HF parquet: 66 rows; RLVR: oracle 1.0 / wrong 0.0 | gates `p4.export_harbor`, `p4.hf_rlvr`; `tests/export/` |
| Plugin scaffold | `knot new-plugin` output compiles and passes its 3 tests; kernel hash unchanged | `tests/scaffold/` |

## 4. Post-training engine

| Check | Result | Reproduce |
|---|---|---|
| Grounded vCoT extraction | 400 freshly certified bundles covering all 11 families and all 4 domains (40 base items each for `tangle`, `gauge` and `panel` with twins, plus 40 `rack_patch` renders): **0 failures**; slowest item 0.163 s. Run 2026-11-27. | `knot generate` per plugin + `knot bench-throughput --domain rack_patch --n 40 --out <dir>`, then `knotwork.posttrain.vcot.extract` on every bundle |
| Reward, verifier server, curriculum, pack | Composite weights, anti-hack rules, endpoints, seed disjointness, ZIP determinism and fixture bundles for all four domains | `tests/posttrain/` |
| Throughput (`rack_patch`, one laptop core, $0.04/vCPU-hour) | `--n 4` at merge: 4 / 4 certified, ~544 items/hour, ~$74 per 1M. `--n 8` re-measured 2026-11-27: 8 / 8 certified, mean gen 798 ms / verify 3,469 ms / vCoT 66 ms, 831 items/hour, $48 per 1M. | `uv run knot bench-throughput --domain rack_patch --n 8 --vcpu-usd-hr 0.04` |

Throughput depends on the machine. The two runs bracket the figure quoted in `README.md`.

## 5. CLI, Studio and Model Arena

| Check | Result | Reproduce |
|---|---|---|
| CLI conformance | 19 registered subcommands; the registered set equals `COMMANDS` ∪ `DIRECT`, and every phase-dispatched command's `--help` exits 0 | `uv run pytest -n 0 tests/conformance/test_cli.py` |
| Studio pages | All five pages render under Streamlit `AppTest` (`test_studio.py`, `test_arena_and_qa_fixes.py`); guide content (5-step tour, 178 tooltips, 10 glossary terms) is validated (`test_guide.py`) | `tests/apps/` |
| Model Arena | Storage, resume keys, spend cap, `pass@k` / `pass^k`, Wilson / z / Fisher statistics, reports, regression gate, `campaign.yaml` round trip, Human QA filter | `tests/arena/` |
| Public sandbox guard | Path confinement to `/tmp/knotwork_demo/` (including `..` and symlinks), caps of 10 items / 3 blocks / *k* ≤ 3 / 10-item batches, no server key read; Offline Mock only (Live mode and key fields absent from the rendered Arena page, `planning.start` forces mock, live resume refused), `campaign.yaml` uploader hidden, `knotwork.plugins` entry points never scanned; no effect when unset (the local Arena page still offers Live and the import) | `tests/apps/test_demo_guard.py` |
| Docker image | **Not re-run for this ledger.** The image definition is unchanged since `3e37fcc`. The smoke procedure is: build, `knot selftest --seeds-only` inside the image, then serve the Studio on `PORT=8080` with `KNOTWORK_PUBLIC_DEMO=1` and expect `ok` from `/_stcore/health`. | `docker build -t knotwork:v2 .` then the commands in `docs/DEPLOY_CLOUD_RUN.md` |

## 6. Continuous integration

| Workflow | Scope | Latest result on `main` |
|---|---|---|
| `ci.yml` (every push and PR; Ubuntu, offline, no keys, `PYTHONHASHSEED=0`) | `tier0`: ruff, ruff format, import-linter, mypy, seed known-answer test, 186 tier0 tests. `tier1`: 270 tests (plugins, pipeline, faults, Studio and Model Arena on shared fixtures). The two jobs run in parallel. | Green on `b92915e`: `tier0` 42 s, `tier1` 85 s, 1 min 29 s wall time |
| `nightly.yml` (scheduled on `main`, or manual) | `tier2`: 104 tests (bulk generation, probes, bench, 10-render `rack_patch` yield gate, background-runner resume), plus the Phase 0 gate, `knot mutate`, `knot selftest` and the Phase 4 gate | Green on `b92915e` (14 min 4 s) |

The full suite is 560 tests. The suite blocks every non-loopback socket and DNS lookup.

## 7. Phase gate records

These local records (`artifacts/gates/phase<N>.json`) are not committed.

| Record | `ok` | Failing checks | Note |
|---|---|---|---|
| `phase0.json` | true | — | |
| `phase1.json` | true | — | |
| `phase2.json` | true | — | |
| `phase3.json` | false | `core.ruff`, `p3.probes` | v1 record. `core.ruff` is clean on current code (CI). `p3.probes` is open on one row (`BLOCKERS.md` 3.6). |
| `phase4.json` | false | `core.pytest`, `p4.scaffold` | The scaffold bug was fixed afterwards (covered by `tests/scaffold`); the Phase 4 gate was not re-run |
| `phase5.json` | false | `p5.bench_mock` | v1 record. The v2 gate function passes (`BLOCKERS.md` 5.6); the full `--phase 5` run was not repeated. |

Gate `p6.docs` (D-47) has two parts:

- `docs_problems()` checks the documentation: required files, relative links, checklist slide order
  and portal fields, README commands and kernel wording, a status line per blocker, and data-card
  sections. It returns no problems on `main` (`tests/apps/test_guide.py`).
- The gate then requires all six records above to be `ok`. It therefore stays red until Phases 3–5
  are re-run and Blocker 3.6 is closed (`BLOCKERS.md` G-1). That requirement is unchanged.

## 8. Live benchmark (human-approved, Checkpoint 2)

Three VLMs through OpenRouter on the 600-item-per-plugin `split=test` pool (`tangle`, `gauge`,
`panel`). The pool used monochrome cables, seed 0.

| Run | Cap | Paid calls | Trials | Cost |
|---|---|---|---|---|
| Two canaries | $5.00 (shared) | not counted | 0 | $0.0501 |
| Main run | $5.00 (shared) | 871 | 997 | $0.9992 |
| True-premise control (SPEC D-37, owner-run) | $0.50 | 72 | 72 | $0.0756 |
| **Total** | | **943** | **1,069** | **$1.12** |

0 HTTP errors and 2 unparsable replies. An offline replay of the cached responses reproduces every
per-model statistic of the main run with 0 network calls.

**True-premise control.**

| Check | Result |
|---|---|
| Matched pairs certified: source re-verifies from files, original premise false, control `CERTIFIED` as `P5`, kernel zone equals the claimed zone, kernel answer "yes", identical pixels | 24 / 24 on every check |
| Request parity between arms (temperature, max tokens, reasoning setting, seed, model slug) | Identical for all three models |
| Control trials / missing / HTTP errors / unparsable | 72 / 0 / 0 / 0 |
| Offline replay of the control | Same replies, grades and statistics; 0 network calls |
| "Yes" on true vs false premise (of 24; exact McNemar p) | GPT-5.4 0 vs 0 (1.00); Gemini 2.5 Flash 15 vs 12 (0.45); Qwen3-VL-235B 23 vs 24 (1.00) |

No model detectably distinguished true from false premises, so the false-premise rejection rates
alone (0 / 50 / 100 %) are not evidence of premise grounding. Abstention is not measured. Raw
outputs are in `samples/live_run/`, and per-model results and caveats are in `README.md` section 5.

## 8b. v2 live 4-domain audit (2026-11-27)

Owner-approved (`.checkpoints/CP2_APPROVED`), run by the owner from
[`docs/examples/full_v2_live_campaign.yaml`](docs/examples/full_v2_live_campaign.yaml): `rack_patch`
standard colour and monochrome, `tangle` monochrome, `gauge` aged dial, `panel` Hard logic; 12 items
per block; GPT-5.4, Gemini 2.5 Flash, Qwen3-VL-235B-A22B; *k* = 3, temperature 0.7, cap $10.00.

| Planned calls | Items certified | Calls issued | Scored | API errors | Spend |
|---|---|---|---|---|---|
| 540 | 59 / 60 (1 quarantined) | 531 | 530 | 1 (not retried) | $0.6727 |

The post-training pack exported from the session holds 55 SFT, 354 DPO and 59 GRPO records
(8 GRPO Frontier, 7 SFT Cold-Start, 0 Saturated cells; train seeds disjoint from eval seeds). Per-block
`pass@1` / `pass@3` / `pass^3`, Wilson intervals and contrasts are in `README.md` section 5 and in
[`docs/evidence/v2_live_audit/`](docs/evidence/v2_live_audit/): `report.md`, `report_metrics.json`, `trials_summary.csv` (no raw replies;
scanned for secret-shaped values), `posttrain_summary.json` and `verification_log.txt`.

### Re-verification on 2026-11-27 (post-freeze fixes, SPEC D-48)

| Change | Evidence |
|---|---|
| Studio xdist race: the first-visit sample was rendered into one shared folder that parallel workers rewrote while others read it ("image file is truncated"); each server process now renders into its own folder | `uv run pytest -n auto tests/apps/test_studio.py` passed 3 / 3 consecutive runs |
| `reward.py` grounding gate: a correct answer with zero grounding on an item with certified boxes scores 0.15 (format only) instead of 0.75, keeps `reward_hack_flag`, and reports raw `outcome_score` beside `effective_outcome_score` 0.0 | `tests/posttrain/test_reward.py` (grounded ≈ 1.0, wrong < 0.25, outcome-only unchanged, kit copy identical) |
| `KNOTWORK_PUBLIC_DEMO=1` hardening: Offline Mock only, campaign import hidden, external plugins not loaded | `tests/apps/test_demo_guard.py` (11 tests) |

The full command log is [`docs/evidence/v2_live_audit/verification_log.txt`](docs/evidence/v2_live_audit/verification_log.txt).

## 9. Open items

See [`BLOCKERS.md`](BLOCKERS.md):

- 3.6: `panel` P1 probe residual (open).
- 5.6: mock staircase tolerance (resolved for the gate function).
- G-1: stale Phase 3–5 gate records (open).
- L-1: live staircases at the floor (documented).
- L-2: one failed API call in the v2 live audit, not retried (documented).

## 10. Reproduction commands

```bash
uv sync --locked
uv run ruff check . && uv run ruff format --check .
uv run lint-imports && uv run mypy
uv run knot selftest --seeds-only
uv run pytest -n auto -m "tier0 or tier1" -q        # what CI runs
uv run pytest -n auto -m tier2 -q                   # what nightly runs
uv run knot mutate --fault all --n 4
uv run knot selftest
uv run knot bench-throughput --domain rack_patch --n 8
uv run python scripts/verify_phase.py --phase N --keep-going
```
