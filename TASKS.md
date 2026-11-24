# TASKS — Knotwork autonomous execution checklist

Sequential, date-free. Work strictly top to bottom. A task is `[x]` only when its **Verification Command**
exits 0 and its **Done Criteria** hold. A phase is done only when
`uv run python scripts/verify_phase.py --phase N` exits 0. Section numbers (§) refer to `SPEC.md`.

Legend: `[ ]` todo · `[x]` done · `[!]` blocked (see `BLOCKERS.md`) · `[~]` skipped by human decision.

---

## Phase 0 — Bootstrap, renderer smoke, contracts, isolation, determinism

- [ ] **0.1 Environment bootstrap**
  - Files: `uv.lock`, `.python-version` (verify only)
  - Verification Command: `uv sync --locked && uv run knot --help && uv run knot selftest --seeds-only`
  - Done Criteria: all exit 0; `--help` lists the 10 subcommands; seeds KAT prints `"ok": true`.

- [ ] **0.2 Git baseline**
  - Files: repository root
  - Verification Command: `git rev-parse --is-inside-work-tree && git log --oneline -1`
  - Done Criteria: repo initialised on branch `main`; first commit `chore: import knotwork starter pack` contains every starter file; `git status` clean.

- [ ] **0.3 Deterministic layered renderer** (§6)
  - Files: `src/knotwork/render/canvas.py`, `tests/conformance/test_render_smoke.py`
  - Verification Command: `uv run pytest tests/conformance/test_render_smoke.py -q`
  - Done Criteria: tests prove (a) beauty is RGB uint8 1024², drawn at 2048 and Lanczos-downsampled; (b) `id_buffer.png` reloads as uint16 and its values ⊆ {0} ∪ object ids; (c) every mask has values ⊆ {0,255} (no AA); (d) text masks are non-AA; (e) rendering the same scene twice gives identical `array_digest`s; (f) after `halo_patch`, `id` at the crossing centre equals the top object and halo pixels of the under-wire are 0 while its amodal mask stays continuous; (g) a font whose SHA-256 differs from the profile raises an error mentioning `E_FONT`; (h) `finish()` returns a valid `RenderManifest`.

- [ ] **0.4 Contracts and schema sync**
  - Files: `src/knotwork/knotcore/contracts.py`, `docs/schemas/*.schema.json` (verify only)
  - Verification Command: `uv run python scripts/export_schemas.py --check && uv run pytest tests/conformance/test_contracts.py -q`
  - Done Criteria: exit 0; no contract edits needed (if one is, follow CLAUDE.md §2 law 7).

- [ ] **0.5 Kernel isolation double layer**
  - Files: `.importlinter`, `src/knotwork/devtools/kernel_audit.py` (verify only)
  - Verification Command: `uv run lint-imports && uv run pytest tests/conformance/test_kernel_isolation.py -q`
  - Done Criteria: `Contracts: 3 kept, 0 broken`; AST scanner tests pass.

- [ ] **0.6 Seed determinism**
  - Files: `src/knotwork/knotcore/seeds.py` (verify only)
  - Verification Command: `uv run pytest tests/conformance/test_seeds.py -q`
  - Done Criteria: all pass including the fresh-process KAT test.

- [ ] **0.7 Phase 0 gate**
  - Files: `.ci_phase`
  - Verification Command: `uv run python scripts/verify_phase.py --phase 0`
  - Done Criteria: `PHASE 0 GATE: PASSED`; write `0` to `.ci_phase`; commit `chore(phase0): gate passed`.

---

## Phase 1 — knotcore trusted pixel kernel (§7) — executable LOC ≤ 850, total ≤ 1350

- [ ] **1.1 Bundle loader and kernel fixtures**
  - Files: `src/knotwork/knotcore/io.py`, `tests/kernel/conftest.py`, `tests/kernel/fixtures.py`, `tests/kernel/test_io.py`
  - Verification Command: `uv run pytest tests/kernel/test_io.py -q`
  - Done Criteria: `load_bundle(dir) -> Bundle` (manifest, claims, beauty, luminance, id uint16, masks bool) with `E_SCHEMA` on invalid JSON/shape; fixtures build tiny synthetic bundles with `render.canvas` directly (two strands with one crossing + terminals + labels; one gauge) — **no plugin imports in tests/kernel**.

- [ ] **1.2 OCR with template fallback** (§7.9)
  - Files: `src/knotwork/knotcore/ocr.py`, `tests/kernel/test_ocr.py`
  - Verification Command: `uv run pytest tests/kernel/test_ocr.py -q`
  - Done Criteria: OCR runs only on the 768-px consumption image; every start letter in the profile alphabet, `1`–`10`, `G1`–`G4` and all tick labels (`00`…`90`) read correctly at 768 px with RapidOCR (≥ 99 % over the fixture set, 100 % after fallback); the crop-hash LRU cache (§7.9) returns the cached result for an identical crop without calling RapidOCR again (test with a counting stub) and is keyed by `array_digest` of the preprocessed crop; forced-fallback test exercises template matching; a swapped label yields `E_LABEL_MISMATCH`; blank crop yields `E_OCR_FAIL`; engine is a lazy singleton.

- [ ] **1.3 TRACE** (§7.1–7.2)
  - Files: `src/knotwork/knotcore/topo.py`, `tests/kernel/test_trace.py`
  - Verification Command: `uv run pytest tests/kernel/test_trace.py -q`
  - Done Criteria: correct end label on fixtures; broken mask ⇒ `E_TRACE_NO_TERMINAL`; mask touching two end boxes ⇒ `E_TRACE_AMBIGUOUS`; evidence has `path_px`, `end_terminal`, `ocr_text`.

- [ ] **1.4 CROSS with geodesic ordering** (§7.3)
  - Files: `src/knotwork/knotcore/topo.py`, `tests/kernel/test_cross.py`
  - Verification Command: `uv run pytest tests/kernel/test_cross.py -q`
  - Done Criteria: counts 0/1/3 crossings correctly; `"*"` sums over strands; centres ordered by `MCP_Geometric` distance from the start box (test with a strand that doubles back so x-order ≠ geodesic order).

- [ ] **1.5 ONTOP** (§7.4)
  - Files: `src/knotwork/knotcore/topo.py`, `tests/kernel/test_ontop.py`
  - Verification Command: `uv run pytest tests/kernel/test_ontop.py -q`
  - Done Criteria: reads the id buffer at the indexed crossing centre; occluder over a crossing ⇒ `E_OCCLUDED_CROSSING`; out-of-range idx ⇒ `E_CLAIM_MISMATCH`.

- [ ] **1.6 READ** (§7.5)
  - Files: `src/knotwork/knotcore/gauge.py`, `tests/kernel/test_read.py`
  - Verification Command: `uv run pytest tests/kernel/test_read.py -q`
  - Done Criteria: hypothesis test over linear/offset scales and integer values: `|v̂ − v| ≤ tol` and `tol < 0.01 × span`; displaced pivot ⇒ `E_NEEDLE_OFFCENTER`; ticks rendered out of value order ⇒ `E_SCALE_MISMATCH`; never reads `claimed_center`.

- [ ] **1.7 ZONE, GAUGE_OF, READ_ALL** (§7.6–7.7)
  - Files: `src/knotwork/knotcore/gauge.py`, `tests/kernel/test_zone.py`
  - Verification Command: `uv run pytest tests/kernel/test_zone.py -q`
  - Done Criteria: zone from arc-mask angular extents; needle within 1° of a boundary ⇒ `E_ZONE_UNRESOLVED`; GAUGE_OF resolves the unique dilated dial.

- [ ] **1.8 COUNT, CMP, ARGMAX** (§7.8)
  - Files: `src/knotwork/knotcore/logic.py`, `tests/kernel/test_logic.py`
  - Verification Command: `uv run pytest tests/kernel/test_logic.py -q`
  - Done Criteria: tolerance-aware CMP; ARGMAX tie within tolerances ⇒ `E_AMBIGUOUS_ARGMAX`; COUNT honours `min_visible_px` and `zone:` predicates.

- [ ] **1.9 Beauty-to-mask witness** (§7.10)
  - Files: `src/knotwork/knotcore/witness.py`, `tests/kernel/test_witness.py`
  - Verification Command: `uv run pytest tests/kernel/test_witness.py -q`
  - Done Criteria: low-contrast wire ⇒ `E_BEAUTY_MISMATCH`; beauty top wire ≠ id top at a crossing ⇒ `E_BEAUTY_MISMATCH`; ink without id ⇒ `E_UNEXPLAINED_INK`; AA fringe of legit strokes is NOT flagged; mask on border / OOB box ⇒ `E_OOB`; returns `WitnessCheck` list.

- [ ] **1.10 Consumption-resolution legibility guard** (§7.11)
  - Files: `src/knotwork/knotcore/legibility.py`, `tests/kernel/test_legibility.py`
  - Verification Command: `uv run pytest tests/kernel/test_legibility.py -q`
  - Done Criteria: 2-px strokes ⇒ `E_ILLEGIBLE_RES`; 15° crossing ⇒ `E_TANGENT_CROSSING`; crossings closer than 4 strokes ⇒ `E_CROSSING_SPACING`; foreign ink near a terminal ⇒ `E_TERMINAL_CLEARANCE`; labels OCR'd at 768.

- [ ] **1.11 Engine: execute, claims, certify** (§7.12)
  - Files: `src/knotwork/knotcore/engine.py`, `tests/kernel/test_engine.py`
  - Verification Command: `uv run pytest tests/kernel/test_engine.py -q`
  - Done Criteria: clean fixture ⇒ `CERTIFIED`; a 4-step premise program returns `Answer(type=PREMISE, value=PremiseValue(premise_holds=(ZONE == claimed_zone), value=READ), canonical="yes|no; <g>")` for both a true and a false premise (§7.12); wrong D1 ⇒ `E_D1_D3_MISMATCH`; wrong ONTOP claim ⇒ `E_ZORDER`; wrong READ claim ⇒ `E_SCALE_MISMATCH`; D2 ≠ D1 ⇒ `E_D1_D2_MISMATCH`; all stages run and codes accumulate; `$sN` substitution works for 3-step programs; `engine.kernel_sha256()` equals `devtools.kernel_audit.kernel_sha256()`.

- [ ] **1.12 Grading** (§13.3)
  - Files: `src/knotwork/knotcore/grading.py`, `tests/kernel/test_grading.py`
  - Verification Command: `uv run pytest tests/kernel/test_grading.py -q`
  - Done Criteria: table-driven tests for all 5 types (free text with/without `ANSWER:` line, native values); hypothesis fuzz never raises; AST test proves imports ⊆ {json, math, re, unicodedata, typing, __future__}.

- [ ] **1.13 Phase 1 gate**
  - Files: `.ci_phase`
  - Verification Command: `uv run python scripts/verify_phase.py --phase 1`
  - Done Criteria: PASSED with knotcore executable LOC ≤ 850 and total ≤ 1350 (record `loc_report().summary()` and the per-module table in the commit body); commit `feat(knotcore): pixel kernel complete`.

---

## Phase 2 — Untrusted plugins, twins, gaslighting, fault matrix, contact sheet

- [ ] **2.1 Tangle braid algebra** (§8.1)
  - Files: `src/knotwork/plugins/tangle/braid.py`, `tests/plugins/test_braid.py`
  - Verification Command: `uv run pytest tests/plugins/test_braid.py -q`
  - Done Criteria: composition matches hand-worked examples (σ₁σ₂ on n=3, polarity rule `over = a iff ε=+1`); sampled permutations uniform (χ² p > 0.01 over 6000 draws, n=4); pure pairs and R-II pairs preserve permutation; ID slots counted; per-strand crossing counts correct.

- [ ] **2.2 Tangle layouts + shapely validation**
  - Files: `src/knotwork/plugins/tangle/layout.py`, `src/knotwork/plugins/tangle/geom.py`, `tests/plugins/test_tangle_layout.py`
  - Verification Command: `uv run pytest tests/plugins/test_tangle_layout.py -q`
  - Done Criteria: strip + horseshoe; jitter fold-free (`is_simple`); geometric intersection counts equal braid crossing counts for 200 sampled scenes; nominal crossing angle ≥ 35°; `geom.geometric_answer` (D2) agrees with braid D1 on T1/T2/T3.

- [ ] **2.3 Tangle render + plugin** (§6.1, §8.1)
  - Files: `src/knotwork/plugins/tangle/render.py`, `src/knotwork/plugins/tangle/plugin.py`, `src/knotwork/plugins/tangle/__init__.py`, `tests/plugins/test_tangle_plugin.py`
  - Verification Command: `uv run pytest tests/plugins/test_tangle_plugin.py -q`
  - Done Criteria: `isinstance(PLUGIN, TaskPlugin)`; schematic/harness styles; unique/partial/mono colours; exact prompts; claims for every crossing/start/strand; ≥ 95 % of 40 sampled clean scenes certify through `knotcore.engine.certify`; fault hooks M1–M7, M10 implemented and `fault_effective` defined; causal/nuisance twins per §8.1.

- [ ] **2.4 Gauge plugin** (§8.2)
  - Files: `src/knotwork/plugins/gauge/{scale,render,geom,plugin,__init__}.py`, `tests/plugins/test_gauge_plugin.py`
  - Verification Command: `uv run pytest tests/plugins/test_gauge_plugin.py -q`
  - Done Criteria: linear + offset scales, horizontal 2-digit labels OCR at 768 (100 % on 40 dials); zones; needle avoids boundaries; ≥ 95 % certify; M8/M9 hooks; twins.

- [ ] **2.5 Panel plugin with gaslighting** (§8.3)
  - Files: `src/knotwork/plugins/panel/{compose,programs,plugin,__init__}.py`, `tests/plugins/test_panel_plugin.py`
  - Verification Command: `uv run pytest tests/plugins/test_panel_plugin.py -q && git diff --stat HEAD -- src/knotwork/knotcore`
  - Done Criteria: P1–P5 + T_GASLIGHT programs execute in the kernel unchanged (the `git diff` on knotcore is empty); premise families emit exactly the 4 steps `TRACE, GAUGE_OF, READ, ZONE` and D1 builds the same `PremiseValue` the kernel builds; schedule gives exact `round_half_up(0.15 n)` gaslight and control counts; ARGMAX margin ≥ 3 × tol; ≥ 90 % certify.

- [ ] **2.6 Plugin registry**
  - Files: `src/knotwork/plugins/registry.py`, `tests/plugins/test_registry.py`
  - Verification Command: `uv run pytest tests/plugins/test_registry.py -q && uv run lint-imports`
  - Done Criteria: `get_plugin(name)` for tangle/gauge/panel; unknown name raises `KeyError`; import contracts still kept.

- [ ] **2.7 Pipeline bundle I/O + certification orchestrator** (§3, §7.13)
  - Files: `src/knotwork/pipeline/io.py`, `src/knotwork/pipeline/certify.py`, `tests/pipeline/test_io_certify.py`
  - Verification Command: `uv run pytest tests/pipeline/test_io_certify.py -q`
  - Done Criteria: writes bundles exactly per §3 with decoded-array digests and `bundle_id`; converts failed certificates into `QuarantineRecord` with correct `stage` and `primary_code`.

- [ ] **2.8 Twin builder**
  - Files: `src/knotwork/pipeline/twins.py`, `tests/pipeline/test_twins.py`
  - Verification Command: `uv run pytest tests/pipeline/test_twins.py -q`
  - Done Criteria: causal twin answer differs with `pixel_diff_frac < 0.05`; nuisance answer equal with `≥ 0.02`; retries ≤ 3; twin family consistent with premise truth.

- [ ] **2.9 `knot generate` + offline freshness receipt** (§8.4, §11.1)
  - Files: `src/knotwork/pipeline/generate.py`, `src/knotwork/freshness/receipt.py`, `src/knotwork/freshness/beacon.py` (offline path), `tests/pipeline/test_generate.py`
  - Verification Command: `uv run knot generate --plugin tangle --n 6 --seed 1 --out artifacts/tmp/gen && uv run pytest tests/pipeline/test_generate.py -q`
  - Done Criteria: exact base count; index/quarantine/stats/receipt written; `families_expected` in stats; canary in every task.json.

- [ ] **2.10 Determinism under parallelism**
  - Files: `tests/pipeline/test_determinism.py`
  - Verification Command: `uv run pytest tests/pipeline/test_determinism.py -q`
  - Done Criteria: same seed ⇒ identical ids for `--workers 1` and `--workers 2`; different seed ⇒ disjoint ids.

- [ ] **2.11 `knot mutate` — M1–M10 conformance matrix** (§10)
  - Files: `src/knotwork/pipeline/mutate.py`, `tests/pipeline/test_mutate.py`
  - Verification Command: `uv run knot mutate --fault all --n 4 --out artifacts/tmp/mutate.json && uv run pytest tests/pipeline/test_mutate.py -q`
  - Done Criteria: MutateReport per §9.1; `exact_catch_rate == 1.0`; clean false-quarantine ≤ 0.10. If a fault is missed, fix the **kernel check or plugin injection** — never weaken thresholds.

- [ ] **2.12 Contact sheet with Pixel-Witness Proof strips** (§9.1 "Contact sheet")
  - Files: `src/knotwork/pipeline/contact_sheet.py`, `tests/pipeline/test_contact_sheet.py`
  - Verification Command: `uv run python -m knotwork.pipeline.contact_sheet --data artifacts/phase2 --out artifacts/tmp/sheet.png && uv run pytest tests/pipeline/test_contact_sheet.py -q`
  - Done Criteria: PNG ≥ 2048×1536 + sidecar `sheet.json` per §9.1; rows `tangle`/`gauge`/`panel` start base, causal, nuisance; Pixel-Witness Proof strips `pixel_witness:tangle` and `pixel_witness:panel` with exactly `beauty | id_buffer | kernel_overlay` (overlay = kernel TRACE component, crossing centres + halo circles + ONTOP winner, needle PCA axis + dial centre, OCR text — all computed by calling knotcore on the bundle); fault row M1–M10 captioned with their E_* codes.

- [ ] **2.13 Phase 2 gate → [HUMAN-CHECKPOINT-1]**
  - Files: `.ci_phase`, `artifacts/phase2/CHECKPOINT_1_READY.md`
  - Verification Command: `uv run python scripts/verify_phase.py --phase 2`
  - Done Criteria: PASSED; `.ci_phase` = 2; commit `feat(plugins): tangle, gauge, panel + M1-M10 matrix`. **STOP** and ask the human to review `artifacts/phase2/contact_sheet.png` (optionally run the ≤ $0.50 gap check printed in `CHECKPOINT_1_READY.md` — human-only) and create `.checkpoints/CP1_APPROVED`.

---

## Phase 3 — Freshness, shortcut audit, diversity, golden selftest, file-only verify

- [ ] **3.1 Network gateway + drand/NIST beacons**
  - Files: `src/knotwork/net.py`, `src/knotwork/freshness/beacon.py`, `tests/freshness/test_beacon.py`
  - Verification Command: `uv run pytest tests/freshness/test_beacon.py -q`
  - Done Criteria: `net.get_json` refuses unless `KNOTWORK_NETWORK` permits; drand/NIST parsed from `httpx.MockTransport` fixtures; `knot generate --beacon drand` path covered with mocks; no real network.

- [ ] **3.2 Emission mask, burn ledger, receipt**
  - Files: `src/knotwork/freshness/{emission,ledger,receipt}.py`, `tests/freshness/test_emission_ledger.py`
  - Verification Command: `uv run pytest tests/freshness -q`
  - Done Criteria: HMAC mask fraction ≈ `sealed_fraction` (±3 % on 5000 ids); commitments verify; ledger append-only and consulted by `generate --split test`; `receipt_id` excludes `fetched_at_utc`.

- [ ] **3.3 Shortcut probes** (§11.2)
  - Files: `src/knotwork/probes/{heuristics,gbdt,text_only,stats}.py`, `tests/probes/test_probes.py`
  - Verification Command: `uv run pytest tests/probes/test_probes.py -q`
  - Done Criteria: each probe beats chance on a deliberately leaky synthetic dataset (proves it has power) and Wilson/χ²/pHash helpers match reference values.

- [ ] **3.4 `knot probe`**
  - Files: `src/knotwork/probes/run.py`, `tests/probes/test_probe_cli.py`
  - Verification Command: `uv run pytest tests/probes/test_probe_cli.py -q`
  - Done Criteria: ProbeReport schema per §11.2 on a small generated dataset; GroupKFold by `scene_id`.

- [ ] **3.5 `knot diversity` (Vendi)** (§11.4)
  - Files: `src/knotwork/diversity.py`, `tests/probes/test_diversity.py`
  - Verification Command: `uv run pytest tests/probes/test_diversity.py -q`
  - Done Criteria: VS(identical items) = 1 ± 1e-6; VS(orthonormal items) = n ± 1e-6; report per §11.4.

- [ ] **3.6 Benchmark split and shortcut tuning**
  - Files: generator code as needed (never kernel thresholds)
  - Verification Command: `uv run python scripts/verify_phase.py --phase 3 --keep-going`
  - Done Criteria: gates `p3.probes` and `p3.diversity` pass on 600 base items/plugin (`split=test`, mono, `--no-twins`); generation of the split finishes in one sitting (OCR at 768 px only, crop-hash cache warm across items in each worker). If a probe leaks, fix the sampler (balance, placement, label assignment) and record a decision in SPEC §16.

- [ ] **3.7 Golden selftest**
  - Files: `src/knotwork/pipeline/selftest.py`, `tests/golden/golden.json`, `tests/pipeline/test_selftest.py`
  - Verification Command: `uv run knot selftest --update-golden && uv run knot selftest && uv run pytest tests/pipeline/test_selftest.py -q`
  - Done Criteria: SelftestReport `ok: true`; golden committed; second run byte-identical report.

- [ ] **3.8 File-only `knot verify`**
  - Files: `src/knotwork/pipeline/verify.py`, `tests/pipeline/test_verify.py`
  - Verification Command: `uv run knot verify artifacts/selftest/data && uv run pytest tests/pipeline/test_verify.py -q`
  - Done Criteria: VerifyReport per §9.1; tampering detected; AST test proves `verify.py` does not import `knotwork.plugins`.

- [ ] **3.9 Phase 3 gate**
  - Files: `.ci_phase`
  - Verification Command: `uv run python scripts/verify_phase.py --phase 3`
  - Done Criteria: PASSED; `.ci_phase` = 3; commit `feat(freshness,probes): freshness protocol + shortcut audit`.

---

## Phase 4 — Dual exporters, plugin scaffold, CI

- [ ] **4.1 Harbor exporter** (§13.1)
  - Files: `src/knotwork/export/harbor.py`
  - Verification Command: `uv run python -c "import knotwork.export.harbor"`
  - Done Criteria: writes the exact folder layout; `tests/verify.py` = verbatim `knotcore/grading.py` source + main; `task.toml` parses with `tomllib`.

- [ ] **4.2 HF parquet + RLVR exporters** (§13.2)
  - Files: `src/knotwork/export/hf.py`, `src/knotwork/export/rlvr.py`
  - Verification Command: `uv run python -c "import knotwork.export.hf, knotwork.export.rlvr"`
  - Done Criteria: columns exactly per §13.2; dataset card with configs; sealed answers → commitments.

- [ ] **4.3 `knot export` + oracle pytest**
  - Files: `src/knotwork/export/run.py`, `tests/export/test_harbor_oracle.py`, `tests/export/test_hf_rlvr.py`
  - Verification Command: `uv run pytest tests/export -q`
  - Done Criteria: oracle = 1.0, wrong = 0.0, missing = 0.0 for every exported family; parquet round-trips; RLVR grade_prediction oracle/wrong.

- [ ] **4.4 `knot new-plugin` scaffold**
  - Files: `src/knotwork/scaffold/new_plugin.py`, `src/knotwork/scaffold/templates/*`, `tests/scaffold/test_new_plugin.py`
  - Verification Command: `uv run pytest tests/scaffold -q`
  - Done Criteria: generated package satisfies `TaskPlugin`, has a passing test, and knotcore sha is unchanged.

- [ ] **4.5 CI pipeline check**
  - Files: `.github/workflows/ci.yml`, `.ci_phase`
  - Verification Command: `uv run python scripts/verify_phase.py --phase 4`
  - Done Criteria: PASSED (includes `p4.ci`); `.ci_phase` = 4; commit `feat(export): harbor + hf/rlvr exporters, scaffold`.

---

## Phase 5 — Psychometric staircase, VLM clients, statistics → [HUMAN-CHECKPOINT-2]

- [ ] **5.1 1-up-2-down staircase** (§12.2)
  - Files: `src/knotwork/bench/staircase.py`, `tests/bench/test_staircase.py`
  - Verification Command: `uv run pytest tests/bench/test_staircase.py -q`
  - Done Criteria: with a simulated observer of known psychometric curve the threshold lands within 1.5 levels of the 70.7 % level in ≥ 90 % of 200 seeded runs.

- [ ] **5.2 Prompts, client interface, mock VLMs** (§12.1, §12.4)
  - Files: `src/knotwork/bench/{prompts,client_base,client_mock}.py`, `tests/bench/test_mock_client.py`
  - Verification Command: `uv run pytest tests/bench/test_mock_client.py -q`
  - Done Criteria: deterministic per seed; error model exactly as §12.4.

- [ ] **5.3 Replay + OpenRouter clients (mocked transport only)**
  - Files: `src/knotwork/bench/{client_replay,client_openrouter}.py`, `tests/bench/test_openrouter_mock.py`
  - Verification Command: `uv run pytest tests/bench/test_openrouter_mock.py -q`
  - Done Criteria: with `httpx.MockTransport`: body has `provider.data_collection == "deny"` and `usage.include`; auth header; 429 retry; response cached before return; replay miss raises `ReplayMiss`; zero real sockets.

- [ ] **5.4 Budget + statistics** (§12.3, §12.5)
  - Files: `src/knotwork/bench/{budget,stats}.py`, `tests/bench/test_stats_budget.py`
  - Verification Command: `uv run pytest tests/bench/test_stats_budget.py -q`
  - Done Criteria: MLE recovers p on simulated data (|p̂ − p| < 0.02 at n = 2000); bootstrap seeded and reproducible; Abbott formula; budget stops before exceeding cap.

- [ ] **5.5 `knot bench` runner + fixtures**
  - Files: `src/knotwork/bench/run.py`, `tests/bench/fixtures/data/`, `tests/bench/fixtures/replay/`, `tests/bench/test_bench_cli.py`
  - Verification Command: `uv run pytest tests/bench -q`
  - Done Criteria: mock, replay and `--live --dry-run` modes produce reports per §12.6/§12.5; fixtures committed (small dataset from `knot generate`, replay files generated by the mock client written in replay format).

- [ ] **5.6 Phase 5 gate → [HUMAN-CHECKPOINT-2]**
  - Files: `.ci_phase`, `artifacts/phase5/CHECKPOINT_2_READY.md`
  - Verification Command: `uv run python scripts/verify_phase.py --phase 5`
  - Done Criteria: PASSED; `.ci_phase` = 5; commit `feat(bench): staircase + diagnostic harness`. **STOP** and show the human the preflight cost; wait for `.checkpoints/CP2_APPROVED` or `.checkpoints/CP2_SKIPPED`.

- [ ] **5.7 Live 3-VLM run (only if `.checkpoints/CP2_APPROVED` exists)**
  - Files: `bench_replay/`, `artifacts/phase5/bench_live/`, `freshness/burn_ledger.jsonl`
  - Verification Command: `uv run knot bench --data artifacts/phase3/bench_split --out artifacts/phase5/bench_live --live --budget-usd 8 && uv run knot bench --data artifacts/phase3/bench_split --out artifacts/phase5/bench_replayed --client replay`
  - Done Criteria: live report `cost_usd ≤ 8.00`; replay of the same run reproduces it with `network_calls == 0`; commit `data(bench): live run replay cache`. If `CP2_SKIPPED`, mark `[~]`.

---

## Phase 6 — Inspector, figures, report, documentation, portal copy

- [ ] **6.1 Publication figures**
  - Files: `src/knotwork/report/figures.py`, `docs/figures/fig01_*.png … fig11_*.png`
  - Verification Command: `uv run python -m knotwork.report.figures --artifacts artifacts --out docs/figures`
  - Done Criteria: ≥ 11 PNGs ≥ 1600 px wide (dpi 200) mapped to deck slides in `docs/SUBMISSION_CHECKLIST.md`; uses mock/replay results when CP2 skipped (labelled as such).

- [ ] **6.2 Streamlit Forensic Inspector** (§15)
  - Files: `apps/inspector.py`, `tests/apps/test_inspector.py`
  - Verification Command: `uv run pytest tests/apps -q`
  - Done Criteria: `AppTest` runs with no exception and finds the 4 tabs; human staircase writes jsonl.

- [ ] **6.3 `knot report`**
  - Files: `src/knotwork/report/build.py`, `tests/apps/test_report.py`
  - Verification Command: `uv run knot report && uv run pytest tests/apps/test_report.py -q`
  - Done Criteria: single self-contained HTML > 50 KB with embedded figures, gate table, fault matrix, probe table, bench summary.

- [ ] **6.4 Samples, README, VERIFICATION, DATA_CARD**
  - Files: `samples/README.md`, `samples/harbor/` (10 tasks), `samples/hf/` (parquet ≤ 50 rows), `samples/logs/` (gate JSONs, mutate/probe/diversity/selftest/verify reports), `README.md`, `VERIFICATION.md`, `DATA_CARD.md`
  - Verification Command: `uv run pytest tests/docs -q`
  - Done Criteria: README has quickstart (`uv sync`, `knot selftest`, `knot generate`, `knot export --format harbor`, `--offline`), architecture diagram (ASCII), results table; VERIFICATION.md cites `artifacts/gates/phase0.json`…`phase5.json` and `samples/logs/`; DATA_CARD has Intended use, Limitations, License, canary.

- [ ] **6.5 Deck copy + portal submission copy** (run `/ship-pack`)
  - Files: `docs/DECK_COPY.md`, `docs/PORTAL_SUBMISSION.md`, `tests/docs/test_portal_limits.py`
  - Verification Command: `uv run pytest tests/docs/test_portal_limits.py -q`
  - Done Criteria: 11 `## Slide N — <official title>` sections in the official template order of `docs/SUBMISSION_CHECKLIST.md` §C (titles verbatim; Slide 3 quotes *"measures true reasoning performance while preventing test data leakage"*, Slide 8 quotes *"pytest or Harbor framework for evaluation"*); `FIELD:` blocks for challenge, prototype_brief (≤ 1024), technology_stack (≤ 1024), github_repo, demo_video.

- [ ] **6.6 Final audit and Phase 6 gate**
  - Files: `.ci_phase`
  - Verification Command: `uv run python scripts/verify_phase.py --phase 6`
  - Done Criteria: PASSED; `.ci_phase` = 6; commit `docs: submission pack`; print the final summary in the chat (gate table + where every deliverable lives).
