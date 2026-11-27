# Knotwork — Proof-Carrying Visual Task Factory

**Synthetic visual reasoning tasks for VLMs where every label ships with a machine-checked proof, so a wrong answer key cannot reach your benchmark.**

[![ci](https://github.com/swaritbkp/vsfactory/actions/workflows/ci.yml/badge.svg?branch=feat/vtf-v2)](https://github.com/swaritbkp/vsfactory/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-3776AB)
![uv](https://img.shields.io/badge/env-uv%20--locked-6E40C9)
![offline](https://img.shields.io/badge/offline--first-0%20API%20keys-059669)
![faults](https://img.shields.io/badge/fault%20matrix-56%2F56%20caught-047857)

Innodata AI Hackathon 2026 · Track 2 · **Visual Task Factory** · Team Bilota

| Rubric criterion | Where to look (each one is a short read) |
|---|---|
| Working & repeatable (25 %) | [3-command quickstart](#quickstart-offline-3-commands-or-docker) · [Docker](#docker) · CI on every push |
| Validity evidence (25 %) | [Zero-trust kernel](#how-knotwork-proves-every-task-zero-trust-architecture) · [proof metrics](#proof-metrics) · [`VERIFICATION.md`](VERIFICATION.md) |
| Meaningful variation (20 %) | Vendi 2.1×–3.7× over recolouring · causal / nuisance twins · structural OOD holdouts |
| Originality (15 %) | [Rack patch-bay + topological tangles + proof-carrying factory](#what-it-makes-and-what-it-found) |
| Practical impact & communication (15 %) | [Why this matters](#why-this-matters) · [5-page Studio and Model Arena](#5-page-factory-studio-and-model-arena) |

---

## Why this matters

Frontier VLMs are now being evaluated for **data-center operations, field-service copilots and brownfield
MRO**: server rack patch bays, wiring harnesses, plant control panels and legacy analog gauges. A
greenfield plant can replace analog dials with digital telemetry. **You cannot bolt an OBD port onto a
tangled 24-port Cat6 loom or a misrouted wiring harness**: someone, or some model, has to trace it by eye.
When that tracing fails you get miswired racks, missed safety interlocks and costly downtime.

The benchmarks used to judge these models have two quiet failure modes. Static test sets leak into
training data, and labels written by an LLM-as-a-judge are sometimes wrong without anyone noticing.
Knotwork addresses both. It generates fresh items on demand, and it emits an item only when an isolated
kernel has independently re-derived and certified the answer.

## What it makes, and what it found

![Studio: a certified Rack patch-bay task in Witness Overlay](docs/assets/studio_rack_patch.png)

* **Server Rack Patch-Bay** (`rack_patch`, new in v2): a 2U `SWITCH-A` / `PATCH-B` chassis with an
  8–14 cable loom. Typical question: "trace the cable from switch port SW-05; which patch port does it
  end in?"
* **Topological cable tangles** (`tangle`): cables woven from braid words, so the answer depends on
  real over/under crossings and cannot be read off position or colour.
* **Proof-carrying factory**: every bundle `(image, question, answer, evidence)` carries a kernel
  certificate. `gauge` (analog dials) and `panel` (cables from sockets into gauges) run through the same
  kernel unchanged, which shows that new domains plug in without touching the trusted code.

**Headline finding (live, human-approved run).** We tested GPT-5.4, Gemini 2.5 Flash and Qwen3-VL-235B
on `tangle`, `gauge` and `panel` items: **1,069 trials for $1.12 in total**. Overall accuracy was only
about **20–25 %**. When every cable was rendered in one colour, so colour could no longer stand in for
tracing, **GPT-5.4 fell from 59.4 % to 3.1 %** (32 matched pairs, exact McNemar *p* = 4 × 10⁻⁵). The
geometry was identical in both conditions; only the colour shortcut was removed. (`rack_patch` was added
after this run, and its offline mock sessions are shown below.)

![Model Arena: a completed offline mock Rack patch-bay + Panel comparison with its leaderboard and Report tab](docs/assets/studio_model_arena.png)

<details>
<summary>Full live results table and caveats</summary>

One live run on the 600-item-per-plugin `split=test` pool (monochrome cables), seed 0, adaptive
staircases plus the diagnostic blocks of `SPEC.md` §12, then a 72-trial true-premise control (SPEC D-37).
943 paid calls / 1,069 trials, 0 HTTP errors: main run $0.9992 (871 calls, 997 trials), control $0.0756
(72 calls), canaries $0.0501.

| | GPT-5.4 | Gemini 2.5 Flash | Qwen3-VL-235B |
|---|---|---|---|
| Overall accuracy | 25.2 % | 20.5 % | 20.2 % |
| Tangle · panel · gauge reading | 27.3 · 22.8 · 15.6 % | 17.0 · 24.4 · 25.0 % | 21.0 · 19.4 · 6.2 % |
| Colour ablation: unique colours → monochrome (32 pairs; exact McNemar *p*) | 59.4 → 3.1 % (4 × 10⁻⁵) | 18.8 → 9.4 % (0.38) | 37.5 → 15.6 % (0.065) |
| Twins: original / causal / nuisance | 29 / 12.5 / 25 % | 25 / 12.5 / 29 % | 25 / 12.5 / 25 % |
| "Yes" on true vs false premise (of 24; McNemar *p*) | 0 vs 0 (1.00) | 15 vs 12 (0.45) | 23 vs 24 (1.00) |
| Main run cost | $0.808 | $0.121 | $0.070 |

Caveats: accuracy sits near the floor, so the staircases bound thresholds rather than measure them
(`BLOCKERS.md` L-1). Per-model twin drops are directional only (n = 24, *p* = 0.125 / 0.25 / 0.375). No
model detectably told true premises from false ones, so false-premise rates alone are not evidence that
a model checked the premise. Abstention is not measured. Raw outputs are in
[`samples/live_run/`](samples/live_run/), and `uv run knot report` rebuilds every figure offline.
</details>

## How Knotwork proves every task (zero-trust architecture)

```mermaid
flowchart LR
    subgraph U["Untrusted plugins (draw scenes, may be wrong)"]
        R[rack_patch]
        T[tangle]
        G[gauge]
        P[panel]
    end
    subgraph K["Trusted computing base: knotcore (import-isolated)"]
        direction TB
        M["re-derive answer from renderer object masks (D3)"]
        X["cross-check visible image: beauty-only sanity oracle, CVD gate, stray-ink witness, OCR labels, 768-px legibility"]
        E{"D1 symbolic == D3 raster, all claims hold?"}
        M --> X --> E
    end
    U -- "beauty + id buffer + masks + manifest + claims (no plugin code, no payload)" --> K
    E -- yes --> C["CERTIFIED bundle + certificate.json"]
    E -- no --> Q["automatic quarantine (exact E_* code)"]
```

Plugins are treated as untrusted. They hand the kernel rasters and declarative JSON, never code or their
internal scene state. The kernel **re-derives every answer from renderer object masks, cross-checked
against the visible image via a beauty-only sanity oracle, colourblindness (CVD) gate, and stray-ink
witness.** A bundle is emitted only if that raster answer (D3) equals the generator's symbolic answer
(D1) and every plugin claim re-checks. Anything else goes to quarantine with an exact error code.

The trust boundary is enforced in code, and CI checks it on every push:
`import-linter` forbids `knotcore` from importing plugins, the pipeline, the network or geometry/UI
libraries, and the TCB audit (`kernel_audit.tcb_report`, SPEC D-38) allow-lists its imports, bans
network, process and dynamic-code modules, bans `SceneIR.payload` access, and pins a deterministic
`kernel_sha256`.

### Proof metrics

| Evidence | Result | Source |
|---|---|---|
| Fault-injection matrix M1–M14 (4 injections per fault) | **56 / 56 caught** with the expected code, **0 / 56 clean controls rejected** | `knot mutate --fault all --n 4` |
| A real render defect, caught by our own kernel | panel halo clipping broke the top cable at crowded crossings (4 / 120 clean panels failed the crossing-colour check); after the renderer fix, 0 / 400 | commit `aaf2528` |
| Golden self test | **66 / 66** bundles regenerated and re-verified; file-only `knot verify` catches a 1-bit id-buffer edit | `knot selftest`, `knot verify` |
| `rack_patch` yield, unmodified kernel | **10 / 10 certified** on the 10-render gate set, 8 / 10 on the first attempt (gate ≥ 80 %) | `tests/arena/test_rack_patch.py` |
| Structural diversity (Vendi, 256 items) | **2.1×–3.7×** a recolour-only baseline (tangle 3.73×, panel 2.95×, gauge 2.13×) | `knot diversity` |
| Variation controls | causal twins (minimal edit, answer changes) vs nuisance twins (restyle, answer kept); structural OOD holdout splits (held-out families and parameter ranges) | Studio → Batch & OOD Build |

## 5-page Factory Studio and Model Arena

![Forensics: 14/14 fault coverage and the Vision-vs-Reasoning twin comparison](docs/assets/studio_forensics.png)

`uv run streamlit run apps/inspector.py`. It runs offline: the Studio sets `KNOTWORK_NETWORK=deny` at
start-up. Every page has a **Guided mode** strip (step *n* of 5 with a Next step button), a **Page guide**
popover with steps and glossary terms, a **Quick tour** dialog, and a hover tooltip on each control,
KPI and leaderboard column.

| Page | What it does |
|---|---|
| **Studio** | Generate and certify one task for any plugin (Tangle, Gauge, Panel, Rack patch-bay) with presets; switch between Beauty / Witness Overlay / ID Buffer / CVD Simulation views; build batches with OOD holdouts and twins. |
| **Forensics** | KPI strip (yield, quarantine rate, 14 / 14 fault coverage); counterfactual twins that separate vision failures from reasoning failures (perceptual-clean, palette-swap, step-down and text-IR twins); quarantine browser; fault proofs that open on a committed default pair (`M3 → E_LABEL_MISMATCH`, [`samples/fault_proof/`](samples/fault_proof/)). The opening twin sample uses real certified renders paired with **labelled sample demo replies, not live VLM output**. To inspect real failures, send them live from Model Arena. |
| **Human Eval** | Human calibration lab: one blind certified item at a time, scored live against the benchmarked models. |
| **Export** | Harbor task folders (oracle `solve.sh`, `tests/test.sh` → reward), HF parquet, RLVR jsonl. |
| **Model Arena** | A **Sessions Sidebar** (search, clone, compare, archive, register an external folder). Each session keeps a frozen `data_dir` separate from its `results_dir`. A multi-block campaign builder covers every plugin, `rack_patch` included, from 1 to 100,000 items. Choose **Offline Mock** mode, or opt in to the OpenRouter vision catalog: the explicit **"Refresh live catalog"** button, plus a custom API key kept for the session only. Background campaigns are crash-safe and resumable (pause / stop / retry-failed-only / spend cap). Any wrong answer can be sent to Forensics with **"Send failure to Forensics"**. Optional **Repeats per item (k = 1–5)** adds ZeroBench-style reliability: pass@1, pass@k (solved at least once) and pass^k (solved on every repeat), with a finding whenever a model's pass@k − pass^k gap is 10 points or more. Reports export as `report.html`, `report.md`, `report_metrics.json`, `trials.csv` and an audit ZIP, with 95 % Wilson CIs, z / Fisher tests, inline SVG charts and a Methodology & Literature References section. |

## Quickstart (offline, 3 commands) or Docker

```bash
uv sync --locked                              # locked env, no API keys
uv run knot selftest --seeds-only             # determinism known-answer test (seconds)
uv run streamlit run apps/inspector.py        # 5-page Studio at http://localhost:8501
```

More CLI (all offline): `knot selftest` (golden end-to-end), `knot mutate --fault all --n 4` (M1–M14
matrix), `knot generate --plugin panel --n 20 --out data/panel`, `knot verify data/panel` (file-only,
runs no plugin code), `knot export --format harbor`, `knot bench --offline --client mock`, `knot report`.
Paid benchmarking is opt-in only: it needs a human approval file plus `OPENROUTER_API_KEY` (`SPEC.md` §1).

### Docker

```bash
docker build -t knotwork .
docker run --rm knotwork uv run knot selftest --seeds-only   # kernel known-answer test, then exit
docker run --rm -p 8501:8501 knotwork                        # the Studio at http://localhost:8501
```

The image is Python 3.12 slim built with `uv sync --locked`. It runs as the non-root `appuser` with
`KNOTWORK_NETWORK=deny`, `UV_OFFLINE=1` and no API keys baked in. Its default command serves the
Studio on `${PORT:-8501}`; self-hosted it has no caps and reads any folder you mount.

**Hosted sandbox.** The same image can run as a public Cloud Run demo with `KNOTWORK_PUBLIC_DEMO=1`
([`docs/DEPLOY_CLOUD_RUN.md`](docs/DEPLOY_CLOUD_RUN.md)). Every page then shows a sandbox banner, and all
storage is confined to `/tmp/knotwork_demo/` (paths outside it, including `..` traversal, are refused).
Runs are capped at 10 items per block, 3 blocks, k ≤ 3, 10-item Studio batches and a $1.00 spend cap.
The server never reads `OPENROUTER_API_KEY`: runs default to Offline Mock, and a visitor's own key stays
in their browser session. Unset or `0`, the switch changes nothing.

## Repository map (v2)

| Path | What |
|---|---|
| `src/knotwork/knotcore/` | trusted kernel (TCB): contracts, seeds, topology, OCR, witness, CVD, oracle, grading |
| `src/knotwork/plugins/` | untrusted scene plugins: `rack_patch`, `tangle`, `gauge`, `panel` (+ registry) |
| `src/knotwork/render/` · `pipeline/` | renderer (2× beauty, 1× non-AA masks and id buffer) · generate / certify / verify / mutate / twins |
| `src/knotwork/arena/` | Model Arena engine: sessions storage, catalog, resumable runner, report engine, Forensics loop |
| `src/knotwork/studio/` | Studio logic: factory, fault proofs, forensics, presentation, Human QA review log, public-demo guard |
| `src/knotwork/posttrain/` | post-training engine: grounded vCoT, RLVR reward + verifier server, curriculum and pack exporter, throughput |
| `src/knotwork/export/` · `bench/` · `probes/` · `freshness/` | Harbor / HF / RLVR exporters · staircase bench · shortcut probes · beacon seeds and burn ledger |
| `apps/` | Streamlit entry `inspector.py` and the 5 pages in `app_pages/` |
| `samples/` | browsable twins, live failures, live-run outputs, demo forensics, default fault proof |
| `tests/` | offline suite, tiered: `tier0` + `tier1` in `ci.yml`, `tier2` in `nightly.yml` |
| `Dockerfile` · `.github/workflows/` | offline image · CI (`ci.yml`) and nightly (`nightly.yml`) |
| `SPEC.md` · `VERIFICATION.md` · `DATA_CARD.md` · `BLOCKERS.md` | spec and decision log · gate results · data card · open limitations |

## Post-Training Engine (Grounded vCoT, RLVR Verifier & Adaptive GRPO)

`knotwork.posttrain` turns certified items and Arena sessions into training data and a verifier, offline
and without touching `knotcore` (it only calls the kernel's grader and topology primitives).

* **Grounded visual chain-of-thought (vCoT).** Each certified bundle yields a step-by-step trace
  whose every step is boxed on real pixels: terminal and label boxes from the manifest, crossings
  from the kernel's geodesic-ordered mask overlaps (over / under from the id buffer), dial, scale
  labels and needle from the gauge masks and the READ witness. The SFT target is
  `<think>Step k (...): ... <box>[x1, y1, x2, y2]</box> ...</think>` then `\boxed{answer}`, with
  boxes on a 0–1000 grid. No model writes it, so it cannot hallucinate. Studio shows it as the
  **vCoT Steps** view.
* **Anti-reward-hacking RLVR reward.** `knotwork.posttrain.reward.compute_score(data_source,
  solution_str, ground_truth, extra_info)` is the native `verl` custom-reward signature. It returns
  `score` = 0.60 outcome (the kernel grader on `\boxed{}`) + 0.25 spatial grounding (one-to-one
  greedy IoU against the certified boxes, divided by max(predicted, certified), so spraying boxes
  does not pay) + 0.15 format and calibrated abstention. A wrong answer earns half the process
  credit, so it stays below 0.25. A right answer with zero grounding raises `reward_hack_flag`.
* **Stateless verifier server.** It binds to localhost only, has no outbound network
  access, and works under `KNOTWORK_NETWORK=deny`:

  ```bash
  uv run knot serve-verifier --host 127.0.0.1 --port 8090
  curl -s localhost:8090/health
  curl -s -X POST localhost:8090/score -d '{"solution_str": "<think>...</think>\\boxed{07}", "ground_truth": "07"}'
  ```

  `POST /score_batch` takes `{"items": [...]}` from distributed rollout workers.
* **Adaptive curriculum and Post-Training Pack.** Model Arena's **Post-Training Pack** tab (or
  `uv run knot posttrain-pack --results-dir <session>/results`) sorts every block and model by
  accuracy into **Saturated** (> 75 %), **GRPO Frontier** (20–75 %, where a binary-reward GRPO
  group's variance p(1−p) is large) and **SFT Cold-Start** (< 20 %, vanishing advantage). It then
  writes `sft_grounded_vcot.jsonl`, `dpo_preferences.jsonl` (real failed replies as rejected),
  `grpo_rlvr_verl.parquet` / `.jsonl`, and `curriculum_spec.json`. Training seeds start strictly above
  every evaluated seed, with a fresh OOD eval range above them; generate them with
  `knot posttrain-generate --spec ... --split train`. It also writes `rlvr_kit/`
  (a self-contained `knotwork_verl_reward.py`, `run_verl_grpo.sh` using
  `reward.custom_reward_function.path` / `.name`, and `run_trl_grpo.py`) and
  `provenance_attestation.json` (git and kernel hashes, dataset and eval-seed hashes,
  `train_seed_disjoint`, and the one external asset, the bundled DejaVu font), all in one ZIP.
  The Arena's **Temperature** control (default 0) makes k > 1 repeats stochastic, which
  separates pass@k from pass^k and yields Best-of-N preference pairs.
* **Throughput and unit economics.** This command measures `gen_ms`, `verify_ms` and `vcot_ms`
  per item and prints its cost formula:

  ```bash
  uv run knot bench-throughput --domain rack_patch --n 4 --vcpu-usd-hr 0.04
  ```

  On a laptop core: 4 / 4 certified rack items, about 544 certified items per hour, about $74 per
  1M certified samples at $0.04 per vCPU-hour. Arena reports include the same table from each
  session's observed timings.

Credits: grounding every reasoning step in a box, and scoring answer–grounding consistency, follows
**MM-GCoT** (Wu et al., CVPR 2026 [8]). Rewarding the visual-evidence trajectory, not only the final
answer, in multimodal RLVR follows **TGRL** (Lu et al., 2026 [9]). Knotwork's version is
deterministic: the reference trajectory comes from certified masks, not a stronger model.

## MLOps, CI Release Gating & Plugin SDK

* **Regression pinning and `knot gate`.** In Model Arena's Item inspector, **Pin to Regression** (next
  to *Send failure to Forensics*) adds a failed item to `<results>/regression_suite.yaml`: its seed,
  preset, question family and certified answer. Pinning it again updates the case. The session view
  offers the file for download, with a ready CI line:

  ```bash
  uv run knot gate --suite regression_suite.yaml --model openai/gpt-5.4 --mode mock --json-out gate.json
  ```

  Exit codes: `0` when accuracy ≥ `min_accuracy` (default 0.85), `1` below it (merge-blocking), `2` for a
  missing, invalid or empty suite. `--mode live` re-renders each case from its seed, checks it still
  certifies to the pinned answer, and asks the model through OpenRouter (key from the environment).
* **Declarative campaigns.** **Export campaign.yaml** (on a session or in the builder) writes the blocks,
  models and run settings; **Import campaign.yaml** loads one back into the builder. API keys are never
  written: the schema has no key field and refuses secret-shaped values. Run one headlessly:

  ```bash
  uv run knot arena-run --campaign campaign.yaml --report   # resumable; writes report.html / .md / _metrics.json
  ```
* **Human QA triage.** Human Eval (after each answer) and the Arena Item inspector show an
  Unreviewed / Approved / Flagged badge with **Approve** and **Flag** buttons and an optional note,
  appended to `review_log.jsonl` (the latest decision per item wins). **Approved items only** in Export
  and in the Post-Training Pack (`knot posttrain-pack --approved-only`) keeps just the approved items and
  records `qa_filter: approved_only` and `approved_count` in the manifest or provenance attestation. The
  toggle is off by default, so existing exports are unchanged.
* **Agent-native docs and plugin wheels.** [`llms.txt`](llms.txt) is a machine-readable summary (also
  served as `GET /llms.txt` by `knot serve-verifier`), and [`AGENTS.md`](AGENTS.md) is the guide for
  coding agents. External task plugins install as wheels through the `knotwork.plugins` entry point
  group ([`docs/PLUGIN_SDK.md`](docs/PLUGIN_SDK.md)). `uv run knot plugins --json` lists built-in and
  external plugins with their source, version and status, and rejects name collisions and incomplete
  protocols with a reason.

## Related work and research positioning

| Work | What it established | What Knotwork adds |
|---|---|---|
| **BlindTest**, *Vision language models are blind*, Rahmanzadehgervi et al., ACCV 2024 [1] | VLMs that score well on chart and document benchmarks still fail simple 2-D tasks such as counting line intersections, spotting overlaps and following paths. | Takes the same weakness to industrial scenes (2U rack patch bays, control panels, analog gauges) and certifies every label with the TCB kernel; the fault matrix catches 56 / 56 injected defects. |
| **The Vendi Score**, Friedman & Dieng, TMLR 2023 [2] | A reference-free diversity metric (an effective number of distinct samples). It measures diversity, not quality. | Reports Vendi only over items the kernel has certified with 0 `E_*` codes, so the 2.1×–3.7× gain over recolouring is diversity among valid items. |
| **VLind-Bench**, Lee et al., Findings of NAACL 2025 [3]; **PriVE-Bench / PriVE-Tools**, Sun et al., EMNLP 2026 [4] | Counterfactual image pairs separate answers driven by visual evidence from answers driven by language or category priors. | Generates the counterfactuals parametrically and certifies them: causal vs. nuisance twins plus Perceptual Clean, Palette Swap, Step Down and Text IR twins, inside Forensics and the Model Arena failure loop. |
| **ZeroBench**, Roberts et al., ICML 2026 [5] | 100 hand-built questions designed to be unsolved at release, scored with pass@k and the stricter pass^k (correct on every one of k attempts). | Adopts pass@1 / pass@k / pass^k in Model Arena, over seed-controlled programmatic items with OOD holdout splits instead of a fixed hand-authored set. |
| **VisualSphinx**, Feng et al., 2025 [6] | Large-scale synthetic vision-logic puzzles for RL (GRPO), with LLM and VLM scoring used to filter quality. | Replaces model-based quality checks with an offline deterministic verifier, and exports Harbor tasks and RLVR data whose rewards come from certified answers. |

## References and credits

1. P. Rahmanzadehgervi, L. Bolton, M. R. Taesiri, A. T. Nguyen. *Vision language models are blind.* ACCV 2024. [arXiv:2407.06581](https://arxiv.org/abs/2407.06581)
2. D. Friedman, A. B. Dieng. *The Vendi Score: A Diversity Evaluation Metric for Machine Learning.* TMLR 2023. [arXiv:2210.02410](https://arxiv.org/abs/2210.02410)
3. K. Lee et al. *VLind-Bench: Measuring Language Priors in Large Vision-Language Models.* Findings of NAACL 2025. [arXiv:2406.08702](https://arxiv.org/abs/2406.08702)
4. J. Sun et al. *Seeing What Is Actually There: PriVE-Bench and PriVE-Tools for Counterfactual Evaluation of Agentic Visual Evidence in VLMs.* EMNLP 2026. [ACL Anthology 2026.emnlp-main.760](https://aclanthology.org/2026.emnlp-main.760/)
5. J. Roberts et al. *ZeroBench: An Impossible Visual Benchmark for Contemporary Large Multimodal Models.* ICML 2026 (PMLR 306). [arXiv:2502.09696](https://arxiv.org/abs/2502.09696)
6. Y. Feng et al. *VisualSphinx: Large-Scale Synthetic Vision Logic Puzzles for RL.* 2025. [arXiv:2505.23977](https://arxiv.org/abs/2505.23977)
7. E. B. Wilson. *Probable inference, the law of succession, and statistical inference.* JASA 22(158), 1927. Used for every 95 % binomial interval.
8. Q. Wu, X. Yang, Y. Zhou et al. *Grounded Chain-of-Thought for Multimodal Large Language Models.* CVPR 2026. [arXiv:2503.12799](https://arxiv.org/abs/2503.12799)
9. J. Lu, J. Wu, J. Li, K. Huang, S. Yang, M. Chen, J. Wu, K. Liu, X. Wang. *Beyond Where to Look: Trajectory-Guided Reinforcement Learning for Multimodal RLVR.* 2026. [arXiv:2603.26126](https://arxiv.org/abs/2603.26126)

## Honest disclosures and current limits

* **Blocker 3.6, shortcut probes (`p3.probes`):** the original tangle T1/T3 and premise-family leaks
  now pass in v2. One new row still fails: panel P1 image statistics, **0.173 vs a 0.164 limit**. The
  gate fails on that single row, and the threshold is unchanged.
* **Blocker 5.6, mock staircase (`p5.bench_mock`):** passes with a distinct item on every trial. Mock A
  panel sits exactly at the **1.5-level boundary** because the split has few panel items with 6 or more
  crossings.
* **Kernel size:** the original 850 / 1350 LOC caps are now informational telemetry (SPEC D-38). The
  boundary is enforced by the TCB contract and import isolation instead.
* **CI scope:** `ci.yml` runs `tier0` + `tier1` on every push. `tier2` (bulk generation, probes, the
  10-render `rack_patch` gate) and the phase gates run in `nightly.yml` on `main`.
* Scenes are synthetic 2-D renders. They do not support claims about natural-image perception or
  real instruments ([`DATA_CARD.md`](DATA_CARD.md)). Full details are in [`BLOCKERS.md`](BLOCKERS.md).

## Team Bilota

Built by **Swarit Sharma** and **Pankaj David** for the Innodata AI Hackathon 2026, Track 2.

© 2026 BilotaAI. Code: Apache-2.0 ([`LICENSE`](LICENSE)). Bundled font: DejaVu Sans (Bitstream Vera
license, `assets/fonts/LICENSE-DejaVu.txt`).
