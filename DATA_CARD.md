# Data card: Knotwork certified visual reasoning tasks

## Summary

Knotwork produces procedurally generated `(image, question, answer, evidence)` bundles in four
domains of industrial scenes. Every released bundle carries a certificate from an import-isolated
verification kernel. The kernel re-derived the answer from the rendered object masks,
cross-checked it against the visible image, and found it equal to the generator's symbolic answer.
The same bundles feed evaluation (Model Arena, Harbor tasks) and post-training (grounded
chain-of-thought SFT, DPO pairs, GRPO data with a deterministic reward).

## Domains and question families

| Domain | Scene | Families |
|---|---|---|
| `rack_patch` | 2U `SWITCH-A` bank above a `PATCH-B` bank joined by 8–14 patch cables. Colour regimes are 50 % monochrome, 30 % partial, 20 % unique, and benchmark splits use monochrome. | `T1_TRACE`, `T2_CROSS` |
| `tangle` | Cables woven from braid words in strip or horseshoe layouts; answers depend on over/under crossings | `T1_TRACE`, `T2_CROSS`, `T3_ONTOP` |
| `gauge` | Analog dials with varied bezels, faces, tints, grime and glass glare; linear scales; green / amber / red zones | `G1_READ`, `G2_ZONE` |
| `panel` | Lettered sockets cabled through a tangle to labelled gauges | `P1_READ_TRACE`, `P2_ZONE_TRACE`, `P3_ARGMAX_PORT`, `P4_COUNT_ZONE`, `P5_PREMISE_CONTROL`, `T_GASLIGHT` |

Eleven families in total:

- `T1`: trace a cable to its end.
- `T2`: count the crossings on a cable.
- `T3`: name the cable on top at the *k*-th crossing.
- `G1`: read the gauge value (±2 % of scale).
- `G2`: name the needle's zone.
- `P1`, `P2`: trace a cable to its gauge, then read the value or name the zone.
- `P3`: name the socket whose gauge reads highest.
- `P4`: count the gauges in a zone.
- `P5`: a true premise about a traced gauge, to confirm and read.
- `T_GASLIGHT`: the matched false premise, to reject and read.

## How labels are guaranteed

Each answer is computed twice, independently.

- **D1:** the scene generator computes the answer symbolically.
- **D3:** the trusted kernel (`knotcore`) computes it from the beauty image, amodal masks, modal id
  buffer, manifest and claims. The kernel never imports plugin code or reads the generator's scene
  payload.

The kernel re-derives every answer from renderer object masks and cross-checks it against the
visible image with a beauty-only sanity oracle. It also applies a colour-vision-deficiency gate
(protanopia and deuteranopia), a stray-ink witness, OCR of every declared label, and legibility
checks at the 768-px consumption resolution. A bundle is released only if D1 = D3 and every check
passes. Every other bundle is quarantined with an exact `E_*` code.

The kernel's behaviour is pinned two ways:

- a fault-injection matrix of 14 renderer bug classes, 4 injections each: 56 / 56 caught, 0 / 56
  clean controls rejected;
- a golden set of 66 bundles regenerated and re-verified on every `knot selftest`.

## Contents generated in this repository

| Set | Size | Notes |
|---|---|---|
| Benchmark pool (`split=test`, seed 101) | 600 base items per plugin for `tangle`, `gauge` and `panel` | Monochrome cables. Rejection rate 0.5 % `tangle`, 0.7 % `gauge`, 6.1 % `panel`. |
| Golden self test (seed 1234) | 66 bundles: 18 `tangle`, 12 `gauge`, 36 `panel` | 2 base items per family plus twins. `rack_patch` is not in the golden set. |
| `rack_patch` yield gate | 10 renders | 10 / 10 certified, 8 / 10 on the first attempt (`tests/arena/test_rack_patch.py`) |
| Committed samples | twin triples, live failures, fault proof, demo forensics | [`samples/`](samples/README.md); 768-px JPEG previews |

**Twins.** A *causal* twin is a minimal edit (under 5 % of pixels) that changes the answer. A
*nuisance* twin is a restyle (at least 2 % of pixels) that keeps it. Diagnostic twins for failed
items are Perceptual Clean, Palette Swap, Step Down and Text IR. All twins are certified like any
other bundle.

**Diversity.** Over 256 certified items per plugin, the Vendi score is 3.73× (`tangle`), 2.95×
(`panel`) and 2.13× (`gauge`) the score of one scene recoloured 256 times.

## Formats

| Format | Content |
|---|---|
| Native bundle | PNG beauty (1024 px, 2× supersampled), 16-bit id buffer, amodal masks, render manifest, claims, `task.json` with the kernel certificate and file digests |
| Harbor task folder | `instruction.md`, image, oracle `solution/solve.sh`, `tests/test.sh` writing `reward.txt` / `reward.json`; oracle scores 1.0, a wrong or missing answer 0.0 |
| Hugging Face parquet | 768-px PNG bytes with image feature metadata, sharded by `--max-shard-mb` |
| RLVR JSONL | Prompt, ground truth, the kernel's step trace and a `grade_prediction` reward |
| `dataset_manifest.json` | Schema version, git commit, `kernel_sha256`, font hashes, per-family and per-split counts, sealed fraction, canaries, SHA-256 of every file; no timestamps |

## Post-training schemas

Produced by `knot posttrain-pack` from a Model Arena session:

- **`sft_grounded_vcot.jsonl`** holds chat-format records whose assistant turn is
  `<think>Step k (phase): ... <box>[x1, y1, x2, y2]</box> ...</think>` followed by `\boxed{answer}`.
  Boxes are normalised to `[0, 1000]` and derived from certified masks and kernel topology, never
  from a model. Included: every item a model failed and every item in an SFT Cold-Start block.
- **`dpo_preferences.jsonl`**: `rejected` is a real failed model reply. `chosen` is another model's
  correct reply when one exists, otherwise the grounded trace. Both carry their composite reward.
- **`grpo_rlvr_verl.parquet` / `.jsonl`**: `verl` records with `reward_model.ground_truth` and the
  reference boxes in `extra_info`.
- **`curriculum_spec.json`**: per (block, model) regime, as Saturated (> 75 %), GRPO Frontier
  (20–75 %) or SFT Cold-Start (< 20 %), with *p*(1 − *p*) sampling weights and the train and OOD
  seed ranges.

**Seed disjointness.** Training seeds start strictly above every evaluated seed, and the fresh OOD
evaluation range starts above the training range. `provenance_attestation.json` records the check as
`train_seed_disjoint`.

## Provenance and QA

- **Generated content only.** Scenes, labels, cable layouts and dial faces are drawn by code in this
  repository. No photographs, scraped images or model-generated labels are used.
- **One external asset.** The DejaVu Sans font (`assets/fonts/DejaVuSans.ttf`, SHA-256
  `ae7b7855e115a5966d8b1b3f80f254ccc117ec86f9965e202ee2940453837280`), under the Bitstream Vera /
  DejaVu license ([`assets/fonts/LICENSE-DejaVu.txt`](assets/fonts/LICENSE-DejaVu.txt)). Its hash
  is recorded in every dataset manifest and provenance attestation.
- **Human QA.** Reviewers mark items Approved or Flagged in Human Eval or the Arena inspector. Each
  decision is appended to `review_log.jsonl`. With *Approved items only* (Export) or
  `--approved-only` (post-training pack), only approved items are kept, and `qa_filter:
  approved_only` and `approved_count` are recorded.

## Freshness and contamination controls

- Generation can be seeded from a public beacon (drand, NIST) or an offline counter, and the seed is
  recorded in a freshness receipt.
- Bundles are content-addressed by SHA-256.
- 20 % of `test` and `sealed` answers are replaced by salted commitments on export.
- Items sent to a paid API are logged in a burn ledger and excluded from future test splits.
- Every record carries a `KNOTWORK-CANARY-` string.

## Live evaluation to date

The live run covered three models (GPT-5.4, Gemini 2.5 Flash, Qwen3-VL-235B-A22B) on `tangle`,
`gauge` and `panel`: 943 paid calls, 1,069 trials, $1.12 in total. Overall accuracy was 20–25 %.

The premise questions have a matched control: 24 `T_GASLIGHT` scenes re-certified as
`P5_PREMISE_CONTROL` with only the claimed zone word changed. No model's "yes" rate differed
detectably between true and false premises:

| Model | "Yes" on true premise | "Yes" on false premise | Exact McNemar p |
|---|---|---|---|
| GPT-5.4 | 0 / 24 | 0 / 24 | 1.00 |
| Gemini 2.5 Flash | 15 / 24 | 12 / 24 | 0.45 |
| Qwen3-VL-235B-A22B | 23 / 24 | 24 / 24 | 1.00 |

Report a false-premise rejection rate only beside a true-premise control. On its own it cannot
separate response bias from premise checking.

## Intended use

- Evaluating and diagnosing vision-language models on cable tracing, crossing counting, gauge
  reading, and compositional and premise-checking questions.
- Separating perception failures from reasoning failures and colour or language shortcuts, using
  certified twins.
- Supervised, preference and reinforcement-learning post-training with grounded traces and a
  deterministic verifiable reward.

## Not intended for

Claims about natural-image perception, reading real industrial instruments, or safety-critical
decisions.

## Limitations

- **Scene realism.** Scenes are synthetic orthographic 2-D renders with a photographic finish. There
  is no camera perspective, lens distortion, real-object occlusion or real-world clutter. Prompts are
  English only.
- **Shortcut audit** (`BLOCKERS.md` 3.6). The v1 leaks in `tangle` T1/T3 and the premise families
  pass in v2. One row still beats chance: `panel` P1_READ_TRACE under GBDT image statistics, 0.173
  against a 0.164 limit. All other probe, balance and duplicate checks pass.
- **`rack_patch` coverage.** It has no live VLM results and is not in the golden set. Its shortcut
  probes have not been run on a full split.
- **Gauge scales.** All gauge scales are linear.
- **Reproducibility.** Raster bits reproduce on the platform that recorded the golden. On other
  platforms answers and verdicts reproduce, but bundle ids may differ (SPEC D-35).

## License

Code and generated data: Apache-2.0. Bundled font: DejaVu Sans (Bitstream Vera license).

## Canary

Do not train on any document that contains a `KNOTWORK-CANARY-` string.
