# Samples

Small, browsable excerpts of Knotwork output (about 1.6 MB). Images are 768-px JPEG **previews** at the
resolution models see; the full bundles (lossless PNG beauty, 16-bit id buffer, amodal masks, manifest,
claims) regenerate with `uv run knot selftest`, which writes them to `artifacts/selftest/data/`.
Regeneration is bit-exact on the platform that recorded the golden and answer-exact elsewhere
(SPEC D-35).

## `twins/<plugin>/{base,causal,nuisance}/`: one certified twin triple per plugin

From the golden selftest (seed 1234). Each folder holds `image.jpg`, the full `task.json` (question
program, answer, kernel certificate with every step's evidence, file digests, canary) and a short
`card.json`.

| Plugin | Family | Base answer | Causal twin (answer changes) | Nuisance twin (answer kept) |
|---|---|---|---|---|
| tangle | T1_TRACE | `1` | `2`: end labels 1 and 2 swapped (0.03 % of pixels differ) | `1`: restyled (74 % of pixels differ) |
| gauge | G1_READ | `15` | `78`: needle moved (0.5 % of pixels) | `15`: restyled (95 %) |
| panel | P1_READ_TRACE | `7` | `50`: traced gauge's needle moved (0.3 %) | `7`: restyled (75 %) |

All nine bundles are `CERTIFIED`: the kernel re-derived the answer from pixels (D3) and it matched the
generator's symbolic answer (D1), with every claim re-checked.

## `live_failures/`: three items all three models answered wrong

From the live OpenRouter run (`live_run/`). `failure.json` holds the question, the certified answer and
each model's reply; `task.json` is the bundle's full record.

| Folder | Question | Truth | GPT-5.4 | Gemini 2.5 Flash | Qwen3-VL-235B |
|---|---|---|---|---|---|
| `1_trace` | trace the cable from terminal N (1 crossing) | `8` | `7` | `7` | `7` |
| `2_gauge_read` | what value does gauge G1 read? | `14` | `20` | `10` | `0` |
| `3_false_premise` | "since the gauge on socket E is in the green zone …" | `no; 67` | `no; 65` | `yes; 30` | `yes; 40` |

## `live_run/`: the live benchmark's own outputs

`bench_report.json` (per-model statistics), `trials.jsonl` (all 997 trials: block, item key, certified
answer, extracted reply, correctness, cost) and `bench_live_summary.json` (the summary behind
`README.md` and `docs/figures/`). The raw response cache stays local (SPEC D-36).
