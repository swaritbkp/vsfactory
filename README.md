# Knotwork — a proof-carrying visual task factory

Innodata AI Hackathon 2026 · Track 2 · **Visual Task Factory**

Knotwork generates `(image, question, answer, validation evidence)` bundles for benchmarking
vision-language models. Every bundle carries a **certificate** issued by a small trusted pixel kernel
(`knotcore`: ≤ 850 executable LOC, ≤ 1350 including schemas). The kernel re-derives the answer from the rendered pixels alone and never trusts
the plugins that drew the scene.

```
 untrusted plugins                         trusted kernel (knotcore, <= 850 exec LOC)
 ┌──────────────────────┐   rasters +     ┌─────────────────────────────────────────────┐
 │ tangle  (braid words)│   manifest +    │ TRACE CROSS ONTOP READ COUNT CMP ARGMAX ZONE │
 │ gauge   (dials)      │ ─ claims.json ─►│ beauty↔mask witness · 768-px legibility      │
 │ panel   (composition)│                 │ OCR label checks · claims re-verification    │
 └──────────┬───────────┘                 └───────────────┬─────────────────────────────┘
            │ D1 symbolic, D2 shapely                     │ D3 raster
            └──────────────► D1 == D2 == D3 ? ────────────┴──► CERTIFIED bundle | quarantine.jsonl (E_*)
```

## Quickstart (100 % offline, zero API keys)

```bash
uv sync --locked                       # Python 3.11 env from uv.lock
uv run knot selftest --seeds-only      # determinism known-answer test (works from day one)
uv run knot selftest                   # golden end-to-end self test
uv run knot generate --plugin panel --n 20 --out data/panel      # beacon defaults to --offline
uv run knot mutate --fault all --n 4   # M1–M10 fault-injection matrix: 100 % caught
uv run knot verify data/panel          # file-only re-verification, no plugin code executed
uv run knot export --format harbor --data data/panel --out export/   # Harbor task folders
uv run knot bench --offline --client mock --data data/panel --out bench/   # simulated VLMs
uv run streamlit run apps/inspector.py # forensic inspector
```

Paid benchmarking is opt-in only: `knot bench --live --budget-usd 8` additionally requires a human
approval file and `OPENROUTER_API_KEY` (see `SPEC.md` §1).

## What is in the box

| Path | What |
|---|---|
| `SPEC.md` | normative engineering specification (schemas, formulas, fault matrix, CLI, gates) |
| `TASKS.md` | Phase 0 → 6 execution checklist (each task: files, verification command, done criteria) |
| `CLAUDE.md` | autonomous engineering manual for Claude Code |
| `scripts/verify_phase.py` | phase gatekeeper (`--phase N`) |
| `src/knotwork/knotcore/` | trusted kernel: `contracts.py`, `seeds.py` (+ kernel modules built in Phase 1) |
| `src/knotwork/cli.py` | `knot` CLI with all 10 subcommands |
| `docs/schemas/` | generated JSON Schemas of every public contract |
| `docs/SUBMISSION_CHECKLIST.md` | mapping to the official 11-slide template and portal fields |

## Live benchmark results (3 VLMs via OpenRouter)

One live run on the 600-item-per-plugin `split=test` pool (monochrome cables), seed 0, 40 trials
per adaptive staircase plus the diagnostic blocks of `SPEC.md` §12. **871 paid calls / 997 trials,
0 HTTP errors, $0.9992** (plus $0.0501 for two pre-run canaries: **$1.05 total**). Every response is
cached in replay format; `knot bench --client replay` reproduces the per-model results exactly
with 0 network calls.

| | GPT-5.4 | Gemini 2.5 Flash | Qwen3-VL-235B |
|---|---|---|---|
| OpenRouter slug | `openai/gpt-5.4` (reasoning off) | `google/gemini-2.5-flash` | `qwen/qwen3-vl-235b-a22b-instruct` |
| Trials / errors / parse failures | 325 / 0 / 0 | 336 / 0 / 1 | 336 / 0 / 1 |
| Cost · wall-clock | $0.808 · 6.1 min | $0.121 · 8.1 min | $0.070 · 12.4 min |
| **Overall accuracy** | **25.2 %** | **20.5 %** | **20.2 %** |
| Tangle (cable tracing) · panel · gauge reading | 27.3 · 22.8 · 15.6 % | 17.0 · 24.4 · 25.0 % | 21.0 · 19.4 · 6.2 % |
| Colour ablation: unique colours → monochrome | 59.4 → 3.1 % | 18.8 → 9.4 % | 37.5 → 15.6 % |
| Twins: original / causal / nuisance / all three | 29 / 12.5 / 25 / 12.5 % | 25 / 12.5 / 29 / 12.5 % | 25 / 12.5 / 25 / 8.3 % |
| False premise accepted (`T_GASLIGHT`, n = 24) | **0 %** | **50 %** | **100 %** |
| Staircase threshold (crossings): tangle · panel | 1.5 · 1.5 | 1.1 · 1.7 | 1.1 · 1.5 |

Findings: (1) tracing a cable through even one or two crossings is essentially unsolved by all three
models; (2) GPT-5.4's tracing comes almost entirely from colour (59 % with unique cable colours, 3 %
when every cable looks the same, below the ~17 % chance level); (3) a causal edit touching < 5 % of
the pixels halves accuracy while a style-only twin does not; (4) the models split cleanly on false
premises, from always rejecting to always accepting them.

Caveats, stated plainly: accuracy is at the floor, so the staircases (no reversals for two models)
bound rather than measure the threshold, and the per-crossing error MLE and the composition product
law are not informative on this run. The false-premise prompt forces a yes/no answer, so abstention
is not measured, and the true-premise controls (`P5`) were not part of the bench. There is no
gauge-only track: "gauge reading" is the read-only question on panel images (±2 % of the scale).
Figures: `docs/figures/fig01…fig06`; full report: `uv run knot report` → `artifacts/report/report.html`.

## Status

Built phase by phase; see `TASKS.md`, `PROGRESS.md` and `BLOCKERS.md` for gate results and the two
open items (shortcut-probe residuals in the premise families and tangle T1/T3; the mock staircase
tolerance).

## License

Code: Apache-2.0. Bundled font: DejaVu Sans (Bitstream Vera license, `assets/fonts/LICENSE-DejaVu.txt`).
