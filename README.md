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

**15-second fast path.** After `uv sync --locked`, this builds a self-contained HTML report with
the benchmark and premise-control tables and four figures (fig01–fig04) from the committed live-run
results in `samples/live_run`. The gate, fault-matrix and probe sections, fig05 and fig06 need
locally generated `artifacts/`; all six figures are committed in [`docs/figures/`](docs/figures/).

```bash
uv run pytest tests/apps/test_report.py && uv run knot report   # -> artifacts/report/report.html
```

**Full offline quickstart** (about 28 min on 2 cores: `selftest` ~8 min, `generate` ~7 min,
`verify` ~6 min, `mutate` ~5 min, the rest under a minute each). The long commands print nothing
until they finish, so a silent terminal means they are still working, not hung.

```bash
uv sync --locked                       # Python 3.11 env from uv.lock
uv run knot selftest --seeds-only      # determinism known-answer test (works from day one)
uv run knot selftest                   # golden end-to-end self test
uv run knot generate --plugin panel --n 20 --out data/panel      # beacon defaults to --offline
uv run knot mutate --fault all --n 4   # M1–M10 fault-injection matrix: 100 % caught
uv run knot verify data/panel          # file-only re-verification, no plugin code executed
uv run knot export --format harbor --data data/panel --out export/   # Harbor task folders
uv run knot bench --offline --client mock --data data/panel --out bench/   # simulated VLMs
uv run knot report                     # self-contained artifacts/report/report.html (figures + tables)
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
per adaptive staircase plus the diagnostic blocks of `SPEC.md` §12, followed by an additive 72-trial
true-premise control (SPEC D-37). **943 paid calls / 1,069 trials, 0 HTTP errors**: main run $0.9992
(871 calls, 997 trials), control $0.0756 (72 calls), plus $0.0501 for two pre-run canaries, **$1.12
total**. Every response is cached in replay format; `knot bench --client replay` reproduces the main
run's per-model results exactly, and `python -m knotwork.bench.p5_control` replays the control, both
with 0 network calls.

| | GPT-5.4 | Gemini 2.5 Flash | Qwen3-VL-235B |
|---|---|---|---|
| OpenRouter slug | `openai/gpt-5.4` (reasoning off) | `google/gemini-2.5-flash` | `qwen/qwen3-vl-235b-a22b-instruct` |
| Main run: trials / errors / parse failures | 325 / 0 / 0 | 336 / 0 / 1 | 336 / 0 / 1 |
| Main run: cost · wall-clock | $0.808 · 6.1 min | $0.121 · 8.1 min | $0.070 · 12.4 min |
| **Overall accuracy** | **25.2 %** | **20.5 %** | **20.2 %** |
| Tangle (cable tracing) · panel · gauge reading | 27.3 · 22.8 · 15.6 % | 17.0 · 24.4 · 25.0 % | 21.0 · 19.4 · 6.2 % |
| Colour ablation: unique colours → monochrome | 59.4 → 3.1 % | 18.8 → 9.4 % | 37.5 → 15.6 % |
| Twins: original / causal / nuisance / all three | 29 / 12.5 / 25 / 12.5 % | 25 / 12.5 / 29 / 12.5 % | 25 / 12.5 / 25 / 8.3 % |
| "Yes" on **true** premise (`P5` control, n = 24) | **0 %** (0) | **62.5 %** (15) | **95.8 %** (23) |
| "Yes" on **false** premise (`T_GASLIGHT`, n = 24) | **0 %** (0) | **50 %** (12) | **100 %** (24) |
| Premise gap (true − false) · exact McNemar p | 0 pts · 1.00 | +12.5 pts · 0.45 | −4.2 pts · 1.00 |
| Unpaired Fisher p · balanced accuracy | 1.00 · 50 % | 0.56 · 56 % | 1.00 · 48 % |
| Dial reading correct on these items: true · false prompt (of 24) | 1 · 0 | 2 · 4 | 4 · 4 |
| Staircase threshold (crossings): tangle · panel | 1.5 · 1.5 | 1.1 · 1.7 | 1.1 · 1.5 |

Findings: (1) tracing a cable through even one or two crossings is essentially unsolved by all three
models; (2) the colour collapse is led by GPT-5.4: 59 % with unique cable colours vs 3 % when every
cable looks the same, below the ~17 % chance level (32 matched pairs, exact McNemar p < 0.001;
Gemini p = 0.38 and Qwen p = 0.065 are not significant); (3) twins, directional only: accuracy drops
from 25-29 % on the original to 12.5 % on the causal twin (two end labels swapped, cables unchanged)
but not on the style-only nuisance twin; with n = 24 triples per model the per-model drops are not
significant (exact McNemar p = 0.125 / 0.25 / 0.375); (4) **no model detectably checked the premise**.
On matched pairs, GPT-5.4 answered "no" to all 48 prompts and Qwen "yes" to 47 of 48, so their 0 % and
100 % on false premises reflect a default answer, not a check of the premise against the image.
Gemini said "yes" somewhat more often when the premise was true (15 vs 12 of 24; 5 pairs moved one
way, 2 the other), which is not significant.

**How the control works.** Each of the 24 false-premise scenes was re-certified with one change: the
claimed zone became the zone the traced gauge is really in. All 24 pairs passed: source bundle
re-verified from its files, control `CERTIFIED`, the kernel's own ZONE step equal to the claimed zone,
and pixel-identical images. The prompts differ in that one word; parser, seed, temperature and model
settings are unchanged.

Caveats, stated plainly: accuracy is at the floor, so the staircases (no reversals for two models)
bound rather than measure the threshold, and the per-crossing error MLE and the composition product
law are not informative on this run. For the premise control: 24 pairs detect only a large gap (at
least 6 changed pairs, all in one direction, for p < 0.05), so a modest Gemini effect is not ruled
out; reading to ±2 % on the same items is near the floor, and this design cannot say whether a model
fails at tracing the cable, at reading the zone or at using it; the control was added after the main run, and OpenRouter served Qwen from a
slightly different provider mix (Venice 23 of 24 false prompts, 19 of 24 true); claimed zone words
are not balanced between the arms (true: 7 green, 8 amber, 9 red; false: 10, 8, 6). Both prompts
force a yes/no answer, so abstention is not measured. There is no
gauge-only track: "gauge reading" is the read-only question on panel images (±2 % of the scale).
Figures: `docs/figures/fig01…fig06`; full report: `uv run knot report` → `artifacts/report/report.html`;
browsable task packages and the three failure items: [`samples/`](samples/).

Reproducibility: the committed CI fixtures (`tests/bench/fixtures/`) replay offline at $0 on every
push; the full live replay cache (`bench_replay/`) and burn ledger stay local by design (SPEC D-35 /
D-36). The live results are committed in [`samples/live_run/`](samples/live_run/)
(`bench_report.json`, per-trial `trials.jsonl`, `bench_live_summary.json`, and the control's
`p5_control_trials.jsonl` and `p5_control.json`) and rebuilt into `report.html` by
`uv run knot report`, which recomputes the premise statistics from the per-trial records.

## Status

Built phase by phase; see `TASKS.md`, `PROGRESS.md` and `BLOCKERS.md` for gate results and the two
open items (shortcut-probe residuals in the premise families and tangle T1/T3; the mock staircase
tolerance).

## License

Code: Apache-2.0. Bundled font: DejaVu Sans (Bitstream Vera license, `assets/fonts/LICENSE-DejaVu.txt`).
