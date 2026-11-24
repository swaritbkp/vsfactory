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

## Status

Built phase by phase; see `TASKS.md` and `VERIFICATION.md` (Phase 6) for gate results.

## License

Code: Apache-2.0. Bundled font: DejaVu Sans (Bitstream Vera license, `assets/fonts/LICENSE-DejaVu.txt`).
