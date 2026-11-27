# AGENTS.md: working in the Knotwork repository

A short operating guide for coding agents. [`CLAUDE.md`](CLAUDE.md) is the full engineering manual
and wins on any detail; this file never contradicts it. [`README.md`](README.md) describes the
system, and [`llms.txt`](llms.txt) is the machine-readable summary.

## Layout

| Path | Contents | Trust |
|---|---|---|
| `src/knotwork/knotcore/` | Kernel (TCB): contracts, seeds, io, topo, gauge, logic, OCR, witness, CVD, legibility, oracle, engine, grading | trusted |
| `src/knotwork/plugins/` | Scene plugins (`rack_patch`, `tangle`, `gauge`, `panel`) and the registry | untrusted |
| `src/knotwork/render/`, `pipeline/` | Renderer; generate / certify / verify / mutate / twins / counterfactual | — |
| `src/knotwork/arena/` | Model Arena: sessions, runner, reports, regression gate, `campaign.yaml` | — |
| `src/knotwork/posttrain/` | vCoT, RLVR reward, verifier server, curriculum and pack exporter, throughput | — |
| `src/knotwork/studio/` | Studio logic: factory, guide tooltips, Human QA review log, public-demo guard | — |
| `src/knotwork/export/`, `bench/`, `probes/`, `freshness/` | Harbor / HF / RLVR exporters, staircase bench, shortcut probes, beacon seeds and burn ledger | — |
| `apps/` | Streamlit entry `inspector.py` and the five pages in `app_pages/` | — |
| `tests/` | Offline suite, tiered `tier0` / `tier1` (CI) and `tier2` (nightly) | — |
| `scripts/verify_phase.py` | Phase gates | — |

## Commands

```bash
uv sync --locked                                  # environment; do not edit pyproject.toml / uv.lock casually
uv run ruff format . && uv run ruff check .       # format + lint
uv run mypy && uv run lint-imports                # types (knotcore strict) + 3 import contracts
uv run pytest -n auto -m "tier0 or tier1" -q      # what CI runs
uv run pytest -n 0 tests/conformance/test_cli.py  # CLI contract
uv run knot selftest --seeds-only                 # determinism known-answer test
```

Paths are POSIX and the repository is LF-only. On Windows, use WSL or Git Bash.

## Hard rules

1. **Kernel isolation.** `knotcore` never imports `knotwork.plugins` or any other Knotwork package,
   or any network, shapely, UI or ML library, and never uses `importlib` / `__import__`.
   `lint-imports` and `tests/conformance/test_kernel_isolation.py` enforce this.
2. **The kernel trusts pixels, not plugins.** It reads rasters and declarative JSON only, never
   `SceneIR.payload`.
3. **Offline by default.** Tests block every non-loopback socket and DNS lookup. Never call a paid
   API, and never print, log or commit a key. Never run `scripts/gap_check.py`. Live runs need the
   human's `.checkpoints/` approval and `OPENROUTER_API_KEY`. Never create, edit or delete
   `.checkpoints/*`.
4. **Determinism.** All randomness goes through `knotcore.seeds.SeedTree`. No `random`, `hash()` or
   wall-clock time in outputs.
5. **Never weaken a gate.** Fix the code, not the thresholds in `scripts/verify_phase.py` or configs.
6. **No placeholders** (`TODO`, `NotImplementedError`, `pass`-only bodies) in `src/` or `apps/`.

## CLI conformance registry

The `knot` CLI has 19 flat, hyphenated subcommands (`arena-run`, `serve-verifier`, ...). Every new
command must be added to `tests/conformance/test_cli.py` before you commit, or CI fails. Use
`COMMANDS` for phase-dispatched commands and `DIRECT` for commands that import their implementation
directly.

## Writing a plugin

- In-repo: `uv run knot new-plugin my_domain` scaffolds `src/knotwork/plugins/my_domain/` with a
  conformance test and adds no lines to `knotcore`.
- As an external wheel: declare an entry point in the `knotwork.plugins` group
  ([`docs/PLUGIN_SDK.md`](docs/PLUGIN_SDK.md)). `uv run knot plugins` shows whether it was accepted.
- Plugins may import `knotwork.knotcore` (contracts, seeds) and `knotwork.render`. `tangle` and
  `gauge` never import each other; only `panel` composes them.

## Public sandbox mode

`KNOTWORK_PUBLIC_DEMO=1` ([`docs/DEPLOY_CLOUD_RUN.md`](docs/DEPLOY_CLOUD_RUN.md)) confines storage to
`/tmp/knotwork_demo/`, caps runs, never reads a server API key, locks Model Arena to Offline Mock
(`execution_modes`, `offline_only`; enforced again in `arena.planning.start`), hides the
`campaign.yaml` uploader (`campaign_import_allowed`) and skips external plugin entry points. Any new
page control that takes a folder or a count must go through `knotwork.studio.demo_guard`
(`path_problem`, `cap`, `config_problems`), and any new control that could reach the network or load
code must be off in the demo. With the variable unset or `0`, the guard must change nothing.

## Evidence and literature

Committed live evidence lives in `docs/evidence/` (compact JSON / CSV / Markdown only; raw Arena
workspaces, images and video stay in gitignored `artifacts/`). `literature/` holds BibTeX for every
cited work and PDFs only when the source grants redistribution (CC BY / CC BY-SA) and the file is at
most 10 MB; see `literature/INDEX.md`.

## Before you finish

Run format, lint, mypy, lint-imports and the tests you touched. Keep new tests fast, with mock models
and small fixtures. Commit with a Conventional Commit message (`feat(arena): ...`).
