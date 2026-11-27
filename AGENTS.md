# AGENTS.md — working in the Knotwork repository

A short operating guide for coding agents (Claude Code, Codex, Cursor, …). `CLAUDE.md` is the full
engineering manual and wins on any detail; this file never contradicts it.

## Layout

| Path | What | Trust |
|---|---|---|
| `src/knotwork/knotcore/` | the kernel (TCB): contracts, seeds, topology, OCR, witness, CVD, oracle, grading | trusted |
| `src/knotwork/plugins/` | scene plugins (`rack_patch`, `tangle`, `gauge`, `panel`) and the registry | untrusted |
| `src/knotwork/render/`, `pipeline/` | renderer; generate / certify / verify / mutate / twins | — |
| `src/knotwork/arena/` | Model Arena: sessions, runner, reports, regression gate, `campaign.yaml` | — |
| `src/knotwork/posttrain/` | vCoT, RLVR reward, verifier server, curriculum and pack exporter | — |
| `src/knotwork/studio/` | Studio logic (factory, guide tooltips, Human QA review log) | — |
| `apps/` | Streamlit entry `inspector.py`, pages in `app_pages/` | — |
| `tests/` | offline suite, tiered `tier0` / `tier1` (CI) and `tier2` (nightly) | — |

## Commands

```bash
uv sync --locked                                  # environment (never edit pyproject.toml / uv.lock casually)
uv run ruff format . && uv run ruff check .       # format + lint
uv run mypy && uv run lint-imports                # types (knotcore strict) + import contracts
uv run pytest -n auto -m "tier0 or tier1" -q      # what CI runs
uv run pytest -n 0 tests/conformance/test_cli.py  # CLI contract
uv run knot selftest --seeds-only                 # determinism known-answer test
```

On Windows run these in WSL or Git Bash; paths are POSIX and the repo is LF-only.

## Hard rules

1. **Kernel isolation.** `knotcore` never imports `knotwork.plugins` or any other knotwork package,
   network, shapely, UI or ML libraries, and never uses `importlib` / `__import__`.
   `lint-imports` and `tests/conformance/test_kernel_isolation.py` enforce it.
2. **The kernel trusts pixels, not plugins.** It reads rasters and declarative JSON only.
3. **Offline by default.** Tests block every non-loopback socket and DNS lookup. Never call a paid API,
   never print, log or commit a key, never run `scripts/gap_check.py`. Live runs need the human's
   `.checkpoints/` approval and `OPENROUTER_API_KEY`; never create or edit `.checkpoints/*`.
4. **Determinism.** All randomness goes through `knotcore.seeds.SeedTree`; no `random`, `hash()` or time
   in outputs.
5. **Never weaken a gate.** Fix the code, not the thresholds in `scripts/verify_phase.py` or configs.
6. **No placeholders** (`TODO`, `NotImplementedError`, `pass`-only bodies) in `src/` or `apps/`.

## CLI conformance registry

The `knot` CLI uses flat, hyphenated subcommands (`arena-run`, `serve-verifier`). Every new command
must be added to `tests/conformance/test_cli.py` — `COMMANDS` for phase-dispatched commands, `DIRECT`
for ones that import their implementation directly — before you commit, or CI fails.

## Writing a plugin

* In-repo: `uv run knot new-plugin my_domain` scaffolds `src/knotwork/plugins/my_domain/` with a
  conformance test; it adds zero lines to `knotcore`.
* As an external wheel: declare an entry point in the `knotwork.plugins` group
  ([`docs/PLUGIN_SDK.md`](docs/PLUGIN_SDK.md)); `uv run knot plugins` shows whether it was accepted.
* Plugins may import `knotwork.knotcore` (contracts, seeds) and `knotwork.render`; `tangle` and `gauge`
  never import each other.

## Before you finish

Run format, lint, mypy, lint-imports and the tests you touched; keep new tests fast (mock models,
small fixtures). Commit with a Conventional Commit message (`feat(arena): …`).
