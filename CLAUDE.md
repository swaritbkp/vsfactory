# CLAUDE.md — Autonomous engineering manual for Knotwork

You are the sole engineer building **Knotwork**, a proof-carrying synthetic visual task factory
(Innodata AI Hackathon 2026, Track 2 *Visual Task Factory*). You work autonomously, phase by phase,
from `TASKS.md`, against the normative `SPEC.md`, gated by `scripts/verify_phase.py`.

Read order at session start: this file → `SPEC.md` (skim §0–2, then the section for your task) →
`TASKS.md` (first unchecked task) → `BLOCKERS.md`.

---

## 1. What "done" means

A task is done when its **Verification Command** exits 0 **and** its **Done Criteria** hold. A phase is
done when `uv run python scripts/verify_phase.py --phase N` prints `PHASE N GATE: PASSED`. Nothing else
counts — not "looks right", not "tests mostly pass".

## 2. Architectural laws (never violate; gates enforce most of them)

1. **knotcore LOC caps** (non-blank, non-comment, non-docstring; `kernel_audit.count_loc_source`):
   **executable ≤ 850** (every knotcore module except the declarative `contracts.py`) and **total ≤ 1350**
   (including `contracts.py`). Follow the per-module targets in SPEC §2.2. Never move logic into
   `contracts.py` to dodge the executable cap. Check with
   `uv run python -c "from knotwork.devtools.kernel_audit import loc_report as r; x=r(); print(x.summary(), x.per_file)"`.
2. **knotcore never imports** `knotwork.plugins` (or any other knotwork package, network, shapely, UI or ML
   libraries), never uses `importlib`/`__import__`. Two layers check this: `lint-imports` and
   `tests/conformance/test_kernel_isolation.py`.
3. **knotcore reads rasters + declarative JSON only** (beauty, id buffer, masks, manifest, claims,
   question). It never reads `SceneIR.payload`, never trusts plugin math (scale formulas, centres,
   polarity) — it re-derives from pixels and cross-checks declarations (OCR labels, mask boxes).
4. **Renderer:** beauty is 2× supersampled then Lanczos-downsampled to 1024; masks (`L` {0,255}) and the
   16-bit id buffer are drawn at 1× with **no anti-aliasing**. Masks are amodal; the id buffer is modal and
   must match the beauty (halo pixels are 0). Pure-Python wheels only (Pillow/NumPy/SciPy/skimage/shapely).
5. **Never call paid APIs.** No `--live` unless `.checkpoints/CP2_APPROVED` exists *and* the human has
   exported `OPENROUTER_API_KEY` — and then only the exact command in TASKS 5.7. Tests and gates are
   offline (the test suite blocks every non-loopback socket and DNS lookup). Never print, log or commit
   API keys. **Never run `scripts/gap_check.py`** (the optional ≤ $0.50 CP1 gap check is human-only).
6. **Determinism:** all randomness via `knotcore.seeds.SeedTree` and the canonical seed paths in SPEC §5.
   No `random`, no `np.random.<legacy>`, no `hash()`, no time, no unordered iteration in outputs.
7. **Contracts are frozen.** Change `contracts.py`/`seeds.py` only to fix a genuine bug; then bump
   `SCHEMA_VERSION` if a field changed, run `uv run python scripts/export_schemas.py`, regenerate the golden
   (`uv run knot selftest --update-golden`) and record a `D-n` entry in SPEC §16.
8. **Never weaken a gate.** Do not edit thresholds in `scripts/verify_phase.py`, `configs/*.yaml` kernel
   section, or SPEC §14 to make something pass. Fix the code. The only allowed gate-script edits are bug
   fixes that keep or tighten semantics, each recorded in SPEC §16.
9. **Never create, edit or delete `.checkpoints/*` approval files** (`CP1_APPROVED`, `CP2_APPROVED`,
   `CP2_SKIPPED`, `GAPCHECK_APPROVED`) — not with Write, Edit, `echo`, `touch`, `cp` or any other means.
   Only the human does.
10. **No placeholders.** No `TODO`/`FIXME`/`XXX`/`HACK`, no `NotImplementedError`, no `pass`-only bodies in
    `src/` or `apps/`. If something cannot be built, it is a blocker (§4), not a stub.
11. **Untrusted plugins stay honest.** Plugins may import `knotwork.knotcore` (contracts, seeds) and
    `knotwork.render`; `tangle` and `gauge` never import each other; only `panel` composes them.

## 3. Autonomous Execution Loop Protocol

Repeat until a `[HUMAN-CHECKPOINT]` or the end of TASKS.md:

1. **Pick** the first `[ ]` task in `TASKS.md` (skip `[x]`, `[~]`; stop on `[!]` only if it blocks the next task).
2. **Read** the SPEC sections it cites and the existing code it touches. State a 3–6 line plan in the chat.
3. **Implement** cleanly: typed (mypy-clean), small functions, docstrings stating invariants, tests first
   when practical (the Done Criteria are your test list).
4. **Verify:** run the task's exact Verification Command.
5. **If failing:** diagnose from the actual output (read the traceback, add a focused test, print the
   offending metric) and fix. **Max 3 attempts** per task. After the third failed attempt write a blocker
   entry (§4), mark the task `[!]`, and continue with the next task that does not depend on it.
6. **Quality bar:** `uv run ruff format . && uv run ruff check . --fix && uv run ruff check . && uv run pytest -q`
   (the full offline suite must stay green; `uv run mypy` must stay clean).
7. **Mark** the task `[x]` in `TASKS.md` (edit only its checkbox).
8. **Commit** with a Conventional Commit message, e.g.
   `git add -A && git commit -m "feat(knotcore): TRACE with geodesic BFS (task 1.3)"`.
   Types: `feat`, `fix`, `test`, `refactor`, `docs`, `chore`, `perf`, `data`. One task per commit.
9. **Advance** automatically to the next task. At a phase's final task run the phase gate; on PASS update
   `.ci_phase` to that phase number and commit.

Long runs: the Phase 3 gate generates 600 items per plugin — run it with `--workers 0` (already in the
gate) and let it finish; it caches in `artifacts/phase3/bench_split/` keyed by a source digest.

## 4. Blockers

Write to `BLOCKERS.md` using its template: task id, exact command, last 10 lines of failure, the three
attempts (hypothesis + change), suspected root cause, one concrete question for the human. Never mark a
blocked task `[x]`. Never "solve" a blocker by weakening laws in §2.

## 5. Two-Checkpoint Halt Policy

You pause for human input at **exactly two** points; everything else runs without asking:

* **[HUMAN-CHECKPOINT-1]** — after the Phase 2 gate passes (task 2.13). Tell the human:
  *"Phase 2 passed. Please open `artifacts/phase2/contact_sheet.png` (rows, Pixel-Witness Proof strips,
  M1–M10 captions) and, if the renders look right, run `echo ok > .checkpoints/CP1_APPROVED`. Optionally
  run the ≤ $0.50 gap check printed in `artifacts/phase2/CHECKPOINT_1_READY.md` first. Otherwise describe
  the problem."* Then stop. If `artifacts/phase2/gap_check.json` exists when you resume, read it and note
  in SPEC §16 whether difficulty needs adjusting (do not change thresholds).
* **[HUMAN-CHECKPOINT-2]** — after the Phase 5 gate passes (task 5.6). Show the preflight estimate from
  `artifacts/phase5/CHECKPOINT_2_READY.md` and ask the human to either approve the `$6–$8` OpenRouter run
  (`echo ok > .checkpoints/CP2_APPROVED` + `export OPENROUTER_API_KEY=…`) or skip it
  (`echo offline > .checkpoints/CP2_SKIPPED`). Then stop.

Do not ask for confirmation on anything else (file layouts, refactors, dependency choices inside the
locked stack, retries). When SPEC leaves a detail open, choose the simplest option consistent with it and
log a `D-n` decision in SPEC §16.

## 6. Commands you will use

```bash
uv sync --locked                                   # environment
uv run python scripts/verify_phase.py --phase N    # phase gate (add --keep-going to see all failures)
uv run python scripts/verify_phase.py --list       # gate plan
uv run pytest tests/<dir> -q                       # focused tests
uv run lint-imports && uv run mypy                 # isolation + types
uv run knot <generate|verify|mutate|probe|diversity|export|bench|selftest|report|new-plugin> --help
```

Slash commands: `/run-phase N`, `/run-all`, `/audit`, `/ship-pack` (see `.claude/commands/`).

## 7. Style

Python 3.11, `from __future__ import annotations`, full type hints, pydantic v2 models from
`knotcore.contracts` at every boundary, `pathlib.Path`, no global mutable state except lazy singletons
(OCR engine). NumPy vectorised over loops in the kernel. mypy is `strict` for knotcore and SciPy/scikit-image are
untyped: wrap their results explicitly (`np.asarray(x, dtype=...)`, `int(...)`, `typing.cast`) so no
`Any` leaks into return values (`warn_return_any`). Errors are values: kernel functions return codes
(`ErrorCode`) rather than raising, except for schema/IO failures. Keep messages factual; no emojis in code.

## 8. Platform notes

* Windows users run everything inside **WSL** (bash available). All paths are POSIX; `.gitattributes`
  forces LF. `os.cpu_count()` may be large — the gates use `--workers 0` (= cpu_count − 1).
* RapidOCR ships its ONNX models inside the wheel: no downloads. Construct it once per process.
* If `uv` warns about `UV_NATIVE_TLS`, it is harmless.
