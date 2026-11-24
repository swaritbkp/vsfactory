# Opening prompt for Claude Code

Paste everything between the lines into Claude Code (Opus, effort Medium), started from the repository
root, as the first message. If you prefer, `/run-all` does the same thing once `CLAUDE.md` is loaded.

---

You are the autonomous engineer for **Knotwork**, our Innodata AI Hackathon 2026 Track 2 (Visual Task
Factory) entry. This repository is a starter pack: governance docs, a normative spec, a phase-gated task
list, working contracts, seeds, CLI and a gatekeeper script are already here and passing their own tests.

Your job: build, test, debug and complete the **entire working system** by executing `TASKS.md` from the
first unchecked task, phase by phase (Phase 0 → Phase 6), without asking me anything except at the two
human checkpoints.

Do this now:

1. Read `CLAUDE.md` completely and obey it as law (especially §2 architectural laws, §3 loop protocol,
   §5 two-checkpoint halt policy). Then read `SPEC.md` §0–§2 and §14, and `TASKS.md`.
2. Run `uv sync --locked` and `uv run python scripts/verify_phase.py --list` to see the gate plan.
3. Execute the `/run-all` procedure (`.claude/commands/run-all.md`): for each task — plan briefly,
   implement, run its exact Verification Command, self-heal up to 3 attempts (then record a blocker in
   `BLOCKERS.md` and move on), keep `ruff`, `mypy`, `lint-imports` and the whole offline `pytest` suite
   green, tick the box in `TASKS.md`, and make one Conventional Commit per task.
4. At the end of each phase run `uv run python scripts/verify_phase.py --phase N`; only a PASS completes
   the phase. Never weaken a threshold, never stub, never import plugins into `knotcore`, never exceed
   850 executable / 1350 total LOC in `knotcore`, never call a paid API, never run `scripts/gap_check.py`,
   never touch `.checkpoints/`.
5. Stop only at **[HUMAN-CHECKPOINT-1]** (after the Phase 2 gate: I review
   `artifacts/phase2/contact_sheet.png`) and **[HUMAN-CHECKPOINT-2]** (after the Phase 5 gate: I approve or
   skip the live `$6–$8` OpenRouter run). At each stop, tell me exactly which file to create.
6. Keep chat output terse: a short plan per task, the verification result, the commit hash. When you stop,
   print the gate table and the single next action for me.

Start with task 0.1 now.

---

## After a checkpoint

* **Approve CP1:** `echo "renders ok" > .checkpoints/CP1_APPROVED && git add .checkpoints && git commit -m "chore: CP1 approved"`, then in Claude Code: `/run-all`.
* **Approve CP2 (live run):** check `configs/models.yaml` slugs, `export OPENROUTER_API_KEY=sk-or-...`,
  `echo "approved 8 USD" > .checkpoints/CP2_APPROVED && git add .checkpoints && git commit -m "chore: CP2 approved"`, then `/run-all`.
* **Skip CP2:** `echo "offline only" > .checkpoints/CP2_SKIPPED && git add .checkpoints && git commit -m "chore: CP2 skipped"`, then `/run-all`.
* **Resume after any interruption:** `/run-all` (it finds the first unchecked task).
* **Unblock:** read `BLOCKERS.md`, answer the question in the entry (edit the entry's `Status:` to
  `ANSWERED: <your decision>`), then `/run-phase <N>`.
