---
description: Autonomously execute every unchecked task of one phase, gate it, and commit
argument-hint: <phase number 0-6>
---

# /run-phase $ARGUMENTS

You are executing **Phase $ARGUMENTS** of Knotwork autonomously. Follow `CLAUDE.md` exactly.

1. Read `CLAUDE.md` §2–§5, then the `## Phase $ARGUMENTS` section of `TASKS.md`, then `BLOCKERS.md`.
2. Pre-flight: `git status` must be clean (commit or stash leftovers with a `chore:` commit first).
   If $ARGUMENTS is 3 or higher, confirm the required checkpoint file exists
   (Phase 3+: `.checkpoints/CP1_APPROVED`; Phase 6: `.checkpoints/CP2_APPROVED` or `.checkpoints/CP2_SKIPPED`).
   If it is missing, STOP and tell the human exactly which file to create.
3. For each `[ ]` task of Phase $ARGUMENTS, in order, run the **Autonomous Execution Loop** (CLAUDE.md §3):
   plan → implement → run the task's exact Verification Command → self-heal (max 3 attempts, then write a
   `BLOCKERS.md` entry and mark `[!]`) → `uv run ruff format . && uv run ruff check . && uv run pytest -q`
   → mark `[x]` → `git add -A && git commit -m "<type>(<scope>): <summary> (task <id>)"`.
4. Run the phase gate: `uv run python scripts/verify_phase.py --phase $ARGUMENTS`.
   * On FAIL: read every failing gate's detail, fix root causes (never thresholds), re-run. Up to 3 gate
     iterations, then write a blocker and stop.
   * On PASS: write `$ARGUMENTS` into `.ci_phase`, commit `chore(phase$ARGUMENTS): gate passed`.
5. If Phase $ARGUMENTS is 2 → announce **[HUMAN-CHECKPOINT-1]** (CLAUDE.md §5) and stop.
   If Phase $ARGUMENTS is 5 → announce **[HUMAN-CHECKPOINT-2]** with the preflight cost and stop.
6. Finish with a compact table: task id · status · commit hash, plus the gate summary lines.
