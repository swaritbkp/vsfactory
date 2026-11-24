---
description: Run phases sequentially from the first incomplete one until the next human checkpoint
---

# /run-all

Run Knotwork phases in order until the next `[HUMAN-CHECKPOINT]` or completion.

1. Read `CLAUDE.md`. Determine the current phase = the lowest phase in `TASKS.md` that still has a `[ ]`
   task (or whose gate has not passed: `artifacts/gates/phase<N>.json` missing or `"ok": false`).
2. Checkpoint awareness:
   * If the current phase is ≥ 3 and `.checkpoints/CP1_APPROVED` is missing → you are waiting at
     **[HUMAN-CHECKPOINT-1]**: remind the human what to review and stop.
   * If the current phase is 5 and task 5.6 is `[x]`, or the current phase is 6, and neither
     `.checkpoints/CP2_APPROVED` nor `.checkpoints/CP2_SKIPPED` exists → you are waiting at
     **[HUMAN-CHECKPOINT-2]**: show the preflight estimate and stop.
3. Otherwise execute `/run-phase <current>` semantics (see `.claude/commands/run-phase.md`) for the current
   phase, then continue with the next phase in the same session. Do not ask the human anything between
   checkpoints.
4. If `.checkpoints/CP2_APPROVED` exists when you reach task 5.7, run its exact Verification Command once
   (the CLI enforces the budget cap); if `.checkpoints/CP2_SKIPPED` exists, mark 5.7 `[~]`.
5. When stopping, print: phases completed this session, gate status per phase, open blockers, and the
   single next action for the human.
