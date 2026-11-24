# Human checkpoint approvals

Only a **human** creates files in this directory. Claude Code must never create, edit or delete them.

| File | Meaning | Unblocks |
|---|---|---|
| `CP1_APPROVED` | Human eyeballed `artifacts/phase2/contact_sheet.png` and accepts the renders | Phase 3 gate `p3.cp1` |
| `CP2_APPROVED` | Human approved the live OpenRouter run (cost shown in `artifacts/phase5/CHECKPOINT_2_READY.md`) | `knot bench --live` and Phase 6 gate `p6.cp2` |
| `GAPCHECK_APPROVED` | Human allows the optional ≤ $0.50, 20-item live VLM gap check at CP1 (`scripts/gap_check.py`, human-run only) | `scripts/gap_check.py` |
| `CP2_SKIPPED` | Human chose an offline-only submission (mock + replay results) | Phase 6 gate `p6.cp2` |

Create one with e.g. `echo "looks good" > .checkpoints/CP1_APPROVED` and commit it.
