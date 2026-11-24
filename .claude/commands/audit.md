---
description: Full FAANG-grade gate audit of the repository (read-only unless something is broken)
---

# /audit

Run every quality gate and report a single pass/fail table. Do not change code unless a check fails; if
one fails, fix the root cause (never a threshold) and re-run the audit.

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run lint-imports
uv run python -c "from knotwork.devtools.kernel_audit import loc_report as r; x=r(); print(x.summary(), x.per_file); assert x.ok"
uv run python scripts/export_schemas.py --check
uv run pytest -q
uv run knot selftest --seeds-only
uv run knot mutate --fault all --n 4 --out artifacts/audit/mutate.json     # once Phase 2 is done
uv run knot selftest                                                       # once Phase 3 is done
uv run python scripts/verify_phase.py --phase "$(cat .ci_phase)" --upto
```

Commands whose phase is not reached yet exit with code 3 (`E_NOT_IMPLEMENTED`): report them as "n/a
(phase N)", not as failures. Also grep for placeholder markers in `src/` and `apps/` with
`uv run python -c "from pathlib import Path; from knotwork.devtools.kernel_audit import scan_placeholders as s; print(s(Path('src')) + s(Path('apps')) if Path('apps').exists() else s(Path('src')))"`.

Output: a table `check · result · detail`, the knotcore per-module LOC table, and a one-line verdict.
