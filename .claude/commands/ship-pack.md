---
description: Generate the 11-slide deck figures, docs/DECK_COPY.md and docs/PORTAL_SUBMISSION.md
---

# /ship-pack

Produce the submission pack from real artifacts (never invent numbers; every figure and number must come
from `artifacts/` or `samples/logs/`).

1. Ensure artifacts exist: Phase 5 gate passed; if `.checkpoints/CP2_APPROVED` exists use
   `artifacts/phase5/bench_live/`, else `artifacts/phase5/bench_mock/` and label results "simulated VLMs".
2. Figures: `uv run python -m knotwork.report.figures --artifacts artifacts --out docs/figures` →
   `fig01_architecture.png` (trusted kernel vs plugins), `fig02_contact_sheet.png`,
   `fig03_pixel_witness.png`, `fig04_fault_matrix.png` (M1–M10 × E_* heatmap), `fig05_twins.png`,
   `fig06_gaslight.png`, `fig07_probes.png` (accuracy vs chance with Wilson bars), `fig08_vendi.png`,
   `fig09_staircase.png`, `fig10_per_crossing_p.png` + composition law, `fig11_harbor_oracle.png`.
   Each ≥ 1600 px wide, dpi 200, readable at slide size.
3. `docs/DECK_COPY.md`: exactly 11 sections `## Slide 1` … `## Slide 11` in the **official Innodata
   template order** from `docs/SUBMISSION_CHECKLIST.md` §C, using its slide titles verbatim:
   1 Cover (Team Name, Team Leader Name, Problem Statement) · 2 Brief about the Idea (fig02) ·
   3 Opportunity / Differentiation / USP (fig07, fig08; quote *"measures true reasoning performance while
   preventing test data leakage"* and answer it with the Freshness Protocol) · 4 List of Features Offered
   (fig04, fig05) · 5 Process Flow / Use-Case Diagram (fig03) · 6 Wireframes / Mock Diagrams (CLI + the
   4 inspector tabs; capture screenshots with `uv run streamlit run apps/inspector.py`, save as
   `docs/figures/wireframe_tab<1-4>.png`) · 7 Architecture Diagram (fig01) · 8 Technologies Used (fig11;
   quote *"pytest or Harbor framework for evaluation"* and answer it with the Harbor exporter + oracle
   pytest) · 9 Estimated Implementation Cost ($0 offline core, ≤ $8 capped live run with the real
   preflight/actual numbers; fig09, fig10, fig06) · 10 Hackathon Deliverables · 11 Thank You.
   Each slide: title, ≤ 5 bullets, the figure file(s) to place, speaker note (≤ 60 words).
4. `docs/PORTAL_SUBMISSION.md` with blocks exactly in this form (the gate parses them):
   ```
   <!-- FIELD:challenge -->
   Visual Task Factory
   <!-- /FIELD -->
   <!-- FIELD:prototype_brief -->
   …≤ 1024 characters…
   <!-- /FIELD -->
   <!-- FIELD:technology_stack -->
   …≤ 1024 characters…
   <!-- /FIELD -->
   <!-- FIELD:github_repo -->
   https://github.com/<owner>/knotwork
   <!-- /FIELD -->
   <!-- FIELD:demo_video -->
   <paste the demo video URL after upload>
   <!-- /FIELD -->
   ```
   Count characters with `uv run python -c` (len of the stripped block) and print the counts. Aim for
   950–1024 characters for the two long fields: dense, concrete, numbers from artifacts.
5. Run `uv run pytest tests/docs -q` and `uv run python scripts/verify_phase.py --phase 6 --keep-going`;
   fix anything in your scope. Commit `docs: ship pack`.
