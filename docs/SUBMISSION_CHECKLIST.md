# Submission checklist — Innodata AI Hackathon 2026 · Track 2 (Visual Task Factory)

This file maps every official requirement to the artifact that satisfies it and the gate that proves it.
Section C follows the **exact 11-slide order of the official Innodata PDF template**; `docs/DECK_COPY.md`
MUST use these slide titles verbatim and in this order (`## Slide 1` … `## Slide 11`).

## A. Required deliverables

| Requirement (official) | Knotwork artifact | Proven by |
|---|---|---|
| Runnable source code | this repo (`uv sync --locked && uv run knot selftest`) | CI (`.github/workflows/ci.yml`), gate `p3.selftest` |
| Sample task packages | `samples/harbor/` (10 Harbor tasks), `samples/hf/` (parquet), RLVR jsonl | gates `p4.export_harbor`, `p4.hf_rlvr` |
| Validation test logs | `samples/logs/` (gate JSONs phase 0–6, mutate/probe/diversity/selftest/verify reports), `VERIFICATION.md` | gate `p6.docs` |
| Technical documentation | `README.md`, `SPEC.md`, `VERIFICATION.md`, `DATA_CARD.md`, `docs/schemas/` | gate `p6.docs` |
| Live prototype demo | `uv run streamlit run apps/inspector.py` (4 tabs) + demo video | test `tests/apps/test_inspector.py` |

## B. Template phrases we answer explicitly

| Template phrase | Where we answer it | Evidence |
|---|---|---|
| "measures true reasoning performance while preventing test data leakage" | **Freshness Protocol**: beacon-seeded generation (drand / NIST / offline), SHA-256 content addressing, sealed answer commitments, API burn ledger, `KNOTWORK-CANARY-…` canary; plus shortcut probes ≤ chance + 5 % | `freshness_receipt.json`, `artifacts/phase3/probe_report.json`, fig07 |
| "pytest or Harbor framework for evaluation" | **Native Harbor exporter** (`knot export --format harbor`) with oracle `solution/solve.sh` and `tests/test.sh` → `reward.txt`/`reward.json`; automated oracle pytest (oracle = 1.0, wrong = 0.0) | `tests/export/test_harbor_oracle.py`, gate `p4.export_harbor`, fig11 |

## C. 11-slide deck mapping (`docs/DECK_COPY.md`) — official template order

| # | Official slide title | Knotwork content | Figure(s) |
|---|---|---|---|
| 1 | **Cover** — Team Name, Team Leader Name, Problem Statement | Team name, team leader name, problem statement: *Track 2 — Visual Task Factory: reusable synthetic visual tasks with provably valid labels* | — |
| 2 | **Brief about the Idea** | Knotwork: a proof-carrying factory — untrusted plugins draw scenes, a small trusted pixel kernel re-derives every answer from pixels and issues a certificate | fig02_contact_sheet |
| 3 | **Opportunity / Differentiation / USP** | Answer the template phrase verbatim: *"measures true reasoning performance while preventing test data leakage"* → Freshness Protocol (beacon-seeded generation, content addressing, sealed answer commitments, burn ledger, canary) + shortcut probes ≤ chance + 5 % + Vendi diversity vs recolor-only | fig07_probes, fig08_vendi |
| 4 | **List of Features Offered** | 3 plugins (tangle, gauge, panel), 11 question families incl. T_GASLIGHT, causal/nuisance minimal-pair twins, M1–M10 fault matrix (100 % caught), file-only `knot verify`, dual exporters | fig04_fault_matrix, fig05_twins |
| 5 | **Process Flow / Use-Case Diagram** | sample → render (2× beauty / 1× non-AA masks + id buffer) → claims → D1/D2/D3 certification → certified bundle or quarantine (E_*) → export / benchmark | fig03_pixel_witness |
| 6 | **Wireframes / Mock Diagrams** | `knot` CLI (10 subcommands) + 4-tab Streamlit Forensic Inspector: Live Generator & Fault Injector · Pixel Witness & Twin Diff · Human Staircase Pilot · Benchmark, Quarantine Sankey & Harbor Browser (screenshots of `apps/inspector.py`) | inspector screenshots |
| 7 | **Architecture Diagram** | Trusted `knotcore` (≤ 850 executable LOC, import-linter + AST isolation) vs untrusted plugins; data flow across the trust boundary | fig01_architecture |
| 8 | **Technologies Used** | Pure-Python wheels (Pillow, NumPy, SciPy, scikit-image, shapely, RapidOCR, pydantic v2, uv); answer the template phrase verbatim: *"pytest or Harbor framework for evaluation"* → native Harbor exporter (`solve.sh` oracle, `tests/test.sh` → `reward.json`) + oracle pytest (oracle = 1.0, wrong = 0.0) + HF parquet / RLVR | fig11_harbor_oracle |
| 9 | **Estimated Implementation Cost** | $0 core (offline, zero API keys, laptop + free CI); optional live 3-VLM benchmark ≤ $8 hard cap (preflight estimate from `CHECKPOINT_2_READY.md`, actual from bench report); what that spend buys: staircase thresholds, per-crossing error p, composition law, gaslighting failure rate | fig09_staircase, fig10_per_crossing_p, fig06_gaslight |
| 10 | **Hackathon Deliverables** | Public GitHub repo, `samples/harbor/` + `samples/hf/` task packages, `samples/logs/` validation logs + `VERIFICATION.md`, technical docs (`README`, `SPEC`, `DATA_CARD`), demo video | — |
| 11 | **Thank You** | Links (repo, video) + one-command offline reproduction: `uv sync --locked && uv run knot selftest` | — |

## D. Hack2skill portal fields (`docs/PORTAL_SUBMISSION.md`)

| Field | Constraint | Source block |
|---|---|---|
| Challenges | select **Visual Task Factory** | `FIELD:challenge` |
| Prototype Brief | ≤ 1,024 characters | `FIELD:prototype_brief` |
| Technology Stack | ≤ 1,024 characters | `FIELD:technology_stack` |
| GitHub Public Repo Link | public URL | `FIELD:github_repo` |
| Demo Video Link | public URL | `FIELD:demo_video` |
| Solution Deck (PDF) | 11 slides, official template | export of the deck built from `docs/DECK_COPY.md` |

## E. Final human steps (cannot be automated)

1. Make the GitHub repo public and push (`git push -u origin main`).
2. Record the demo video (inspector tabs 1→4, `knot mutate`, `knot export --format harbor`, oracle test).
3. Paste the repo and video URLs into `docs/PORTAL_SUBMISSION.md`, then into the portal.
4. Build the 11 slides in the official template from `docs/DECK_COPY.md` + `docs/figures/`, export PDF.
