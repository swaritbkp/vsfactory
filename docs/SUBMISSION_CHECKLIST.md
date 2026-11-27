# Submission checklist — Innodata AI Hackathon 2026 · Track 2 (Visual Task Factory)

This file maps every official requirement to the artifact that satisfies it and the check that proves it.
Section C follows the **exact 11-slide order of the official Innodata PDF template**; the deck must use
these slide titles verbatim and in this order. The deck copy and portal text are written at submission
time (the `/ship-pack` command drafts them); they are not committed to this repository.

## A. Required deliverables

| Requirement (official) | Knotwork artifact | Proven by |
|---|---|---|
| Runnable source code | this repo: `uv sync --locked && uv run knot selftest --seeds-only`, or the offline [`Dockerfile`](../Dockerfile) | CI (`.github/workflows/ci.yml`: `tier0` + `tier1` on every push), golden selftest in `nightly.yml` |
| Sample task packages | [`samples/twins/`](../samples/twins/) (one certified twin triple per plugin, full `task.json` certificates), [`samples/live_failures/`](../samples/live_failures/), [`samples/fault_proof/`](../samples/fault_proof/) (default `M3 → E_LABEL_MISMATCH` pair); full Harbor / HF parquet / RLVR export via `uv run knot export --format all` | gates `p4.export_harbor`, `p4.hf_rlvr` |
| Validation test logs | [`VERIFICATION.md`](../VERIFICATION.md) (gate results), [`samples/live_run/`](../samples/live_run/) (live bench report + 1,069 per-trial records: 997 main run, 72 true-premise control), CI runs on GitHub Actions | `VERIFICATION.md`, `BLOCKERS.md` |
| Technical documentation | [`README.md`](../README.md), [`SPEC.md`](../SPEC.md), [`VERIFICATION.md`](../VERIFICATION.md), [`DATA_CARD.md`](../DATA_CARD.md), [`docs/schemas/`](schemas/) | — |
| Live prototype demo | 5-page Studio (`uv run streamlit run apps/inspector.py`: Studio, Forensics, Human Eval, Export, Model Arena), `knot` CLI walk-through, `uv run knot report` (self-contained `artifacts/report/report.html`), demo video | `tests/apps/` |

## B. Template phrases we answer explicitly

| Template phrase | Where we answer it | Evidence |
|---|---|---|
| "measures true reasoning performance while preventing test data leakage" | **Freshness Protocol**: beacon-seeded generation (drand / NIST / offline), SHA-256 content addressing, sealed answer commitments, API burn ledger, `KNOTWORK-CANARY-…` canary; plus shortcut probes ≤ chance + 5 % | `freshness_receipt.json`, `artifacts/phase3/probe_report.json` (one residual row: `BLOCKERS.md` 3.6) |
| "pytest or Harbor framework for evaluation" | **Native Harbor exporter** (`knot export --format harbor`) with oracle `solution/solve.sh` and `tests/test.sh` → `reward.txt`/`reward.json`; automated oracle pytest (oracle = 1.0, wrong = 0.0) | `tests/export/test_harbor_oracle.py`, gate `p4.export_harbor` |

## C. 11-slide deck mapping — official template order

| # | Official slide title | Knotwork content | Figure(s) |
|---|---|---|---|
| 1 | **Cover** — Team Name, Team Leader Name, Problem Statement | Team Bilota (Swarit Sharma, Pankaj David); problem statement: *Track 2 — Visual Task Factory: reusable synthetic visual tasks with provably valid labels* | — |
| 2 | **Brief about the Idea** | Knotwork: a proof-carrying factory. Untrusted plugins draw scenes; an import-isolated trusted kernel re-derives every answer from renderer object masks, cross-checked against the visible image via a beauty-only sanity oracle, colourblindness (CVD) gate, and stray-ink witness, and issues a certificate | `docs/assets/studio_rack_patch.png` |
| 3 | **Opportunity / Differentiation / USP** | Answer the template phrase verbatim: *"measures true reasoning performance while preventing test data leakage"* → Freshness Protocol (beacon-seeded generation, content addressing, sealed answer commitments, burn ledger, canary) + shortcut probes ≤ chance + 5 % + Vendi diversity 2.1×–3.7× vs recolour-only | fig06_diversity_vendi |
| 4 | **List of Features Offered** | 4 plugins (rack_patch, tangle, gauge, panel), 11 question families incl. T_GASLIGHT, causal/nuisance minimal-pair twins, M1–M14 fault matrix (56 / 56 caught, 0 / 56 clean rejected), file-only `knot verify`, Harbor / HF / RLVR exporters, 5-page Studio with Model Arena | fig03_twin_consistency, `docs/assets/studio_forensics.png` |
| 5 | **Process Flow / Use-Case Diagram** | sample → render (2× beauty / 1× non-AA masks + id buffer) → claims → D1 / D3 certification → certified bundle or quarantine (E_*) → export / benchmark / Model Arena | — (architecture diagram in `README.md`) |
| 6 | **Wireframes / Mock Diagrams** | Studio pages (Studio, Forensics, Human Eval, Export, Model Arena), `knot` CLI, `knot report` HTML, `samples/` task packages | `docs/assets/*.png`, fig05_failure_examples |
| 7 | **Architecture Diagram** | Trusted `knotcore` (TCB contract: allow-listed imports, no network / process / dynamic code, no `SceneIR.payload`, pinned `kernel_sha256`; import-linter isolation) vs untrusted plugins; data flow across the trust boundary | — (architecture diagram in `README.md`) |
| 8 | **Technologies Used** | Pure-Python wheels (Pillow, NumPy, SciPy, scikit-image, shapely, RapidOCR, pydantic v2, Streamlit, uv), Docker; answer the template phrase verbatim: *"pytest or Harbor framework for evaluation"* → native Harbor exporter (`solve.sh` oracle, `tests/test.sh` → `reward.json`) + oracle pytest (oracle = 1.0, wrong = 0.0) + HF parquet / RLVR | — |
| 9 | **Estimated Implementation Cost** | $0 core (offline, zero API keys, laptop + free CI); live 3-VLM benchmark **$1.12 actual** (943 paid calls, 1,069 trials, $5.00 cap; $0.50 cap on the control): colour-ablation collapse (GPT-5.4 59.4 → 3.1 %), twin consistency, premise control (no model detectably told true from false premises), accuracy by task | fig01_colour_ablation, fig02_false_premise, fig04_accuracy_by_task |
| 10 | **Hackathon Deliverables** | Public GitHub repo, `samples/` (twin triples, live failures, live run outputs, fault proof), `VERIFICATION.md`, technical docs (`README`, `SPEC`, `DATA_CARD`), `docs/figures/`, `docs/assets/`, `Dockerfile`, demo video | — |
| 11 | **Thank You** | Links (repo, video) + offline reproduction: `uv sync --locked && uv run knot selftest --seeds-only` or `docker run --rm knotwork uv run knot selftest --seeds-only` | — |

## D. Hack2skill portal fields

| Field | Constraint | Content |
|---|---|---|
| Challenges | select **Visual Task Factory** | — |
| Prototype Brief | ≤ 1,024 characters | summary of `README.md` (idea, proof metrics, headline finding) |
| Technology Stack | ≤ 1,024 characters | slide 8 content |
| GitHub Public Repo Link | public URL | repository URL |
| Demo Video Link | public URL | video URL |
| Solution Deck (PDF) | 11 slides, official template | built from section C + `docs/figures/` + `docs/assets/` |

## E. Final human steps (cannot be automated)

1. Merge `feat/vtf-v2` into `main`, make the GitHub repo public and push.
2. Record the demo video (Studio tour, `knot selftest`, `knot mutate`, `knot verify`, `knot export --format harbor` + oracle test, `knot report`).
3. Fill the portal fields (section D) with the repo and video URLs.
4. Build the 11 slides in the official template from section C + `docs/figures/` + `docs/assets/`, export PDF.
