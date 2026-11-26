# Knotwork — Engineering Specification (normative)

> **Knotwork** is a *proof-carrying* synthetic visual task factory for the Innodata AI Hackathon 2026,
> Track 2 (*Visual Task Factory*). Every emitted `(image, question, answer, validation evidence)` bundle
> carries a certificate produced by a small trusted kernel that re-derives the answer **from pixels
> only**. Plugins that draw the scenes are untrusted.

This document is the single source of truth. Where it says MUST, tests and `scripts/verify_phase.py`
enforce it. Executable contracts live in `src/knotwork/knotcore/contracts.py`; published JSON Schemas in
`docs/schemas/*.schema.json` are generated from it (`scripts/export_schemas.py`) and a test fails if they
drift. If this document and code disagree, fix the code to match this document, unless the change is a
documented decision recorded in section 16 (Decision log).

---

## 0. Glossary

| Term | Meaning |
|---|---|
| **Kernel / knotcore** | `src/knotwork/knotcore/` — trusted; ≤ 850 executable LOC (≤ 1350 incl. schemas); reads rasters + JSON only. |
| **Plugin** | Untrusted scene generator/renderer implementing `TaskPlugin` (`tangle`, `gauge`, `panel`). |
| **Candidate** | One rendered scene + one question before certification. |
| **Bundle** | A certified candidate on disk (section 3). Identified by `bundle_id = kw-<20 hex>`. |
| **D1 / D2 / D3** | Plugin symbolic answer / plugin shapely-geometric answer / kernel raster answer. |
| **Claim** | A plugin assertion about the scene (e.g. polarity of every crossing) the kernel re-checks. |
| **Quarantine** | Rejected candidate with exact `E_*` reason codes, appended to `quarantine.jsonl`. |
| **Consumption resolution** | 768 px long side — the resolution VLMs see; legibility is checked there. |
| **Twin** | Causal twin (minimal edit, answer flips) or nuisance twin (style change, answer preserved). |

---

## 1. Hard constraints

1. **Offline by default.** `generate | verify | mutate | probe | diversity | export | selftest | report |
   new-plugin` and `bench --offline` (replay + mock) MUST run with no network and no API keys. The
   process-wide policy variable `KNOTWORK_NETWORK` (`deny` default) is set by the CLI; the only module
   allowed to open sockets is `knotwork.net` and it refuses unless the policy allows the target.
2. **Paid calls are triple-locked:** `knot bench --live --budget-usd X` with `0 < X ≤ hard_cap_usd`
   (8.00, `configs/models.yaml`) **and** `.checkpoints/CP2_APPROVED` exists **and** `OPENROUTER_API_KEY`
   is set. Implemented in `knotwork.cli.guard_live` (already working; tests in
   `tests/conformance/test_cli.py`). Nothing calls OpenRouter in tests or gates.
   The only other paid path is the optional CP1 gap check `scripts/gap_check.py` (human-only, ≤ $0.50,
   ≤ 20 items, requires `.checkpoints/GAPCHECK_APPROVED` + `OPENROUTER_API_KEY`; working code with
   mocked-transport tests in `tests/conformance/test_gap_check.py`).
3. **Hardware:** 32 GB RAM, 8 GB VRAM laptop (Windows + WSL) and Ubuntu GitHub Actions. No GPU needed.
   Pure-Python wheels only: Pillow, NumPy, SciPy, scikit-image, shapely, RapidOCR (onnxruntime). No
   skia, cairo, headless browsers, system DLLs.
4. **Python 3.11** (`.python-version`), `requires-python >=3.11,<3.13`, dependency lock in `uv.lock`.
5. **Determinism:** same seed + same `uv.lock` ⇒ identical bundle ids on every machine, regardless of
   `--workers`. Never use Python's `random`, `np.random.<legacy>`, `hash()`, wall-clock time or
   filesystem order in anything that affects output.
6. **No calendar dates** in plans; execution is ordered by Phase 0 → Phase 6 gates only.

---

## 2. Architecture

### 2.1 Package layout and layer rules

```
src/knotwork/
  knotcore/        TRUSTED kernel (bottom layer). Imports: stdlib (no networking), numpy, scipy,
                   skimage, PIL, pydantic, rapidocr_onnxruntime, knotwork.knotcore.*  — nothing else.
    contracts.py   Pydantic v2 contracts (exists)            seeds.py   seed tree + digests (exists)
    io.py          load bundle rasters/JSON                  ocr.py     RapidOCR + template fallback
    topo.py        TRACE, CROSS, ONTOP                       gauge.py   READ, ZONE, GAUGE_OF, READ_ALL
    logic.py       COUNT, CMP, ARGMAX                        witness.py beauty↔mask witness, OOB, ink
    legibility.py  768-px legibility guard                   engine.py  execute, claims, certify
    grading.py     grade_prediction (STDLIB ONLY; copied verbatim into Harbor tests)
  render/          shared deterministic 2D renderer (untrusted helper for plugins)
  plugins/         tangle/, gauge/, panel/, registry.py   (untrusted)
  pipeline/        io, certify, generate, twins, mutate, contact_sheet, verify, selftest
  freshness/       beacon, emission, ledger, receipt
  probes/          heuristics, gbdt, text_only, stats, run
  diversity.py     Vendi score audit
  export/          harbor, hf, rlvr, run
  bench/           staircase, prompts, client_base, client_mock, client_replay, client_openrouter,
                   budget, stats, run
  report/          build, figures
  scaffold/        new_plugin (+ templates/)
  net.py           the only network gateway
  devtools/        static audits shared by tests and gates (exists)
  cli.py           `knot` CLI (exists)
apps/inspector.py  Streamlit forensic inspector (4 tabs)
```

Dependency direction: `cli → {pipeline, export, bench, probes, diversity, report, scaffold} → plugins →
render → knotcore`. `knotcore` imports nothing above it. `plugins.tangle` and `plugins.gauge` are mutually
independent; only `plugins.panel` composes them. Enforced by `.importlinter` (3 contracts) and the AST
scanner in `knotwork.devtools.kernel_audit` (also forbids `importlib.import_module` / `__import__` inside
knotcore).

### 2.2 Kernel LOC budget (two hard caps)

LOC = physical lines carrying a Python token other than COMMENT/NL/NEWLINE/INDENT/DEDENT, excluding
module/class/function docstrings (`kernel_audit.count_loc_source`). Two caps, both enforced by
`kernel_audit.loc_report()`, `tests/conformance/test_kernel_isolation.py` and gate `core.loc`:

* **`EXECUTABLE_LOC_CAP = 850`** — every `knotcore/*.py` **except** the declarative schema module
  `contracts.py` (`DECLARATIVE_MODULES`). This is the code that decides truth.
* **`TOTAL_LOC_CAP = 1350`** — all `knotcore/*.py` including `contracts.py`.

Per-module **targets** (the caps are what is enforced):

| Module | Target | Module | Target |
|---|---|---|---|
| contracts.py (exists, declarative) | ~400 (frozen) | seeds.py (exists) | 115 (frozen) |
| io.py | 50 | ocr.py | 85 |
| topo.py | 100 | gauge.py | 100 |
| logic.py | 40 | witness.py | 65 |
| legibility.py | 80 | engine.py | 100 |
| grading.py | 45 | **executable total** | **≤ 780 of 850** |

Rules when tight: (1) use numpy/scipy/skimage vectorised ops instead of loops; (2) report/format helpers
belong in `pipeline`, never in the kernel; (3) never move a *truth-deciding* computation out of the
kernel to save lines, and never move executable logic into `contracts.py` to dodge the executable cap
(validators there must stay declarative shape checks); (4) `contracts.py`/`seeds.py` change only to fix a
bug (bump `SCHEMA_VERSION` if a field changes and re-run `scripts/export_schemas.py`).

### 2.3 Trust boundary data flow

```
plugin.sample_scene ─► SceneIR (payload opaque to kernel)
plugin.render       ─► beauty.png, id_buffer.png, masks/*.png, manifest.json      ┐ files on disk
plugin.claims       ─► claims.json  (D1 assertions about EVERY crossing/trace/gauge)│
plugin.questions    ─► QuestionProgram                                             ┘
plugin.symbolic_answer  ─► D1        plugin.geometric_answer (shapely) ─► D2 (optional)
knotcore.engine.certify(bundle_dir, question, d1, d2, kernel_cfg) ─► Certificate (D3 + witnesses)
CERTIFIED ⇔ D1 == D3 (and D2 == D1 when D2 present) ∧ all claims verified ∧ no witness/legibility code
```

The kernel MUST NOT read `SceneIR`, plugin modules, or any value it can re-derive from pixels. It may read
manifest metadata that is *declarative* (object ids, kinds, mask file names, label strings, terminal
boxes, gauge→object wiring) and it MUST cross-check every such declaration against pixels (OCR for
labels, masks for boxes, dial mask for centre). `claimed_center`/`claimed_radius` are informational and
never used for truth.

---

## 3. Bundle layout on disk

A dataset directory produced by `knot generate --out OUT`:

```
OUT/
  index.jsonl              one IndexRow per emitted bundle (base + twins), emission order
  quarantine.jsonl         one QuarantineRecord per rejected candidate
  stats.json               GenerateStats
  freshness_receipt.json   FreshnessReceipt
  bundles/<bundle_id>/
    beauty.png             RGB 8-bit, 1024×1024, anti-aliased (2× supersampled, Lanczos down)
    id_buffer.png          16-bit grayscale ("I;16"), 1024×1024, non-AA; 0 = background
    masks/obj_<id:05d>.png 8-bit "L", values {0,255}, non-AA, AMODAL (full object incl. hidden parts)
    manifest.json          RenderManifest
    claims.json            ClaimSet
    task.json              TaskBundle (question, answer, certificate, twins, canary, file digests)
```

**Conventions.** Pixel `(x, y)` covers `[x, x+1) × [y, y+1)`, origin top-left, y down. Angles are degrees
in the math convention measured from +x **counter-clockwise with y up**: `θ = atan2(-(y-cy), x-cx)`,
normalised to `[0, 360)`. *CW progress* from a reference angle `θr` is `φ = (θr − θ) mod 360`.

**id_buffer semantics.** It is the *modal* (visible) object id per pixel and MUST match what is visible
in `beauty.png`: at a crossing halo the under-wire pixels are `0` (background) and the top wire's id is
written over the crossing. Masks are amodal. Load id buffers as `np.asarray(img, dtype=np.uint16)`
regardless of the Pillow mode reported (`I;16` or `I`) and assert `max ≤ 65535`.

**File digests (`TaskBundle.files`).** Keys are relative paths; values are
`seeds.array_digest(decoded array)` for PNGs (RGB uint8 for beauty, uint16 for id buffer, bool for masks)
and `seeds.json_digest(parsed JSON)` for `manifest.json` and `claims.json`. `task.json` is not digested.

**Bundle id.** `bundle_id = seeds.bundle_id(files, question.model_dump(mode="json"),
answer.model_dump(mode="json"))`. `scene_id = seeds.json_digest(scene.model_dump(mode="json"))`.

### 3.1 Auxiliary record formats (pipeline-owned, JSON)

`IndexRow` (index.jsonl):
```json
{"bundle_id":"kw-…","plugin":"tangle","family":"T1_TRACE","split":"dev","role":"base|causal|nuisance",
 "base_id":"kw-…|null","answer_canonical":"7","dir":"bundles/kw-…","scene_id":"<64 hex>"}
```
`GenerateStats` (stats.json):
```json
{"plugin":"tangle","n_requested":30,"n_base_certified":30,"n_bundles":90,"n_candidates":97,
 "n_quarantined":7,"false_quarantine_rate":0.072,"families_expected":["T1_TRACE","T2_CROSS","T3_ONTOP"],
 "by_family":{"T1_TRACE":10},"by_code":{"E_TANGENT_CROSSING":3},"seconds":41.2,"workers":7}
```
`false_quarantine_rate = n_quarantined / n_candidates` over *clean* (non-mutated) candidates.

---

## 4. Contracts (summary; `contracts.py` is normative)

| Model | Purpose | Key invariants |
|---|---|---|
| `RenderManifest` | declarative render metadata | unique ids/names; all refs resolve; boxes may be OOB (kernel finds E_OOB) |
| `ClaimSet` / `Claim` | D1 assertions | one claim per crossing (ONTOP), per start terminal (TRACE), per strand (CROSS, `"*"`), per gauge (READ, ZONE) |
| `QuestionProgram` | executable question | `$sN` refs resolve backwards; `T_GASLIGHT` ⇔ premise present ∧ `holds=False`; `P5_PREMISE_CONTROL` ⇒ true premise |
| `Answer` | typed answer | strict scalars; `PremiseValue` for premise families; `tolerance ≥ 0` |
| `StepTrace` | per-op process trace | `evidence` holds numeric witnesses (path length, centres, angles, OCR text) |
| `Certificate` | kernel verdict | `CERTIFIED ⇔ agree ∧ no codes ∧ claims_failed == 0` |
| `QuarantineRecord` | rejection | `primary_code ∈ codes`; `primary_code = contracts.primary_code(codes)` |
| `FreshnessReceipt` | provenance | hex64 fields; `canary` matches `^KNOTWORK-CANARY-[0-9a-f]{32}$` |
| `TaskBundle` | emitted bundle | certified only; `certificate.bundle_id == bundle_id` |
| `HarborTask` | Harbor export record | `task_name = knotwork/<bundle_id>` |
| `TaskPlugin` | plugin protocol | 8 methods (see 8.0) |

`ErrorCode` members are **declared in priority order**; `CODE_PRIORITY = tuple(ErrorCode)`.
`CLAIM_MISMATCH_CODE`: ONTOP→`E_ZORDER`, READ→`E_SCALE_MISMATCH`, every other op→`E_CLAIM_MISMATCH`.

---

## 5. Seeds and content addressing (`knotcore/seeds.py`, exists)

* Root seed: `root = first128bits(SHA256(randomness_bytes || protocol_commit_utf8))`
  (`derive_root_seed`). Offline randomness: `SHA256("knotwork-offline-beacon-v1:<seed>")`.
* `protocol_commit = SHA256( for f in freshness.protocol_files (sorted): f_path_utf8 ‖ 0x00 ‖ bytes(f with CRLF→LF) ‖ 0x00 )`.
* Children: `SeedTree(root).child(*parts)`; spawn key component = first 8 bytes of
  `SHA256("i:<int>" | "s:<str>")`. `rng()` = `Generator(PCG64(SeedSequence(root, spawn_key)))`.
* **Canonical seed paths** (MUST be used verbatim so twins/faults/workers stay reproducible):

| Purpose | Path |
|---|---|
| family schedule | `(plugin, "schedule")` |
| base slot attempt | `(plugin, "slot", i, "attempt", a)` then `.child("scene")`, `.child("question")` |
| twins | `<attempt>.child("twin_causal", t)`, `<attempt>.child("twin_nuisance", t)` (t = twin retry) |
| mutate | `("mutate", fault, i, "attempt", a)` |
| probes / diversity / bench | `("probe", name, fold)`, `("diversity",)`, `("bench", slot, track)` |

* Slot `i` depends only on `(root, plugin, i, a)` ⇒ output is independent of `--workers`; results are
  sorted by slot index before writing.
* Known-answer vector `selftest_vector()` is pinned in `SELFTEST_EXPECTED`; `knot selftest --seeds-only`
  compares (works today).

---

## 6. Deterministic renderer (`src/knotwork/render/canvas.py`)

`LayeredCanvas(size=1024, supersample=2, background_rgb, font_path)` owns three layers:

| Layer | Resolution | Mode | Anti-aliasing |
|---|---|---|---|
| beauty | 2048 → Lanczos → 1024 | RGB | yes (2× supersample) |
| masks (per object) | 1024 | `L` {0,255} | **no** |
| id buffer | 1024 | int32 work array → saved uint16 | **no** |

API (all coordinates in 1024-px space; the canvas scales beauty drawing by `supersample`):

```python
c = LayeredCanvas(1024, 2, (246,244,238), "assets/fonts/DejaVuSans.ttf")
oid = c.new_object(name, kind, color_rgb, parent=None, label_text=None, box=None, font_px=None) -> int
c.polyline(oid, pts: np.ndarray (N,2) float, width_px: float, beauty=True, mask=True, idbuf=True,
           beauty_rgb=None, shade="flat"|"harness")
c.polygon(oid, pts, beauty=True, mask=True, idbuf=True, beauty_rgb=None)
c.disk(oid, center, radius, ...)            c.arc_band(oid, center, r0, r1, start_deg, end_deg, ...)
c.text(oid, xy, text, font_px, anchor="mm") # beauty AA glyphs; mask/id via fontmode "1" (non-AA)
c.halo_patch(top_oid, pts, width_px, halo_px, radius_px, center)  # see 6.1
c.finish(out_dir, plugin) -> RenderManifest  # writes beauty.png, id_buffer.png, masks/, manifest.json
```

Rules: mask/id drawing uses `ImageDraw` with integer-rounded coordinates and **no** AA; polylines are
drawn as a polygon strip plus round joins (disks at vertices) so widths are exact; text masks use
`ImageDraw.text(..., fontmode="1")`. The font file's SHA-256 MUST equal `render.font_sha256` in the
profile, else raise `E_FONT`. Object ids are assigned sequentially from 1 in draw order.

### 6.1 Per-crossing halo layering (tangle/panel)

1. Draw every wire fully (beauty + amodal mask + id buffer) in a fixed base order.
2. For each crossing `c` with top wire `t`: take the polyline sub-segment of `t` within `halo_radius_px`
   of the crossing centre; on **beauty** draw it with background colour at `width + 2·halo_px`, then redraw
   it in `t`'s style at `width`; on the **id buffer** write `0` over the halo stroke footprint, then write
   `t`'s id over its stroke footprint. Masks are untouched (amodal).
3. Order of crossings processed: increasing x of centre (stable), so overlapping halos are deterministic.

Harness style shading MUST stay inside the stroke footprint (darker 1 px rim, lighter core line); no drop
shadows outside it (they would be unexplained ink, M7 semantics).

---

## 7. Kernel algorithms (normative)

Inputs to every primitive are a loaded bundle `B` (`io.load_bundle(dir)`: manifest, claims, beauty uint8
(H,W,3), luminance L = Rec.709 of sRGB/255 linear-approx `0.2126R+0.7152G+0.0722B` on [0,1], id uint16,
masks dict[int → bool array]) and the kernel config dict `cfg = profile["kernel"]`. `stroke(o)` =
`2 × median(distance_transform_edt(mask_o)[skeletonize(mask_o)])`.

### 7.1 Terminal → strand resolution
`strand_of(terminal)`: the set of `kind=strand` objects whose mask intersects the terminal box. Exactly
one ⇒ that strand; zero ⇒ `E_TRACE_NO_TERMINAL`; more ⇒ `E_TRACE_AMBIGUOUS`.
`label_of(terminal)`: the 768-px OCR result (7.9) of its `label_obj` box, served from the OCR cache
(the legibility guard has already read every label); MUST equal the manifest `label_text`.

### 7.2 TRACE(start_terminal_id) → end terminal label
1. `s = strand_of(start)`; `R` = 8-connected component of `mask[s]` containing `mask[s] ∩ box(start)`
   (`scipy.ndimage.label`, structure = 3×3 ones). If `mask[s] ∩ box(start)` spans >1 component ⇒
   `E_TRACE_AMBIGUOUS`.
2. `E = {t ≠ start : t.role ∈ {end, port}, R ∩ box(t) ≠ ∅}`. `|E| = 1` ⇒ return `label_of(t)` (and
   `t.terminal_id` in evidence); `0` ⇒ `E_TRACE_NO_TERMINAL`; `>1` ⇒ `E_TRACE_AMBIGUOUS`.
3. Evidence: `path_px = |R|`, `end_terminal`, `ocr_text`.

### 7.3 CROSS(a_start, b_start | "*") → count (+ ordered centres)
1. `A = mask[strand_of(a)]`; `others = [strand_of(b)]` or every other strand for `"*"`.
2. For each other `O`: components of `A ∧ mask[O]` (8-conn). Each component = one crossing; centre =
   component centroid (x, y floats).
3. Geodesic order: `d = MCP_Geometric(cost = where(A, 1, inf)).find_costs(starts = pixels of A ∩
   box(a))`; a crossing's key = min `d` over its component pixels. Sort ascending; ties by (x, y).
4. Return integer count; evidence: list of `{"other": obj_id, "center": [x, y], "geo": d}` in order.

### 7.4 ONTOP(a_start, b_start, idx) → start label of the top strand
1. Crossings of `(a, b)` from 7.3; `idx` is 0-based in geodesic order along `a`. Out of range ⇒
   `E_CLAIM_MISMATCH`.
2. Visible fraction: within the crossing component, fraction of pixels whose `id ∈ {id_a, id_b}` MUST be
   `≥ min_visible_crossing_frac` (0.6) else `E_OCCLUDED_CROSSING`.
3. `top = id[round(cy), round(cx)]`; must be `id_a` or `id_b` else `E_OCCLUDED_CROSSING`.
   Return `label_of(start terminal of top)`.

### 7.5 READ(gauge_id) → (value, tolerance)
1. `c` = centroid of `mask[dial_obj]`; `R = sqrt(|mask[dial_obj]| / π)`.
2. Needle PCA: pixels `P` of `mask[needle_obj]`; mean `m`; principal unit vector `u` (largest-eigenvalue
   eigenvector of the 2×2 covariance). Orient `u` so the far end (max projection) is farther from `c`.
   Axis offset `δ = |(c − m) × u|` (2-D cross product magnitude). If `δ > max(max_center_offset_px,
   0.03 R)` ⇒ `E_NEEDLE_OFFCENTER`. Needle angle `θn = atan2(−u_y, u_x)`; length `Ln = max proj − min proj`.
3. Ticks: for each `TickEntry`, centroid angle `θk` of `mask[tick_obj]` about `c`, radius `rk`; value
   `vk = int(OCR768(label_obj))` — the cached 768-px OCR text (7.9) MUST equal manifest `label_text`
   (else `E_LABEL_MISMATCH`). Tick geometry still comes from the 1024-px masks.
4. Reference = tick with the smallest `vk`; `φ = (θ_ref − θ) mod 360` for needle and ticks. Ticks sorted by
   `φ` MUST have strictly increasing `vk` (else `E_SCALE_MISMATCH`). The needle `φn` MUST lie in
   `[φ_first, φ_last]` (else `E_SCALE_MISMATCH`). Bracketing ticks `lo, hi` with `φlo ≤ φn ≤ φhi`.
5. `v = v_lo + (φn − φlo) / (φhi − φlo) × (v_hi − v_lo)` — **never** uses plugin scale math.
6. Tolerance (resolution-derived): `σn = (180/π)/Ln`, `σk = (180/π)/rk`;
   `tol = k × |v_hi − v_lo| / |φhi − φlo| × (σn + σlo + σhi)` with `k = tolerance_guard_k` (2.0).
7. Evidence: `center, R, axis_offset, theta_needle, ticks[{v, theta, r, ocr}], lo, hi, tol`.

### 7.6 ZONE(gauge_id) → zone name
For each `ZoneEntry`, the CW-progress interval `[min φ, max φ]` of `mask[zone_obj]` pixels (reference as
7.5). The zone containing `φn` with ≥ 1.0° margin to both ends wins; none or margin < 1° ⇒
`E_ZONE_UNRESOLVED`. Zone names come from the manifest (declarative) and colours are checked by witness.

### 7.7 GAUGE_OF(port_terminal_id) → gauge_id; READ_ALL(socket ids...) → [(label, v, tol)]
`GAUGE_OF`: the unique gauge whose `binary_dilation(mask[dial_obj], 8 px)` intersects the port box (else
`E_TRACE_NO_TERMINAL`/`E_TRACE_AMBIGUOUS`). `READ_ALL`: for each socket `s` (start terminal):
`READ(GAUGE_OF(end terminal of TRACE(s)))` paired with `label_of(s)`.

### 7.8 COUNT, CMP, ARGMAX
* `COUNT(kind, predicate)`: objects of `kind` whose visible pixel count (`id == obj`) ≥ `count.min_visible_px`
  and that satisfy `predicate ∈ {"any", "zone:<green|amber|red>"}` (zone predicate evaluates `ZONE` of the
  object's parent gauge). Returns int.
* `CMP(x, y)`: `"equal"` if `|x−y| ≤ tol_x + tol_y`, else `"greater"`/`"less"`.
* `ARGMAX(pairs)`: key of the max value; if `v1 − v2 ≤ tol1 + tol2` ⇒ `E_AMBIGUOUS_ARGMAX`.

### 7.9 OCR (`ocr.py`) — 768 px only, cached
OCR runs **only on the 768-px consumption image** (`beauty768`, 7.11). There is no 1024-px OCR pass:
every consumer (legibility 7.11, `label_of` 7.1, tick values 7.5) reads the same cached result.
1. Crop the object `box` scaled by 0.75 (+5 px margin, clipped) from `beauty768`, convert to `L`,
   upscale ×`upscale` (3) Lanczos, pad `pad_px` (40) with the crop's median border value, convert to RGB.
2. **Crop-hash LRU cache:** key = `SHA-256(dtype ‖ shape ‖ bytes)` of the preprocessed uint8 array from
   step 1 (i.e. `seeds.array_digest`); value = `(text, confidence, engine)`. Module-level
   `functools.lru_cache`-style dict with capacity 4096 entries, insertion-ordered eviction; deterministic
   (same crop ⇒ same key ⇒ same result) and per-process (each worker has its own). Identical glyph crops
   (e.g. tick labels repeated across dials, twins, mutate controls) are recognised once.
3. On a miss: `RapidOCR()(img, use_det=True, use_cls=False, use_rec=True)`; take the highest-confidence text.
4. Normalise: NFKC, remove whitespace, upper-case, `"×"→"X"`; if the manifest label is all digits also
   map `O,D→0`, `I,L→1`, `S→5`, `B→8`.
5. If empty or confidence < `min_confidence` (0.5): **template fallback** — render each candidate label of
   the same group (all start labels, all end labels, all tick labels of that gauge) with the bundled font at
   `0.75 × font_px`, compute normalised cross-correlation against the crop, take the best; accept only if
   best − second ≥ 0.05. Record `ocr_engine="template"` in the certificate if used anywhere.
6. Result ≠ manifest label ⇒ `E_LABEL_MISMATCH`; no result ⇒ `E_OCR_FAIL`.
Construct the RapidOCR engine once per process (module-level lazy singleton).

### 7.10 Beauty-to-mask witness (`witness.py`) — closes the circularity loophole
For every object of kind `strand` or `needle`:
* `V = (id == obj)`; interior `I = binary_erosion(V, 1)`; ring `Q = dilate(V, ring_px) ∧ ¬dilate(V, 1)`.
  `contrast = |mean L[I] − mean L[Q]|`. `contrast < min_contrast` (0.12) ⇒ `E_BEAUTY_MISMATCH`.
* Crossing centres (from claims ONTOP args via 7.3) and needle tips (far end from 7.5): mean L of a 5×5
  patch MUST be within 0.10 of the top object's interior mean L, else `E_BEAUTY_MISMATCH`.
* **Unexplained ink:** `ink = max_c |beauty_c − background_c| > ink_delta` (40); `explained =
  dilate(id > 0, ink_dilate_px)`; any 8-connected component of `ink ∧ ¬explained` with area >
  `max_unexplained_area_px` (30) ⇒ `E_UNEXPLAINED_INK`.
* **OOB:** any mask pixel within the outer 2-px canvas border, or any manifest box not fully inside
  `[0, 1024)²` ⇒ `E_OOB`.
Each check appends a `WitnessCheck(name, obj_id, passed, metric, threshold, code)`.

### 7.11 Consumption-resolution legibility guard (`legibility.py`)
`beauty768 = beauty.resize((768, 768), LANCZOS)`; scale `s = 0.75`.
* Stroke: `stroke(o) × s ≥ min_stroke_px_at_consumption` (2.0) for every strand/needle, else
  `E_ILLEGIBLE_RES`.
* Crossing angle: for each crossing, PCA direction of each mask inside a disk of radius `3·stroke` about
  the centre; acute angle between directions `≥ min_crossing_angle_deg` (25°), else `E_TANGENT_CROSSING`.
* Spacing: consecutive crossings along a strand (geodesic key difference) `≥ min_crossing_spacing_strokes
  × stroke` (4), else `E_CROSSING_SPACING`.
* Terminal clearance: distance from each terminal box to the nearest pixel of any mask other than its own
  strand and its own label `≥ min_terminal_clearance_strokes × stroke` (1), else `E_TERMINAL_CLEARANCE`.
* OCR every label object on `beauty768` (7.9, the single OCR pass that fills the cache); mismatch ⇒
  `E_LABEL_MISMATCH`, none ⇒ `E_OCR_FAIL`.

### 7.12 Engine (`engine.py`)
```python
def kernel_sha256() -> str                     # same algorithm as devtools.kernel_audit.kernel_sha256
def execute(b: Bundle, q: QuestionProgram, cfg: dict) -> tuple[Answer, list[StepTrace]]
def check_claims(b: Bundle, claims: ClaimSet, cfg: dict) -> tuple[int, list[ErrorCode]]  # (n_failed, codes)
def certify(bundle_dir: Path, q: QuestionProgram, d1: Answer, d2: Answer | None, cfg: dict,
            bundle_id: str) -> Certificate
```
`certify` order: load (schema errors ⇒ `E_SCHEMA`) → witness (7.10) → legibility (7.11) → claims → execute
→ compare. Every stage runs even after a failure (collect all codes) except when loading fails.
Agreement: label/int/zone/premise-bool by exact canonical equality; floats by `|d1 − d3| ≤ d3.tolerance`.
`D1 ≠ D3` ⇒ `E_D1_D3_MISMATCH` (plus the op-specific `CLAIM_MISMATCH_CODE` when a claim also failed).
`D2` present and `≠ D1` ⇒ `E_D1_D2_MISMATCH`.
Program execution: steps run in order; string args `"$sN"` are replaced by the output of step `sN`; for
every non-premise program the last step's output is the answer. The Answer `canonical` formats are in 13.3.

**Premise programs** (`q.premise is not None`; families `P5_PREMISE_CONTROL`, `T_GASLIGHT`) are exactly
`s0 TRACE(S)`, `s1 GAUGE_OF($s0)`, `s2 READ($s1)`, `s3 ZONE($s1)` (`contracts.PREMISE_OPS`, enforced by the
`QuestionProgram` validator). After running all four steps, `execute()` MUST build the answer itself:
```python
reading, tol = out["s2"]  # READ -> (value, tolerance)
zone = out["s3"]  # ZONE -> ZoneName
holds = zone == q.premise.claimed_zone
Answer(
    type=AnswerType.PREMISE,
    value=PremiseValue(premise_holds=holds, value=reading),
    tolerance=tol,
    canonical=f"{'yes' if holds else 'no'}; {format(reading, 'g')}",
)
```
The kernel never reads `q.premise.holds` (that is the plugin's D1 claim); agreement compares
`premise_holds` exactly and `value` within tolerance.

### 7.13 Quarantine
The pipeline (not the kernel) converts a non-certified Certificate into a `QuarantineRecord` with
`codes = certificate.codes`, `primary_code = contracts.primary_code(codes)`, `stage` = the earliest stage
that produced a code (`witness` < `legibility` < `kernel` (claims/exec) < `certify` (D1/D2/D3)), and
appends it to `quarantine.jsonl`.

---

## 8. Plugins

### 8.0 `TaskPlugin` protocol (contracts.py)
`sample_scene(seeds, profile) → SceneIR`, `render(scene, out_dir, fault=None) → RenderManifest`,
`claims(scene) → ClaimSet`, `questions(scene, seeds, family) → QuestionProgram`,
`symbolic_answer(scene, program) → Answer` (D1), `geometric_answer(scene, program) → Answer | None`
(D2, shapely), `causal_twin(scene, program, seeds) → (SceneIR, edit_description)`,
`nuisance_twin(scene, program, seeds) → SceneIR`. Each plugin package also exposes module-level
`fault_effective(scene: SceneIR, fault: FaultCode) -> bool` and `PLUGIN` (an instance) used by
`plugins/registry.py: get_plugin(name) -> TaskPlugin`.

### 8.1 `plugins/tangle` — braid tangles

**Braid words.** Strands `n ∈ [3, 10]`, positions `0..n−1` (top→bottom). A word is a list of letters
`(i, ε)` with `i ∈ [0, n−2]`, `ε ∈ {+1, −1}`, plus identity slots `ID` (each slot is 2 columns wide).
*Composition:* `P = [0..n−1]` (position → strand). For each letter `(i, ε)`: crossing between
`a = P[i]` and `b = P[i+1]`; **over = a if ε = +1 else b**; then swap `P[i], P[i+1]`. Strand `s` ends at
the position `j` with `P[j] = s`.
*Sampling (exact):* (1) `π ~ uniform S_n` (`rng.permutation`); (2) reduced word = adjacent swaps of a
bubble sort realising `π`, random signs; (3) insert same-sign pairs `(i,ε)(i,ε)` (pure-braid twists,
permutation preserved) at uniform positions until length ∈ `braid_length`; (4) insert
`r2 ~ U[r2_distractor_pairs]` **Reidemeister-II distractors** `(i,ε)(i,−ε)` (topology-preserving, adds 2
crossings); (5) insert `U[identity_slots]` `ID` slots at uniform positions.
End terminal labels are `"1".."n"` top→bottom; start labels are `n` distinct letters from
`profile.tangle.start_labels`, assigned top→bottom in sampled order. Because `π` is uniform, straight-
across succeeds with probability `1/n` exactly.

**Layout.** *strip:* column centre `x_k = left + (k + 0.5)·w`, `w = usable_width / n_columns`; strand at
position `j` has `y_j = top + j·spacing`. A strand involved in a crossing at column `k` moves between
`y_i` and `y_{i+1}` with a cosine S-curve over the column; otherwise it is horizontal. Smooth jitter: add
`Σ_{m=1..3} A_m sin(2π f_m x/W + ϕ_m)` with total amplitude ≤ `jitter_amp_px`, identical per strand.
*horseshoe:* map strip `(x, y)` to an annulus: `θ = 200° − 220°·x/W`, `r = R_out − (y − top)`, centre
`(512, 560)`. Verify with shapely: each strand `LineString.is_simple`; for each strand pair the number of
geometric intersections equals the number of braid crossings between them; minimum distance between
strands outside crossing neighbourhoods ≥ 2·stroke; nominal crossing angle ≥ 35°. Rejection ⇒ resample
(never emit a fold).

**Styles.** `schematic` (flat stroke) and `harness` (rim + highlight core inside stroke).
**Colour regimes.** `unique` (distinct palette colours), `partial` (palette of `ceil(n/3)` colours),
`mono` (all `#2F2F2F`). Palette (≥ 0.2 luminance contrast vs background): `#0072B2 #D55E00 #009E73
#CC79A7 #882255 #117733 #332288 #000000 #8C510A #01665E`. Split `test` uses
`benchmark_color_regimes` (`mono`) so colour carries no answer information for probes/bench.

**Families and exact prompts** (all end with the answer-format line of 13.3):
* `T1_TRACE` — program `[s0 TRACE(start)]`; answer label `"1".."n"`; options = end labels.
  *"The image shows cables running from lettered terminals to numbered terminals. Follow the cable that
  starts at terminal {S}. At which numbered terminal does it end?"*
* `T2_CROSS` — `[s0 CROSS(start, "*")]`; answer int; target count sampled uniformly from
  `t2_target_count`, query strand chosen among strands with that count (resample scene if none).
  *"Follow the cable that starts at terminal {S} along its whole length. How many times does it cross
  another cable?"*
* `T3_ONTOP` — `[s0 ONTOP(a, b, idx)]`; answer label ∈ {A, B} (start labels); pair sampled among pairs
  with ≥ 1 crossing, `idx` uniform; the order of {A, B} in the text is randomised.
  *"Cables from terminals {A} and {B} cross {m} time(s). At crossing number {idx+1} of these, counting
  along cable {A} from its start, which cable is on top?"*
* Claims: TRACE for every start terminal, CROSS(s, "*") for every strand, ONTOP for every crossing.

**Twins.** Causal — T1: swap the label glyphs of the answer end terminal and one other end terminal
(manifest labels updated consistently); T2: fill an `ID` slot adjacent to the query strand with an R-II pair
on it (count +2); T3: flip `ε` of the queried crossing. Nuisance — new jitter phases + style toggle +
optional R-II pair in an `ID` slot on strands not in the query.
**Faults:** M1–M7, M10 (section 10).

### 8.2 `plugins/gauge` — analog dials
`angle(v) = start_deg − (v − vmin)/(vmax − vmin) × sweep_deg` (start 225°, sweep 270°, CW). Objects:
dial face disk (`dial`), 5 major `tick`s with horizontal 2-digit padded labels (`linear`: 00 20 40 60 80,
`offset`: 10 30 50 70 90) at radius 0.62 R with ≥ 0.5·font_px clearance from tick strokes, minor ticks
(`tick_minor`, 4 per major gap), zone arcs (`zone_arc`, green/amber/red bands at 0.92–1.0 R; boundaries on
a 5-unit grid, `z1 ∈ [0.45, 0.65]·span`, `z2 ∈ [0.70, 0.85]·span`), tapered needle polygon (0.15 R tail to
0.85 R tip), hub disk (`decor`). Needle value: integer on `value_grid`, at least `3×grade tolerance` from
any zone boundary. Families: `G1_READ` (*"What value does the gauge needle point to?"*, float answer),
`G2_ZONE` (*"Is the gauge needle in the green, amber, or red zone?"*, zone answer). Claims: READ and ZONE
per gauge. Causal twin: move needle by `≥ 5×tolerance` (G1) or into another zone (G2). Nuisance: bezel
colour, background tint, zone band width. Faults: M8, M9.

### 8.3 `plugins/panel` — composition (adds ZERO lines to knotcore)
Canvas: lettered sockets (start terminals) on the left edge; 2–4 gauges on the right half, each with an
OCR-checked label object `G1`…`Gk` under its dial; each socket
cable is a tangle strand ending at a gauge `port` terminal under the dial; `extra_strands` distractor
cables end at unlabelled blank pads. Namespaces: `panel/…` object names; tangle and gauge renderers are
reused through their public render helpers onto one `LayeredCanvas`.
Programs (D1 from symbolic composition; kernel executes ops only):

| Family | Steps | Answer |
|---|---|---|
| `P1_READ_TRACE` | `s0 TRACE(S)` `s1 GAUGE_OF($s0)` `s2 READ($s1)` | float |
| `P2_ZONE_TRACE` | `… s2 ZONE($s1)` | zone |
| `P3_ARGMAX_PORT` | `s0 READ_ALL(S1..Sk)` `s1 ARGMAX($s0)` | socket label (top-2 margin ≥ 3×tol) |
| `P4_COUNT_ZONE` | `s0 COUNT("needle", "zone:red")` | int |
| `P5_PREMISE_CONTROL` | `s0 TRACE(S)` `s1 GAUGE_OF($s0)` `s2 READ($s1)` `s3 ZONE($s1)`, TRUE premise | premise (built by `execute`, 7.12) |
| `T_GASLIGHT` | `s0 TRACE(S)` `s1 GAUGE_OF($s0)` `s2 READ($s1)` `s3 ZONE($s1)`, FALSE premise | premise (built by `execute`, 7.12) |

Premise prompt: *"Since the gauge connected to socket {S} is in the {claimed} zone, what value does it
read? First say whether that premise is true (yes/no), then give the reading."* Answer
`PremiseValue(premise_holds, value)`, canonical `"no; 47"`. **Exact counts:** per run with `n` base items,
`T_GASLIGHT = P5_PREMISE_CONTROL = round_half_up(0.15·n)`; the remainder is apportioned to P1–P4 by
largest remainder of the profile weights; the slot→family schedule is a permutation drawn from
`(plugin, "schedule")`. True-premise controls make the text uninformative about premise truth.
Twins: causal moves the target needle (or a competitor's for P3) to flip the answer; for premise families
moving the needle into/out of the claimed zone flips `premise_holds` (the twin's family becomes
`P5_PREMISE_CONTROL`/`T_GASLIGHT` accordingly). Nuisance: jitter/style of cables, bezel colour.

### 8.4 Generation pipeline (`pipeline/generate.py`)
`run_generate(plugin, n, out, seed, beacon, profile, split, twins, families, workers) -> int`:
1. Receipt (11.1); `root`; canary. 2. Family schedule (exact counts). 3. For each slot `i`, attempts
`a = 0..max_attempts_per_item−1`: sample scene, question, render to a temp dir, compute D1/D2, write
claims + task draft, `engine.certify`. Quarantine on failure, next attempt. 4. If `twins`: build causal and
nuisance twins (up to 3 retries each); base is emitted only if both twins certify and satisfy
`pixel_diff_frac < causal_max_pixel_frac` (causal; answer differs) and `≥ nuisance_min_pixel_frac`
(nuisance; answer equal). `pixel_diff_frac = mean(any_c |beauty_a − beauty_b| > 8)`. 5. If no attempt
succeeds the run fails with exit 1 and names the slot. 6. Move certified bundles to `bundles/<id>/`, write
index (slot order: base, causal, nuisance), quarantine, stats, receipt. `workers=0 ⇒ cpu_count−1`; use
`concurrent.futures.ProcessPoolExecutor` with per-slot seeds only.

---

## 9. CLI contract (`knot`, implemented in `cli.py`)

Global: exit codes `0` ok · `1` verification/gate failure · `2` usage · `3` not yet implemented · `4`
safety refusal. Every command sets `KNOTWORK_NETWORK` (deny unless stated). Implementation functions
receive exactly these keyword arguments and return an `int` exit code.

| Command | Options (defaults) | Target | Output |
|---|---|---|---|
| `generate` | `--plugin {tangle,gauge,panel}` `--n N` `--out DIR` `--seed 0` `--beacon offline\|drand\|nist` `--profile configs/default_profile.yaml` `--split dev\|test\|sealed` `--twins/--no-twins` `--families CSV` `--workers 1` | `pipeline.generate:run_generate` | section 3 |
| `verify PATH` | `--out PATH/verify_report.json` | `pipeline.verify:run_verify(path,out)` | VerifyReport |
| `mutate` | `--fault M1..M10\|all` `--n 5` `--seed 0` `--out artifacts/mutate/report.json` `--profile` | `pipeline.mutate:run_mutate` | MutateReport |
| `probe` | `--data DIR` `--out artifacts/probes/report.json` `--seed 0` `--folds 5` | `probes.run:run_probes` | ProbeReport |
| `diversity` | `--data DIR` `--out …` `--n 256` `--seed 0` | `diversity:run_diversity` | DiversityReport |
| `export` | `--data DIR` `--format harbor\|hf\|rlvr\|all` `--out DIR` `--limit` | `export.run:run_export(data,fmt,out,limit)` | section 13 |
| `bench` | `--data` `--out` `--models configs/models.yaml` `--offline/--live` `--client replay\|mock` `--replay-dir bench_replay` `--budget-usd` `--track tangle\|panel\|all` `--max-trials 50` `--seed 0` `--dry-run` | `bench.run:run_bench` | BenchReport / preflight.json |
| `selftest` | `--golden tests/golden/golden.json` `--update-golden` `--seeds-only` | `pipeline.selftest:run_selftest(golden,update_golden)` | SelftestReport |
| `report` | `--artifacts artifacts` `--out artifacts/report/report.html` | `report.build:run_report(artifacts,out)` | report.html |
| `new-plugin NAME` | `--dest src/knotwork/plugins` | `scaffold.new_plugin:run_new_plugin(name,dest)` | package |

`generate --seed` is valid only with `--beacon offline` (usage error otherwise). For `bench`, the CLI passes
`client="openrouter"` when `--live` without `--dry-run`. `--dry-run` writes `OUT/preflight.json` and makes no
network call.

### 9.1 Report formats
* **VerifyReport:** `{"ok": bool, "n_bundles": int, "n_ok": int, "failures": [{"bundle_id", "codes":
  [E_*], "detail"}]}`. `verify` re-computes file digests and bundle id (mismatch ⇒ `E_DIGEST_MISMATCH`),
  validates schemas (`E_SCHEMA`), and re-runs `engine.certify` with the stored D1 (`certificate.d1`) — no
  plugin code is imported (`verify.py` MUST NOT import `knotwork.plugins`; test asserts). Exit 1 if not ok.
* **MutateReport:** `{"n_per_fault": int, "faults": {"M1": {"expected": "E_OOB", "plugin": "tangle",
  "injected": int, "caught": int, "caught_exact": int, "primary_match": int, "codes_seen": {code: count}}},
  "catch_rate": float, "exact_catch_rate": float, "clean": {"n": int, "quarantined": int,
  "false_quarantine_rate": float}}`. `caught` = quarantined at all; `caught_exact` = expected code ∈ codes.
* **ProbeReport, DiversityReport:** section 11. **BenchReport, preflight:** section 12.
* **SelftestReport** (`artifacts/selftest/report.json`): `{"ok": bool, "checks": [{"name", "ok", "detail"}]}`;
  selftest also writes the golden dataset to `artifacts/selftest/data/` (2 base items per family of every
  plugin, seed 1234, with twins) and compares `{bundle_id, answer canonical, verdict}` per slot and the
  id-buffer/mask digests with `tests/golden/golden.json` (exact). `--update-golden` rewrites the file.
* **Contact sheet** (`python -m knotwork.pipeline.contact_sheet --data DIR --out PNG`): PNG ≥ 2048×1536 plus
  a sidecar `PNG-stem.json` = `{"rows": [{"name": str, "panels": [{"kind": str, "bundle_id": str|null,
  "caption": str}]}], "faults": [{"fault": "M1", "code": "E_OOB", "bundle_id": str|null}]}`. Rows:
  `tangle`, `gauge`, `panel` (panels start `base`, `causal`, `nuisance`); **Pixel-Witness Proof strips**
  `pixel_witness:tangle`, `pixel_witness:panel` (and optionally `pixel_witness:gauge`) whose panels are
  exactly `beauty` | `id_buffer` (id values mapped through a fixed 20-colour qualitative colormap, 0 =
  black) | `kernel_overlay` (beauty dimmed 50 % with the kernel's evidence drawn on top: the TRACE
  connected component `R` in cyan, every crossing centre with its halo radius circle and ONTOP winner,
  the needle PCA axis line and dial centre, OCR'd label text next to each label box); and one fault row
  with M1–M10 thumbnails captioned `M<k> → <E_CODE>` (from a `mutate` run with `--n 1`). Overlays are
  computed by calling `knotcore` functions on the bundle — never from plugin geometry.

---

## 10. Fault conformance matrix (`knot mutate`)

For each fault, `pipeline.mutate` samples scenes from `("mutate", fault, i, "attempt", a)` until
`fault_effective(scene, fault)` is true (≤ 20 attempts), renders **with** the fault, certifies with the
plugin's normal claims/question/D1, and records the outcome. The same scene rendered **without** the fault is
the clean control (feeds `clean.false_quarantine_rate`). 100 % of injections MUST be quarantined with the
expected code present in `codes` (gate `p2.mutate`).

| Fault | Plugin | Injection (render-time only unless noted) | Effective when | Expected |
|---|---|---|---|---|
| M1 | tangle | one strand's end (and its terminal box) translated +60 px past the right edge (clipped) | always | `E_OOB` |
| M2 | tangle | wires drawn in strand-index order with no per-crossing top redraw (beauty **and** id buffer) | ∃ crossing whose ε-top ≠ index-order top | `E_ZORDER` |
| M3 | tangle | glyphs of two end labels swapped in beauty + label masks; manifest unchanged | always | `E_LABEL_MISMATCH` |
| M4 | tangle | one crossing's column widened ×6 so the local crossing angle < 20° | computed angle < 20° | `E_TANGENT_CROSSING` |
| M5 | tangle | every strand stroke = 2 px at 1024 (1.5 px at 768) | always | `E_ILLEGIBLE_RES` |
| M6 | tangle | opaque `occluder` disc (radius 2·stroke) registered in manifest/id buffer over a crossing | always | `E_OCCLUDED_CROSSING` |
| M7 | tangle | extra dark "shadow" spline drawn on beauty only (no object, no mask, no id) | always | `E_UNEXPLAINED_INK` |
| M8 | gauge | needle angle computed with `sweep × 1.12` (ticks correct) | `|Δv| > 3 × kernel tol` | `E_SCALE_MISMATCH` |
| M9 | gauge | needle pivot displaced by (14, −9) px from the dial centre | always | `E_NEEDLE_OFFCENTER` |
| M10 | tangle | one strand's beauty colour = 92 % background blend (masks/id pristine) | always | `E_BEAUTY_MISMATCH` |

---

## 11. Freshness, shortcut audit, diversity

### 11.1 Freshness Protocol (`freshness/`)
* **Beacons** (`beacon.py`): `drand` — `GET {drand_url}` → `{"round", "randomness"}` (hex); `nist` —
  `GET {nist_url}` → `pulse.outputValue` (hex, 128 hex chars; use its SHA-256 as 64-hex randomness),
  `pulse.pulseIndex` as round; `offline` — `seeds.offline_randomness(seed)`, `round = seed`. All HTTP via
  `knotwork.net.get_json(url)` (httpx, 10 s timeout, policy `beacon`). Tests mock httpx; no live calls.
* **Receipt** (`receipt.py`): `FreshnessReceipt` with `randomness_hex`, `root_seed =
  derive_root_seed(randomness, protocol_commit)`, `emission_mask_commit = SHA256(salt)`, `canary =
  canary_for(root_seed)`, `receipt_id = json_digest(receipt fields without receipt_id)`. `fetched_at_utc` is
  set only for network beacons (ISO-8601 runtime timestamp) and is excluded from `receipt_id`.
* **Emission mask** (`emission.py`): salt = 32 random bytes at `.secrets/emission_salt` (created with
  `os.urandom` on first use, gitignored). For split `test`/`sealed`, an item's answer is *sealed* iff
  `HMAC-SHA256(salt, bundle_id)[0] / 256 < sealed_fraction`; exporters replace a sealed answer with
  `answer_commitment = SHA256(salt ‖ canonical)`. Generation itself is unaffected (stays deterministic).
* **Burn ledger** (`ledger.py`): append-only `freshness/burn_ledger.jsonl` lines `{"bundle_id",
  "scene_id", "slot", "model", "request_sha256"}` for every item sent to a live API; `generate --split
  test|sealed` rejects (resamples) scenes whose `scene_id` is in the ledger.
* **Canary**: embedded in every `task.json`, Harbor `instruction.md` (HTML comment) and `task.toml`
  metadata, HF/RLVR rows, and the dataset card.

### 11.2 Shortcut probes (`probes/`, `knot probe`)
Evaluated on `split=test` data (600 base items per plugin, `--no-twins`) with 5-fold `GroupKFold` by `scene_id`. Chance per item = `1/|options|`
for label answers, `1/7` for T2 (uniform targets 0..6), majority-bin frequency over 10 equal-width bins
for numeric answers, `1/3` for zones, `1/2·1/bins` handling for premise (premise bool × value bin).
Probes (each returns a prediction per item):
* `straight_across` (tangle T1): end label with the start terminal's position index.
* `nearest` (tangle T1; panel P1/P2): end/port terminal whose box centre is nearest (Euclidean) to the
  start terminal box centre; for panel, read/zone of that gauge's *true* value is NOT used — the probe
  predicts the gauge's answer bin by taking the nearest gauge and the majority bin of training answers.
* `color_match` (tangle T1, panel P1/P2): end terminal whose 9×9 patch mean colour at the stroke entering it
  is closest (RGB L2) to the colour at the start terminal.
* `gbdt_image_stats`: `HistGradientBoostingClassifier` on global image statistics only — 16-bin
  histograms per RGB channel, mean/std, Sobel edge density, colourfulness — no spatial features.
* `text_only`: TF-IDF (word 1–2 grams) + logistic regression on the question text (options included).
ProbeReport:
```json
{"plugins": {"tangle": {"n": 600, "pooled": {"chance": 0.21, "probes": {"nearest": {"acc": 0.22,
  "n": 500, "wilson_upper": 0.26}}}, "families": {"T1_TRACE": {"chance": 0.16, "probes": {…}}}}},
 "balance": {"tangle/T1_TRACE/m=6": {"chi2": 4.1, "dof": 5, "p": 0.53}},
 "dedup": {"tangle": {"pairs_within_4": 0, "dup_rate": 0.0}}}
```
`pooled` = all families of the plugin to which the probe applies, chance = mean per-item chance.
**Gate:** Wilson-95 upper bound of every pooled probe ≤ chance + 0.05; every per-family point estimate ≤
chance + max(0.05, 1.96·sqrt(chance·(1−chance)/n_family)) (family slices are small, so a fixed 5 % margin
would false-fail by sampling noise alone). Every probe entry carries its own `n`. Balance: χ² goodness-of-fit of the correct option index vs uniform per (family, #options)
group with ≥ 50 items, and of numeric answers over 10 bins; p ≥ 0.01 / #tests (Bonferroni). Dedup:
64-bit pHash (`imagehash.phash`) Hamming ≤ 4 pairs / n ≤ 0.01.

### 11.3 Wilson bound
`ŵ = (p + z²/2n + z·sqrt(p(1−p)/n + z²/4n²)) / (1 + z²/n)`, `z = 1.959964`.

### 11.4 Vendi Score (`diversity.py`)
Embedding per item: concat of (a) L2-normalised 32×32 grayscale downsample of `beauty.png` (1024-d) and (b)
L2-normalised one-hot of `(family, n_strands or n_gauges, layout, style)`; final vector L2-normalised.
`K_ij = cos(e_i, e_j)`; `λ = eigvalsh(K / n)` clipped at 0; **`VS = exp(−Σ λ_i log λ_i)`** (0·log 0 = 0).
Baseline "recolor-only": take one base scene of the plugin and produce `n` variants that change only colours
(palette permutation / tints) via the plugin's renderer with fixed geometry. DiversityReport:
`{"plugins": {"tangle": {"n": 256, "vendi_ours": 41.3, "vendi_recolor": 2.1, "ratio": 19.7}}}`.
Gate: `ratio ≥ 2.0` for every plugin.

---

## 12. Benchmark (`bench/`)

### 12.1 Prompts (`prompts.py`)
System: *"You are taking a visual reasoning test. Answer from the image only. End your reply with a single
line 'ANSWER: <answer>'."* User content: the image (PNG, 768 px long side, base64 data URL) + the question
text + answer-format line (13.3). Generation limits: `max_tokens` = slot `max_tokens` or
`request_defaults.max_tokens` (512) so reasoning VLMs can finish thinking before the `ANSWER:` line;
optional per-slot `reasoning_effort` (`low|medium|high`) is sent as `"reasoning": {"effort": …}` and omitted
when null. A reply with no parsable `ANSWER:` line counts as wrong **and** as a parse failure.

### 12.2 Adaptive 1-up-2-down staircase (`staircase.py`)
Levels: tangle `k` = number of crossings on the queried strand ∈ {1..12} (T2-free; uses T1 items whose query
strand has exactly `k` crossings); panel level ℓ ∈ {1..8} = crossings on the traced cable (P1 items).
Start at level 3. After **two consecutive correct** → level + 1 (harder); after **one wrong** → level − 1.
Stop at 10 reversals or `max_trials`. Threshold = mean level of the last 6 reversals (all reversals if < 6).
Converges to `P(correct) = sqrt(0.5) = 0.7071`. Items are drawn from the pool by level without replacement
using `("bench", slot, track)` seeds. For mock models `expected_level` = the real `k*` solving
`(1−p)^k + (1 − (1−p)^k)/m̄ = 0.7071` (m̄ = mean #options at that level).

### 12.3 Statistics (`stats.py`)
* **Per-crossing error MLE:** `P(correct | k, m) = (1−p)^k + (1 − (1−p)^k)/m`; maximise
  `Σ log P` / `log(1−P)` over T1 trials with `scipy.optimize.minimize_scalar(bounds=(1e-6, 1−1e-6),
  method="bounded")`; 95 % CI = 2.5/97.5 percentiles of 1000 bootstrap resamples (seeded).
* **Abbott chance correction:** `P* = (P_obs − c)/(1 − c)` with `c` the mean chance of the trials.
* **Composition product law:** on the same P1 panel images ask three questions: TRACE-only ("Which gauge
  (G1–Gk) is the cable from socket S plugged into?", label, chance 1/k), READ-only ("What value does gauge
  G<j> read?", numeric) and the composed `READ(TRACE)` (P1, low-chance numeric). Predicted `P*_comp = P*_trace × P*_read`; `gap = P*_comp_obs − P*_comp_pred`, 95 % bootstrap CI.
* **Colour ablation:** accuracy on `unique` minus `mono` for matched T1 items (same braid word,
  regime changed) → `delta`.
* **Track-switch attribution:** for a wrong T1 answer, the fraction explained by a single switch at some
  crossing `j` (the predicted terminal equals the end of the other strand after switching at `j`).
* **Twin consistency:** `causal_consistency = P(base correct ∧ causal correct)`; `nuisance_consistency =
  P(same answer on base and nuisance)`.
* **Gaslight failure rate:** fraction of `T_GASLIGHT` trials where the model's premise token is "yes".

### 12.4 Clients
`client_base.VLMClient.ask(slot, image_png: bytes, prompt: str, seed: int) -> Reply(text, usage_in,
usage_out, cost_usd, cached: bool)`.
* `client_openrouter`: async `httpx.AsyncClient`, `POST endpoint` with `{"model": slug, "messages": […],
  "temperature": 0, "max_tokens": <slot or 512>, "reasoning": {"effort": <slot>} (only when set),
  "provider": {"data_collection": "deny", "allow_fallbacks": true}, "usage": {"include": true}}`, header `Authorization: Bearer $OPENROUTER_API_KEY`; concurrency 4;
  exponential backoff (3 retries) on 429/5xx; every response saved to the replay cache **before** returning.
* `client_replay`: key = `SHA256(canonical_json({slot, slug, prompt, image_sha256, seed}))`, file
  `bench_replay/<slot>/<key>.json` = `{"request": {…without auth}, "response": {…}, "text": "…",
  "usage": {…}, "cost_usd": x}`; miss ⇒ raise `ReplayMiss` (the run records the trial as missing, never
  falls back to network).
* `client_mock`: deterministic simulated VLMs from `models.yaml: mock_slots` — T1: each crossing on the true
  path switches track with prob `p_cross`; READ: Gaussian noise `σ = read_sigma_frac × span`; gaslight:
  accepts the false premise with prob `gaslight_accept`; unique colours: with prob `color_bonus` answers
  correctly regardless of crossings. Errors across sub-tasks are independent (product law holds).

### 12.5 Budget (`budget.py`)
Preflight: `calls × est_tokens × est_price × per_call_safety_margin` per slot (models.yaml), written to
`preflight.json` `{"estimated_usd", "calls", "per_slot": {slot: usd}, "hard_cap_usd", "budget_usd"}`.
Live: before each call `spent + estimate(call) > budget_usd − stop_when_remaining_usd_below` ⇒ stop
cleanly, mark the report `"stopped": "budget"`. Actual cost = `usage.cost` from OpenRouter when present,
else tokens × price. Every live item is appended to the burn ledger.

### 12.6 BenchReport (`OUT/bench_report.json`)
```json
{"mode": "mock|replay|live", "network_calls": 0, "cost_usd": 0.0, "stopped": null,
 "models": {"mock_a": {
   "staircase": {"tangle": {"threshold_level": 5.2, "expected_level": 5.6, "reversals": [..], "trials": 50},
                 "panel": {...}},
   "per_crossing": {"p_hat": 0.041, "ci": [0.028, 0.055], "n": 400},
   "color_ablation": {"acc_unique": 0.9, "acc_mono": 0.7, "delta": 0.2},
   "track_switch": {"n_errors": 31, "single_switch_explained_frac": 0.87},
   "composition": {"p_trace": .., "p_read": .., "p_comp": .., "chance": {"trace": .., "read": .., "comp": ..},
                   "abbott": {"trace": .., "read": .., "comp_obs": .., "comp_pred": .., "gap": .., "gap_ci": [lo, hi]}},
   "twins": {"causal_consistency": .., "nuisance_consistency": ..},
   "gaslight": {"failure_rate": .., "n": 30},
   "parse_fail_rate": 0.0}}}
```
`parse_fail_rate` = fraction of this model's trials whose reply had no parsable `ANSWER:` line (13.3); it is
reported per model for every mode (0.0 for mock unless the mock is configured to emit unparsable replies).
`network_calls` counts real HTTP requests (0 for mock/replay); gates assert it.

---

## 13. Exporters (`export/`) and grading

### 13.1 Harbor (`knot export --format harbor` → `OUT/harbor/<bundle_id>/`)
```
instruction.md          question + answer instructions + "<!-- KNOTWORK-CANARY-… -->"
task.toml               see below
environment/Dockerfile  FROM python:3.11-slim / WORKDIR /app / COPY image.png /app/image.png / CMD ["bash"]
environment/image.png   beauty.png resized to 768 px (consumption resolution)
solution/answer.json    {"answer": <native oracle value>}
solution/solve.sh       copies $KNOT_SOLUTION_DIR(/solution)/answer.json → $KNOT_APP_DIR(/app)/answer.json
tests/test.sh           runs: python3 "${KNOT_TESTS_DIR:-/tests}/verify.py"; always exit 0
tests/verify.py         STDLIB ONLY: verbatim source of knotcore/grading.py + a main() that reads
                        expected.json and $KNOT_APP_DIR/answer.json, writes reward.txt and reward.json
tests/expected.json     Answer (model_dump, mode="json")
```
`task.toml`:
```toml
schema_version = "1.4"

[task]
name = "knotwork/<bundle_id>"
description = "Knotwork <family> visual reasoning task (<plugin>)"
keywords = ["knotwork", "visual-reasoning", "<plugin>", "<family>"]

[metadata]
category = "visual-reasoning"
difficulty = "easy|medium|hard"   # by kernel difficulty: crossings/depth tertiles
family = "<family>"
bundle_id = "<bundle_id>"
canary = "KNOTWORK-CANARY-…"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 300.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 2048
```
`instruction.md` tells the agent: the image is `/app/image.png`; write `{"answer": <value>}` to
`/app/answer.json`; `<value>` type per 13.3. **Rewards:** `reward.txt` (single number) and `reward.json`
`{"reward": 1.0|0.0}` (numeric values only) in `${KNOT_LOG_DIR:-/logs/verifier}`; metadata (grader sha256,
bundle id, parse status) goes to `verifier_meta.json`. Missing/invalid answer ⇒ reward 0.0. The
`KNOT_*_DIR` overrides let the oracle pytest run the same scripts locally (WSL/Ubuntu; bash optional).
Oracle test (`tests/export/test_harbor_oracle.py`) and gate `p4.export_harbor` assert oracle = 1.0,
wrong = 0.0, missing = 0.0.

### 13.2 HuggingFace + RLVR
* `OUT/hf/data/<split>-00000-of-00001.parquet` (pyarrow, zstd) columns: `bundle_id, plugin, family,
  split, image (struct{bytes: binary, path: string}), question (string), options (list<string>),
  answer_type, answer (canonical string), answer_json (string), tolerance (float64), program_json,
  certificate_json, steps_json, canary, twin_role, base_id, difficulty_json`. `OUT/hf/README.md` dataset card
  with YAML front-matter `configs: [{config_name: default, data_files: [{split: <s>, path:
  data/<s>-*.parquet}]}]`, licence, canary, intended use.
* `OUT/rlvr/tasks.jsonl`: `{"id", "prompt", "image_path": "images/<id>.png", "answer": Answer,
  "steps": [{"op", "inputs", "output"}], "reward_fn": "knotwork.knotcore.grading:grade_prediction",
  "canary"}` plus `OUT/rlvr/images/`.
* Sealed answers (11.1) appear as `answer_commitment` with `answer` omitted.

### 13.3 Answer formats and `grade_prediction(pred, answer) -> float` (`knotcore/grading.py`)
`pred` is either free text or a native value. Free text: if a line matches `^\s*ANSWER:\s*(.+)$`
(case-insensitive, last such line wins) use group 1, else the whole text.

| Type | Canonical | Answer-format line in prompt | Correct iff |
|---|---|---|---|
| label | `"7"`, `"C"` | `ANSWER: <label>` | NFKC/strip/upper equality |
| int | `"3"` | `ANSWER: <integer>` | first integer in text equals value |
| float | `"47"` (`format(v, "g")`) | `ANSWER: <number>` | first number `x` with `|x − v| ≤ tolerance` |
| zone | `"red"` | `ANSWER: <green\|amber\|red>` | exactly one zone word present and equal |
| premise | `"no; 47"` | `ANSWER: <yes\|no>; <number>` | first yes/no token equals `premise_holds` AND first number within tolerance |

Emitted answer tolerance for floats: `max(d3.tolerance, grade_tolerance_frac × span)`. Returns 1.0 or 0.0.
`grading.py` imports only `json, math, re, unicodedata, typing` (test asserts) so it can be copied verbatim.

---

## 14. Gates (mirrors `scripts/verify_phase.py`)

Core (every phase): required paths present (cumulative), knotcore executable LOC ≤ 850 and total ≤ 1350, AST isolation, no kernel
placeholders, `lint-imports` 0 broken, `ruff check` + `ruff format --check`, `mypy` (strict kernel),
cumulative `pytest` over test dirs of phases ≤ N.

| Phase | Gates |
|---|---|
| 0 | font sha256; 10 CLI subcommands; seed KAT identical in two processes; `uv lock --check` |
| 1 | `engine.kernel_sha256() == devtools.kernel_sha256()` |
| 2 | generate tangle 30 / gauge 20 / panel 40 (seed 11): exact base counts, false-quarantine ≤ 0.10, all families present; determinism (same seed ⇒ same ids; `--workers 2` ⇒ same ids; new seed ⇒ disjoint); twins valid; gaslight & controls exact `round_half_up(0.15 n)`; `mutate --n 4` 100 % exact catch; contact sheet ≥ 2048×1536 with Pixel-Witness Proof strips (tangle + panel) and M1–M10 captions → writes `CHECKPOINT_1_READY.md` (incl. optional ≤ $0.50 gap-check one-liner) |
| 3 | `.checkpoints/CP1_APPROVED`; receipt valid + canary propagation; `knot selftest` ok; `knot verify` clean + pixel/answer tamper ⇒ `E_DIGEST_MISMATCH`; probes (600/plugin, split test): pooled Wilson-95 upper ≤ chance + 5 %, per-family point estimate ≤ chance + max(5 %, 1.96·SE); balance; dedup; Vendi ratio ≥ 2 |
| 4 | Harbor export structure + oracle 1.0 / wrong 0.0 / missing 0.0 (20 tasks); HF parquet columns; RLVR `grade_prediction` 1.0/0.0; `new-plugin` scaffold compiles, passes its tests, knotcore unchanged; ci.yml steps |
| 5 | mock bench: `|p_hat − p_true| ≤ max(0.03, CI half-width)` and `p_true ∈ [ci_lo − 0.015, ci_hi + 0.015]`; `parse_fail_rate` present; staircase within 1.5 levels of `k*`; composition gap CI ∋ 0; network_calls 0; replay offline; preflight ≤ $8 → writes `CHECKPOINT_2_READY.md` |
| 6 | CP2 decision file; all CLI targets implemented, zero placeholders in src/apps; ≥ 11 figures ≥ 1600 px; report.html self-contained > 50 KB; deck 11 slides; portal fields ≤ 1024 chars; README/VERIFICATION/DATA_CARD complete; gate JSONs phases 0–5 ok |

---

## 15. Streamlit Forensic Inspector (`apps/inspector.py`) and report

`uv run streamlit run apps/inspector.py`. Structure: pure functions + `main()`; module bottom calls
`main()`. Tested with `streamlit.testing.v1.AppTest.from_file("apps/inspector.py").run()` (no exception, 4
tabs). Tabs:
1. **Live Generator & Fault Injector** — plugin/family/seed/fault selectors → render + certify in a temp dir,
   show beauty, verdict, codes.
2. **Pixel Witness & Twin Diff Viewer** — overlay masks/id buffer/witness rings/crossing centres/needle PCA
   axis on the beauty; base vs causal vs nuisance with pixel-diff heatmap and `pixel_diff_frac`.
3. **Human Staircase Pilot** — the 12.2 staircase with a human answering; results saved to
   `artifacts/human_pilot/<session>.jsonl`; shows running threshold.
4. **Benchmark, Quarantine Sankey & Harbor Browser** — BenchReport charts; Sankey (plotly-free: matplotlib
   or Streamlit native) candidates → {certified, quarantined by code}; browse exported Harbor folders.
`report/figures.py` renders `docs/figures/fig01…fig11*.png` (≥ 1600 px wide, dpi 200) from artifacts; each
maps to a deck slide (docs/SUBMISSION_CHECKLIST.md). `report/build.py` embeds figures as base64 in a single
self-contained `report.html`.

---

## 16. Decision log
Append entries `D-<n>: <decision> — <reason>` here when an implementation choice resolves something this
spec leaves open. Never silently change a threshold.

* D-1: `TaskPlugin.questions` takes an explicit `family` (schedule-driven exact counts) — needed for exact
  gaslight counts and balanced families.
* D-2: OpenRouter "no training" routing is the request-body provider preference `data_collection: deny`
  (OpenRouter has no header for it).
* D-3: Harbor `reward.json` holds numeric values only; metadata goes to `verifier_meta.json`.
* D-5: Kernel LOC is split into an executable cap (850, excludes declarative `contracts.py`) and a total
  cap (1350); a single 1000 cap left ~8 lines of margin under `mypy --strict`.
* D-6: OCR runs once at 768 px with a crop-hash cache; the bench split is 600 items per plugin, keeping the
  Phase 3 gate within a same-day budget while pooled Wilson bounds stay tight (n ≥ 500 per plugin).
* D-7: Premise families are 4-step programs and `engine.execute` constructs the `PremiseValue` answer.
* D-8: Reasoning VLMs need room: `max_tokens` 512 (per-slot override), optional `reasoning.effort`, and
  `parse_fail_rate` reported per model.
* D-9: A separately locked ≤ $0.50 gap check (`scripts/gap_check.py`, `.checkpoints/GAPCHECK_APPROVED`)
  exists for CP1; it cannot reach the CP2 budget path.
* D-4: Probes and bench use `split=test` items rendered in the `mono` colour regime so colour carries no
  answer information; colour effects are measured separately by the matched colour ablation.
* D-10: Windows portability (gate-script bug fix, semantics unchanged): `scripts/verify_phase.py` runs
  children with `PYTHONUTF8=1`/`PYTHONIOENCODING=utf-8`, decodes their output as UTF-8
  (`errors="replace"`), and reads/writes every text file as UTF-8 instead of the locale codepage (cp1252
  on Windows). No threshold or check changed. The repository's GitHub name is `vsfactory`.
* D-11: CW progress `φ` is wrapped to `[−45°, 315°)` (`gauge._phi`) instead of `[0°, 360°)`: zone-arc
  pixels rasterised a fraction of a degree before the scale start would otherwise get `φ ≈ 360°` and make
  the first zone span the whole dial. The 270° sweep leaves a 90° dead band, so no valid pixel moves.
* D-12: Kernel primitives signal a failed check by raising `io.KernelFault(code)`; `engine` catches every
  fault and records it as a certificate code (errors are values at the kernel boundary, one line per
  check instead of tuple plumbing through every primitive).
* D-13: OCR details SPEC 7.9 leaves open: a label's template-fallback group = label objects with the same
  `parent`; candidate glyphs are rendered through the beauty chain (2× font, Lanczos to 1024 then 768)
  with a 4-px background margin (plain 0.75× renders gave < 0.05 margins for P/R/F and 1/10).
* D-14: `TRACE` ending on an unlabelled terminal returns its `terminal_id`; `GAUGE_OF` accepts a port's
  `terminal_id` or its OCR'd label (so `GAUGE_OF($s0)` composes with `TRACE`); the witness samples the
  needle-tip colour inward of the PCA far end (a 5×5 patch on the last pixel straddles background; D-22).
* D-15: Tick numerals sit on face-coloured numeral plates (part of the label object: mask + id) drawn over
  the needle and extending 10 px past the glyphs, so a needle passing a numeral never enters that
  numeral's 768-px OCR crop (without plates the needle corner was read as an extra glyph, e.g. `20V`).
  M9 (pivot moved by (14, −9)) is only observable when that shift has a component perpendicular to the
  needle above the kernel limit `max(3, 0.03 R)`; `fault_effective(M9)` checks exactly that geometry.
* D-16: `sample_scene` receives optional hints in the profile dict: `_family` (T2 target counts, G2 zone,
  panel target/premise truth/ARGMAX margin/red count are sampled so answers are balanced and well posed)
  and `_split` (test/sealed use `benchmark_color_regimes`). Plugins reject and resample internally.
* D-17: M4 widens the chosen crossing's column by max(6, k), k bringing the nominal S-curve crossing to
  ~16°: with dy ≤ 1.3 w (needed for legible crossings) ×6 alone usually leaves ~40°. Effective iff the
  shapely angle of the widened layout is < 20°.
* D-18: Tangle layout details: reduced words come from an odd–even transposition (parallel bubble) sort
  (≤ n packed columns); commuting letters share a column; an ID slot is two barrier columns and a twin
  fills it with a slot-bound R-II pair so no other column moves; every strand gets 24-px straight leads;
  the horseshoe falls back to the strip when its radial budget cannot keep dy ≥ max(3.6 strokes, 34 px)
  (n ≥ 8–9). Panel braids use the reduced word plus an optional R-II pair (no ID slots).
* D-19: Panel geometry: dials (R = 120) in a right-hand column (staggered for 4 gauges); each port is on
  the dial's left rim with its box 8 px from the dial mask (exactly `GAUGE_OF`'s 8-px dilation reach, and
  ≥ 1 cable stroke of clearance); the gauge label `Gk` under the dial is the port's label (TRACE answers
  `Gk`); cables reach ports/pads by Manhattan routing in ordered channels (provably crossing-free);
  distractor cables start and end at unlabelled pads; panel numerals at 0.47 R; cable stroke 7 px.
* D-20: Renderer: 1× (mask/id) strokes use half-width (w − 1)/2 because PIL fills pixels on polygon edges
  (otherwise a "2 px" M5 stroke measures 4 px); crossing halos have flat ends and the top wire is redrawn
  over exactly the halo run (round caps let a redraw overwrite a neighbouring crossing's top).
* D-21: A twin edit that cannot be constructed (e.g. no ID slot next to the query strand) is retried and
  is not a candidate; `false_quarantine_rate` counts rendered candidates only. Gauge D2 is the shapely
  needle-polygon reading; panel D2 is omitted (optional, SPEC 2.3).
* D-22: The witness samples the needle-tip colour 6 px inward of the PCA far end (3 px picked up Lanczos
  ringing at the tip edge on 120-px panel dials).
* D-23: The kernel's RapidOCR engine runs with one intra-/inter-op thread: identical reduction order on
  every machine (determinism independent of core count) and no oversubscription when generation runs one
  process per core. Worker pools also pin BLAS/OpenMP to one thread each.
* D-24: Photographic realism lives entirely in `render/` + `plugins/` (knotcore unchanged). `render.realism`
  applies a seeded finish at output resolution (uneven overhead lighting, powder-coat/brushed grain,
  smudges, dust, sensor grain; dial bevel/grooves, recess and inner face shadow, aged-enamel vignette,
  rim grime, curved-glass glare, metallic hub cap). Kernel-compatibility rules: (a) far (> 4 px) from
  every object the finish may not move a pixel beyond ±36 levels of the manifest background (the
  `E_UNEXPLAINED_INK` threshold is 40), and pixels already outside that band (e.g. injected M7 ink) are
  never pulled back, so the finish cannot hide a defect; (b) high-frequency effects (grain, brushing,
  dust, smudges, glare, grime) are attenuated over label OCR boxes, needles and crossing halos, while
  smooth lighting applies everywhere; (c) noise uses numpy's portable Generator plus filtering, no array
  transcendentals. Causal twins copy the finish parameters (same texture, so only the edit changes
  pixels); nuisance twins resample them (lighting angle, grime, grain, metal finish, dial ageing,
  housing/backplate tone) for a real domain shift.
* D-25: Dial construction: a bevelled metal bezel ring (chrome, black anodized, brass, gunmetal) is part
  of the dial object (concentric, so the kernel's mask-centroid dial centre is unchanged); light
  enamel/cream faces only (dark faces would need light numerals, which the kernel's dark-on-light template
  OCR fallback cannot verify); minor + medium ticks form one `tick_minor` object
  (`gauge.minor_ticks_per_major: 9`, medium at the half mark); tapered pointer with a counterweight disc
  (symmetric about the axis, so the PCA axis is exact) and a beauty-only contact shadow on the face; printed
  unit / serial / accuracy-class marks are one `decor` object, only on dials with R >= 200 and in the dead
  band the needle never sweeps.
* D-26: Cables: matte (`jacket`) or glossy (`harness`) cylindrical shading inside the stroke footprint,
  a beauty-only drop shadow (background darkened 10 %, inside the background tolerance), and heat-shrink
  sleeves (18 px) + ferrules (6 px) drawn as part of the cable object (terminal clearance unaffected);
  terminal labels moved 8 px outward so sleeves stay out of their OCR crops. Panels: corner hex screws
  and printed placards are `decor` objects away from terminals; the DIN rail behind the sockets is
  background relief (no object near a terminal box, which `E_TERMINAL_CLEARANCE` forbids); panel dial
  slots moved so bezels and `Gk` labels stay on the canvas; ports sit 8 px from the bezel.
* D-27: Layout diversity so the Phase 3 pHash dedup gate (Hamming <= 4 pairs / n <= 0.01 on 600 items)
  can hold; all changes are untrusted render/plugin code (knotcore unchanged). Gauge: the dial centre is
  sampled anywhere on the canvas (bezel >= 40 px from the edge), `gauge.radius_px` 300-380 -> 110-240
  (large centred dials were near-identical under a 64-bit pHash: 1.36 dup pairs per item), numerals sit
  at `0.47 + (R - 120) * 0.15 / 140` R (clipped to [0.47, 0.62]) so their plates stay clear of the needle
  tip on small dials, and seeded scene furniture is added: a painted mounting plate (decor object, 85 %),
  a process stem / elbow / header pipe (decor object), 1-2 recessed housing relief panels and four mid-dark
  housing tints. Panel: the whole composition is mirrored left <-> right with p = 0.5 (ports move to the
  right rim, pads/labels/rail/placards mirror), the dial column shifts by a common random (dx, dy) inside
  per-k bounds, the socket block shifts vertically, and 1-2 relief sub-panels are painted. Relief panels
  are beauty-only background darkened 14-30 levels (inside D-24 rule (a), never an object near a
  terminal). Measured on 200-300 item samples: gauge 0.003, panel 0.000, tangle 0.003 dup pairs per item.
* D-28: READ answers are balanced over the 10 equal-width bins of the printed scale: for `G1_READ` and the
  traced gauge of `P1_READ_TRACE` a scale decile is drawn uniformly, then an admissible value inside it
  (values stay >= 3 tol from zone boundaries and scale ends). Uniform-over-admissible sampling failed the
  SPEC 11.2 chi-square (p = 4.5e-5 on 180 G1 items) because the boundary exclusions deplete deciles.
* D-29: Shortcut-probe details SPEC 11.2 leaves open. Targets: label answers -> option index; T2 / COUNT ->
  the integer (chance 1/7 and 1/(n_gauges+1)); zones -> the word; floats -> decile of the queried gauge's
  printed scale (chance = family majority-decile frequency); premise -> `<yes|no>|decile` (chance = 1/2 x
  majority-decile frequency). Learned probes (`gbdt_image_stats`, `text_only`) are trained per family, with
  `P5_PREMISE_CONTROL` and `T_GASLIGHT` pooled (their answers differ only in premise truth, which is the
  thing being tested), 5-fold GroupKFold by `scene_id`, seeds `("probe", name, fold)`. `straight_across`
  uses the manifest's slot order of start and end terminals (= their geometric order in both layouts).
  `color_match` centres the 9x9 patch where a 9x9 window holds the most ink in a 3-16 px band around the
  terminal box (on the entering stroke, not its anti-aliased edge). Panel `nearest` / `color_match` pick a
  gauge and predict the majority training-fold target among items of the same family whose heuristic picked
  the same gauge label (the gauge's true value is never read); the report also gives `trace_hit_rate`.
  Every probe entry carries its own `n` and the mean per-item `chance` of the items it applies to.
* D-30: Exporters. Harbor `difficulty` = tertiles per plugin of `10 x program depth + scene size (strands
  or gauges) + crossings in CROSS evidence`; `tests/verify.py` = verbatim `grading.py` + two constants
  (`GRADER_SHA256`, `BUNDLE_ID`) + `main()`; `test.sh` falls back to `python` when `python3` is absent.
  Sealed items (split test/sealed, emission mask) are skipped by the Harbor exporter (a Harbor task needs
  its oracle) and carry `answer_commitment` (with `answer`/`answer_json` null) in HF and RLVR. HF/RLVR
  images are the 768-px consumption PNGs. `knot new-plugin` scaffolds a COUNT ("how many discs") plugin
  that reuses the `P4_COUNT_ZONE` family id (a new family would be a contracts change).
* D-31: Golden selftest layout: `artifacts/selftest/data/<plugin>/` (one `knot generate` run per plugin
  with an explicit slot plan of exactly two items per family, seed 1234, twins on) plus a combined
  `data/index.jsonl` whose `dir`s are prefixed with the plugin. `run_generate` gained an optional `plan`
  argument (not exposed by the CLI). The golden compares per slot {family, role, bundle_id, canonical
  answer, verdict, id-buffer digest, digest of the mask digests} and the kernel sha256.
* D-32: Diversity tokens `(family, n_strands | n_gauges, layout, style)` come from the scene params,
  re-sampled exactly from the dataset receipt root and the bundle seed path (checked against `scene_id`);
  gauges use the scale kind as their layout token. The recolor baseline tints every `*_rgb` payload field
  (x U(0.75, 1.25) per channel) and permutes + tints cable colour lists of one base scene; nothing else
  changes.
* D-33: Bench specifics. The panel staircase asks the TRACE-only question ("Which gauge (G1-Gk) ...") on
  P1 images, so the SPEC 12.2 formula with m = n_gauges describes it exactly. The staircase level range
  is `[1, min(cap, max level present in the pool)]`; a level whose items are used up is refilled
  (reshuffled) and an empty level draws from the nearest non-empty one; `expected_level` (mock only) is
  clipped to the same range (a k* beyond the measurable range cannot be reached by any staircase).
  Composition chance for numeric answers = min(1, 2 tol / span). Block sizes = `models.yaml plan` x
  `max_trials / 50`; derived items (unique-colour re-renders for the ablation; causal + nuisance twins)
  are re-built from seed paths and re-certified once and shared by all models. The mock's switched trace
  guesses uniformly among the options for label questions and reads a random other gauge in READ(TRACE).
  Replay entries elide the base64 image (stored as its sha256); the request body also carries `seed`.
  `knotwork.net.apost_json` is the async gateway used by the OpenRouter client.
* D-34: Exact answer balance from a per-family slot ordinal. `generate` passes `_ordinal` (the slot's
  index among same-family slots of the schedule) as a profile hint and records the slot schedule in
  `stats.json` (`schedule`) so scenes re-build exactly (diversity, bench derivations). Plugins use it
  only where answers would otherwise be balanced in expectation alone: T1 cycles n over 3..10 and, within
  each n, the answer position (the queried strand is the one ending there); T2 cycles the target count
  0..6; T3 lists the top strand at option index `ordinal mod 2`; G2 / P2 / premise families cycle the
  needle zone; G1 / P1 cycle the scale decile (D-28); gaslight items alternate the false claimed zone.
  On the first 600-item split, sampling noise alone (e.g. n = 3 T1 items answered by option 0 in 16 of 25)
  let `text_only` and a constant-majority predictor clear chance by more than the pooled Wilson margin
  allows. Without the hint (mutate, twins, unit tests) plugins sample as before.
* D-35: Raster bits are not portable across platforms. A Windows-recorded golden and replay cache
  failed on Ubuntu CI: Lanczos (libm transcendentals for filter coefficients), PNG encoding (bundled
  zlib builds) and the float realism finish differ between platform builds of the same pinned wheels,
  so beauty digests -- and with them bundle ids -- differ. Determinism stays guaranteed per platform
  (same seed => identical ids for any `--workers`, gate `p2.determinism`); across platforms the
  semantics are guaranteed. Consequences: (a) replay keys use `image_id` = digest of the decoded pixels
  of the *source* beauty.png instead of the re-encoded 768-px PNG bytes (mock VLM seeds likewise);
  (b) bench derived renders are indexed (`derived/index.json`) and test fixtures ship them as
  `tests/bench/fixtures/data/_derived`, so replays never re-render; (c) the golden records its
  `platform`; on that platform everything is compared bit-exact (gate `p3.selftest`), elsewhere
  family / role / canonical answer / verdict and the kernel sha256 must match exactly while bundle-id
  and id-buffer / mask digest differences are reported as a `raster_portability` count. The live
  replay cache was re-keyed in place (renames only) and replays the live run exactly.
* D-36: Live-run data handling (owner decision): the replay cache `bench_replay/` and
  `freshness/burn_ledger.jsonl` stay local and gitignored (task 5.7 asked to commit the cache; gate
  `p6.cp2` only checks it exists on disk). gpt-5.4 runs with `reasoning.effort: none` and
  `max_tokens: 512`: the canary showed `low` spending the whole 1024-token budget on reasoning and
  returning empty replies. The live client returns a cached response for a request it has already
  paid for (resume-safe), retries timeouts like 5xx, and records a failed request as a trial error
  instead of aborting the run. Report figures label their source (live / replay / mock) in every
  subtitle.
* D-37: Additive true-premise control (owner request after the live run). For each of the 24
  `T_GASLIGHT` items asked live, `python -m knotwork.bench.p5_control` re-certifies the same scene
  with the claimed zone set to the traced gauge's true zone, giving a `P5_PREMISE_CONTROL` question
  on pixel-identical input (prompt differs in the zone word only; same parser, seed, models config).
  `bench_split` and the 997 live trials are not rewritten; records go to
  `artifacts/phase5/p5_control/`. The yes-rates are compared with McNemar's exact test (matched
  pairs), Fisher's exact test next to it. A live run is this module with `--live --budget-usd` at
  most $0.50, behind the same triple lock as `knot bench --live`; it is a second command allowed to
  spend, beside the one in TASKS 5.7, and needs the owner's explicit go-ahead.
* D-38: v2 build (owner direction, branch `feat/vtf-v2`). (a) The 850 / 1350 kernel LOC caps are now
  informational telemetry; `core.loc` warns and never fails, while `core.ast_isolation` enforces the
  TCB contract (`kernel_audit.tcb_report`: allow-listed imports, no network / process / dynamic-code
  modules, no `SceneIR.payload` access, deterministic `kernel_sha256`). (b) CI is tiered: `ci.yml`
  runs tier0 then tier1 with pytest-xdist, `nightly.yml` runs tier2 and the phase gates; gate
  `p4.ci` checks both files. (c) ONTOP is decided by the majority of non-halo id pixels in a core
  disk (`topo.core_vote`) with a ring-continuity check for close votes; `E_WITNESS_OCCLUDED` reports a
  background or foreign object where a strand or needle must be on top; `claims_failed` counts unique
  claims and `E_D1_D3_MISMATCH` is recorded even when a program step faults; `grade_prediction`
  rejects replies that commit to different answers unless `strict_single_answer=False`.
* D-39: v2 trust gaps (owner direction). (a) Panel render faults M11 (two socket cables exchange
  gauge ports -> `E_CLAIM_MISMATCH`), M12 (occluded panel crossing -> `E_OCCLUDED_CROSSING`), M13
  (target needle moved -> `E_SCALE_MISMATCH`), M14 (port labels exchanged -> `E_LABEL_MISMATCH`).
  (b) `knotcore.oracle` re-reads `beauty.png` alone along kernel-claimed coordinates
  (`E_ORACLE_GHOST`). (c) `knotcore.cvd` requires distinct strand colours and gauge zone colours to
  stay >= 8 CIELAB units apart under simulated protanopia and deuteranopia (`E_CVD_AMBIGUOUS`);
  palette entry #01665E (5.0 units from #882255) became #1111FF. (d) Premise scenes randomise each
  dial's zone colour order and balance the target value's decile, so the claimed zone carries no
  numeric information; probes score both premise families as one yes/no task (chance 1/2) and
  `probes.baselines` requires majority-class, mean-luminance, option-position and text-length
  baselines <= chance + 0.02. (e) T1 / T3 answer slots are balanced within blocks but shuffled by a
  run-seeded permutation. (f) The staircase steps by 2 until the third reversal, then 1, runs up to
  75 trials or 20 reversals, and reports a logistic maximum-likelihood `k*` over all trials; the
  mean of the last 6 reversals stays as `reversal_level`.

