"""Plain-English microcopy for the Studio: hover tooltips, page guides, the guided tour and the
glossary. One module so the wording stays consistent and factual across all five pages.

Invariants (checked by `tests/apps/test_guide.py`):
* every tooltip is at most `MAX_TOOLTIP` characters and says what a control does and why;
* every page in `PAGES` has a guide and exactly one tour step, in judge order;
* no text overclaims: the kernel works from renderer object masks cross-checked against the
  visible image (never "from the image alone"), and the kernel is governed by its TCB contract,
  not a line-count cap.
"""

from __future__ import annotations

from typing import Final, TypedDict

MAX_TOOLTIP: Final = 200

#: verified wording for how the kernel certifies an answer (README, VERIFICATION.md)
KERNEL_CLAIM: Final = (
    "re-derives every answer from renderer object masks, cross-checked against the visible image "
    "via a beauty-only sanity oracle, colourblindness (CVD) gate, and stray-ink witness"
)


class PageGuide(TypedDict):
    title: str
    summary: str
    steps: list[str]
    try_first: str
    terms: list[str]


class TourStep(TypedDict):
    page: str
    goal: str
    tip: str


#: page key -> (navigation title, page file relative to apps/inspector.py)
PAGES: Final[dict[str, tuple[str, str]]] = {
    "studio": ("Studio", "app_pages/studio.py"),
    "arena": ("Model Arena", "app_pages/arena.py"),
    "forensics": ("Forensics", "app_pages/forensics.py"),
    "human_eval": ("Human Eval", "app_pages/human_eval.py"),
    "export": ("Export", "app_pages/export.py"),
}

GLOSSARY: Final[dict[str, str]] = {
    "TCB (Trusted Computing Base)": "The small kernel, knotcore, that issues certificates; an "
    "import contract stops it from loading plugin, network or dynamic code.",
    "D1 = D3 Verification": "The plugin's symbolic answer (D1) must equal the answer the kernel "
    "re-derives from the rendered masks (D3), or the item is quarantined.",
    "Beauty-Only Oracle": "A sanity check that reads the anti-aliased image a model sees, so a "
    "mask that disagrees with the visible picture cannot certify.",
    "CVD Gate": "A colourblindness check: when colour carries meaning, cables must stay "
    "distinguishable under simulated deuteranopia and protanopia.",
    "Stray-Ink Witness": "Visible ink that no object mask explains must stay under a small area "
    "limit (E_UNEXPLAINED_INK), so the picture cannot show what the masks do not.",
    "Quarantine (E_* codes)": "Where rejected renders go, each tagged with the exact error code "
    "(for example E_LABEL_MISMATCH) that explains why.",
    "Causal vs. Nuisance Twins": "A causal twin makes a minimal edit that changes the answer; a "
    "nuisance twin restyles the scene and keeps the answer.",
    "OOD Holdout Split": "A split that holds out whole question families or parameter ranges, to "
    "test generalisation rather than memorisation.",
    "Wilson 95% CI": "A confidence interval for an accuracy that stays honest at small sample "
    "sizes and near 0 % or 100 %.",
    "Cost per Correct ($/correct)": "Total spend for a model on a block divided by its correct "
    "answers: what one right answer costs.",
}

PAGE_GUIDES: Final[dict[str, PageGuide]] = {
    "studio": {
        "title": "Studio",
        "summary": "Generate one task, watch the kernel certify it, and see the evidence it used.",
        "steps": [
            "Pick a plugin (Rack patch-bay or Tangle are good first picks) and a preset.",
            "Press Generate & Certify Task.",
            "Switch the view to Witness Overlay to see the kernel's evidence.",
            "Open Batch & OOD Build to make a whole dataset with twins and holdouts.",
        ],
        "try_first": "Choose Rack patch-bay, press Generate & Certify Task, then open Witness "
        "Overlay.",
        "terms": ["D1 = D3 Verification", "Beauty-Only Oracle", "CVD Gate", "OOD Holdout Split"],
    },
    "arena": {
        "title": "Model Arena",
        "summary": "Run vision models on certified items and export a research-grade report.",
        "steps": [
            "Press New session, name it and keep data and results in separate folders.",
            "Choose Offline Mock (no key) or Live (OpenRouter), and set a spend cap.",
            "Add blocks: one domain, preset and sample count each.",
            "Launch, then open the Report tab to export HTML, Markdown, JSON, CSV and a ZIP.",
        ],
        "try_first": "Open a finished session in the sidebar and look at its leaderboard and "
        "Report tab.",
        "terms": ["Wilson 95% CI", "Cost per Correct ($/correct)", "Quarantine (E_* codes)"],
    },
    "forensics": {
        "title": "Forensics",
        "summary": "Check the fault matrix and tell vision failures from reasoning failures.",
        "steps": [
            "Read the KPI strip: yield, quarantine rate and fault matrix coverage.",
            "In Counterfactual twins, compare the original item with its twins.",
            "Open Fault proofs to see a clean render certified and a faulty one quarantined.",
        ],
        "try_first": "Open Fault proofs: the committed M3 pair shows E_LABEL_MISMATCH being "
        "raised.",
        "terms": ["Causal vs. Nuisance Twins", "Quarantine (E_* codes)", "Stray-Ink Witness"],
    },
    "human_eval": {
        "title": "Human Eval",
        "summary": "Answer blind certified items and compare your score with three VLMs.",
        "steps": [
            "Look at the image and read the question.",
            "Pick or type your answer and press Submit.",
            "Check the verdict, then turn on the witness overlay to see why.",
        ],
        "try_first": "Answer three items and compare your score with GPT-5.4, Gemini 2.5 Flash "
        "and Qwen3-VL-235B.",
        "terms": ["D1 = D3 Verification", "Wilson 95% CI"],
    },
    "export": {
        "title": "Export",
        "summary": "Write certified datasets in the formats evaluation and training tools read.",
        "steps": [
            "Choose a dataset (a Studio batch is preselected).",
            "Choose formats: Parquet / HF, Harbor, RLVR; a manifest is always written.",
            "Press Export dataset, then verify lineage against the files.",
        ],
        "try_first": "Export the preselected dataset and press Verify against files under "
        "Manifest & lineage.",
        "terms": ["TCB (Trusted Computing Base)", "OOD Holdout Split"],
    },
}

TOUR_STEPS: Final[list[TourStep]] = [
    {
        "page": "studio",
        "goal": "Inspect a certified task and its proof",
        "tip": "Generate a Rack patch-bay or Tangle task, then switch to Witness Overlay.",
    },
    {
        "page": "arena",
        "goal": "Compare models on certified items",
        "tip": "Run a multi-block campaign (Offline Mock or Live) with separate data and results "
        "folders, then export the report.",
    },
    {
        "page": "forensics",
        "goal": "Check the fault proofs and twins",
        "tip": "14 / 14 fault types are caught (56 / 56 injections); the twins separate vision "
        "from reasoning failures.",
    },
    {
        "page": "human_eval",
        "goal": "Solve a blind task yourself",
        "tip": "Compare your score with GPT-5.4, Gemini 2.5 Flash and Qwen3-VL-235B.",
    },
    {
        "page": "export",
        "goal": "Export a certified dataset",
        "tip": "Write Parquet / HF, Harbor and RLVR, with a SHA-256 manifest and lineage.",
    },
]

TOOLTIPS: Final[dict[str, str]] = {
    # header utilities
    "guided_mode": "Show or hide the step strip and tips at the top of every page.",
    "page_guide": "Steps, what to try first and key terms for this page.",
    "quick_tour": "A five-step walkthrough of the Studio, with a button to jump to each page.",
    "next_step": "Go to the next page of the guided tour.",
    "hide_guide": "Hide the guide strip on every page. Turn Guided mode back on to show it.",
    # Studio
    "plugin": "The scene family to draw. Every plugin goes through the same trusted kernel.",
    "preset": "Clean: easy and legible. Hard logic: long reasoning chains. High stress: noise, "
    "occlusion and realism. CVD: colour-vision check.",
    "generate": "Render a new scene and certify it. Only tasks the kernel verifies are shown "
    "as certified.",
    "view": "Beauty is what a model sees. Witness Overlay shows the kernel's evidence. ID Buffer "
    "shows pixel owners. CVD simulates colourblindness.",
    "deficiency": "Which colour-vision deficiency to simulate for the CVD view.",
    "family": "The kind of question asked about the scene, such as tracing a cable or counting "
    "crossings.",
    "seed": "Fixes every random choice. The same seed and settings give the same task.",
    "reasoning_load": "How many steps the question needs, such as crossings to follow. Auto "
    "lets the preset decide.",
    "visual_stress": "Clutter and difficulty of the picture itself. Higher makes reading harder "
    "without changing the logic.",
    "occlusion": "How much of the scene is partly covered. The kernel still requires the answer "
    "to be visible.",
    "realism": "Surface texture and lighting. Changes the look, never the answer.",
    "color_regime": "unique: one colour per cable. partial: some repeats. mono: one colour, so "
    "colour cannot stand in for tracing.",
    "fine_knobs": "Low-level render settings for stroke, jitter, noise and capture resolution.",
    "stroke_px": "Cable stroke width in pixels. Thinner strokes are harder to follow.",
    "jitter_px": "Random wobble in cable paths, in pixels.",
    "noise": "Simulated sensor noise on the final image.",
    "dpi_scale": "Capture resolution relative to full size. Lower values blur fine detail.",
    "certified_answer": "The answer the kernel re-derived and confirmed. It is the label for "
    "this task.",
    "plugin_family": "Which plugin drew the scene and which question family was asked.",
    # batch build
    "bt_n": "How many certified base items to build. Twins are added on top.",
    "bt_split": "dev and test for evaluation, sealed for held-back answers, ood for a "
    "holdout that tests generalisation.",
    "bt_held_families": "Question families kept out of this split, for an OOD holdout.",
    "bt_held_params": "Parameter ranges kept out of this split, such as strands=9:10.",
    "bt_twins": "Also build a causal twin (answer changes) and a nuisance twin (answer kept) for "
    "each item.",
    "bt_seed": "Seed for the whole batch. The same seed rebuilds the same dataset.",
    "bt_workers": "Parallel render processes. 0 uses all but one CPU core.",
    "bt_shard": "Items per output shard folder. 0 writes one flat folder.",
    "bt_out": "Folder the dataset is written to. The Export page preselects it.",
    "bt_launch": "Start the batch in the background. Progress appears below.",
    # Forensics
    "fx_run": "The dataset whose yield and quarantine rate the KPI strip reports.",
    "kpi_yield": "Share of rendered candidates the kernel certified in the selected run.",
    "kpi_quarantine": "Share of clean candidates the kernel rejected. Lower is better.",
    "kpi_faults": "Fault types (M1–M14) caught with their exact expected code. 4 injections each.",
    "fx_section": "Twins separate vision from reasoning failures. Quarantine lists rejected "
    "renders. Fault proofs show injected defects caught.",
    "cf_out": "Folder holding a twin report (suites.json). The committed sample opens first.",
    "cf_build": "Mine twin suites from items that models answered wrong.",
    "cf_data": "Folder holding the failed items' certified bundles.",
    "cf_fail": "File or folder listing which items failed, with the models' replies.",
    "cf_demo": "Generate a fresh item and build all its twins. No model failed it.",
    "cf_item": "Which failed item to diagnose.",
    "cf_reply": "Type what a model answered on this twin. The diagnosis updates as you type.",
    "q_record": "A quarantined candidate and its primary error code.",
    "q_replay": "Re-render this candidate from its seed and certify it again, with failed checks "
    "marked.",
    "fp_fault": "The defect to inject, the plugin it applies to and the code the kernel must raise.",
    "fp_seed": "Scene to inject the fault into. The clean control uses the same seed.",
    "fp_go": "Render the scene with and without the fault and run both through the kernel.",
    # Human Eval
    "he_answer": "Your answer, read from the image. It is graded against the certified answer.",
    "he_submit": "Record your answer and response time, and show the verdict.",
    "he_witness": "Draw the kernel's evidence on the image: terminals, crossings and labels.",
    "he_next": "Show the next blind item.",
    "he_participant": "Your id for this session. Responses are stored under it.",
    "he_item_set": "The certified items to answer. Items the models answered allow a fair "
    "comparison.",
    "he_download": "Download every recorded response as JSON lines.",
    "he_session": "Your accuracy in this session.",
    "col_model": "The vision model from the committed live run.",
    "col_he_accuracy": "Accuracy on the committed live benchmark run.",
    "col_items": "Items the model answered in the live run.",
    "col_shared": "Accuracy on items that both you and the model answered.",
    # Export
    "ex_dataset": "The certified dataset to export. A Studio batch you just built is preselected.",
    "ex_formats": "Parquet / HF for datasets, Harbor for task folders with an oracle, RLVR for "
    "reward-based training.",
    "ex_shard": "Split parquet files above this size in MB. 0 writes one file.",
    "ex_limit": "Export only the first N bundles. 0 exports all of them.",
    "ex_out": "Folder the export is written to.",
    "ex_go": "Write the files, then check each against the manifest's SHA-256 hashes.",
    "ex_bundles": "All bundles in the dataset, twins included.",
    "ex_bases": "Base items, before twins are added.",
    "ex_manifest": "The manifest lists every exported file with its SHA-256 hash.",
    "ln_bundle": "The bundle whose lineage graph is drawn.",
    "ln_check": "Recompute the hashes from the files and compare them with the lineage record.",
    # Model Arena: sidebar
    "ar_new": "Start a new evaluation session with an empty setup form.",
    "ar_search": "Filter sessions by name, client or model.",
    "ar_filter": "Show sessions by state.",
    "ar_clone": "Same blocks, models and presets; fresh seeds and folders.",
    "ar_rename": "Rename the session or set its client and tags.",
    "ar_compare": "Pick another session to see a regression diff in the Compare tab and report.",
    "ar_archive": "Remove the session from this list. Its folders on disk are kept.",
    "ar_mount": "A results folder from another session or machine, added to this list.",
    # Model Arena: setup
    "ar_name": "A readable name. It also sets the session id and default folder names.",
    "ar_client": "Optional. Printed on reports.",
    "ar_eval": "Optional. Who ran the evaluation, printed on reports.",
    "ar_notes": "Optional. What this run is for, printed on reports.",
    "ar_data_mode": "Generate certifies new items. Reuse freezes an existing dataset, so every "
    "model sees the same images and answers.",
    "ar_reuse_dir": "A certified dataset folder. Its images and answers stay frozen.",
    "ar_data_dir": "Images, task.json, certificate.json and dataset_manifest.json are written "
    "here, apart from results.",
    "ar_results_dir": "config.json, results.jsonl and reports/ go here, so one dataset can serve "
    "many sessions.",
    "ar_mode": "Offline Mock simulates models with no key or charges. Live calls OpenRouter "
    "under your spend cap.",
    "ar_override": "Use a different OpenRouter key for this browser session only.",
    "ar_key": "Kept in this browser session only and passed to the runner. Never written to disk.",
    "ar_cap": "Hard spend limit, checked before every live call. The run stops cleanly at the cap and can "
    "resume with a higher one.",
    "ar_conc": "How many model calls run at the same time.",
    "ar_rpm": "Maximum requests per minute, to stay within provider rate limits.",
    "ar_delay": "Simulated latency per mock call, so offline runs behave like live ones.",
    "ar_repeats": "Ask each item k times. k > 1 adds pass@k and pass^k (ZeroBench's consistency "
    "protocol). Live calls use temperature 0, so this measures answer stability.",
    "col_rel_items": "Items with all k repeats scored. Partly answered items are left out.",
    "col_pass1": "pass@1: mean single-try accuracy over every repeat.",
    "col_passk": "pass@k: share of items answered correctly at least once in k tries.",
    "col_passhatk": "pass^k: share of items answered correctly on all k tries, the strict "
    "reliability bar.",
    "ar_refresh": "Fetch the current vision-model list from OpenRouter. Nothing is fetched "
    "until you click; offline, a curated list is used.",
    "ar_free_only": "Show only models with a :free variant.",
    "ar_models": "Models that answer every block unless a block picks its own.",
    "ar_custom_model": "Any OpenRouter model id, such as vendor/model or with :nitro.",
    "ar_add_model": "Add the custom model id to the list.",
    # Model Arena: blocks and launch
    "blk_domain": "The plugin that draws this block's items.",
    "blk_preset": "Difficulty preset for this block's items.",
    "blk_n": "Items in this block, from 1 to 100,000.",
    "blk_seed": "First seed for this block. Items use consecutive seeds.",
    "blk_quick": "Set the sample count with one click.",
    "blk_default": "Use the default models, or pick models for this block only.",
    "blk_models": "Models that answer this block only.",
    "ar_add_block": "Add another block with its own domain, preset and count.",
    "ar_launch": "Start the campaign in the background. It survives page reloads and can resume.",
    "pf_images": "Items to render and certify before any model is asked.",
    "pf_trials": "Total model calls: items times models, summed over blocks.",
    "pf_time": "Estimated wall time at the chosen concurrency and rate limit.",
    "pf_cost": "Upper bound from catalog prices. Mock runs charge nothing.",
    # Model Arena: running session
    "ar_pause": "Pause after the calls in flight finish. Resume continues without repeats.",
    "ar_stop": "Stop after the calls in flight finish. You can still resume later.",
    "ar_resume": "Continue from the last recorded trial. No trial is asked twice.",
    "ar_retry": "Re-ask only the calls that failed with an API error.",
    "kpi_trials": "Model calls finished out of the total planned.",
    "kpi_yield_arena": "Share of generated items the kernel certified.",
    "kpi_best": "The most accurate model so far, with its Wilson 95% interval.",
    "kpi_spend": "Money spent so far. In mock mode, what a live run would have cost.",
    "kpi_genverify": "Per item: render, then kernel certification; API is the mean model latency.",
    "col_block": "Domain and preset of the block.",
    "col_accuracy": "Correct answers divided by scored answers.",
    "col_ci": "Wilson 95% confidence interval for the accuracy.",
    "col_correct": "Correct answers out of scored answers.",
    "col_latency": "Median and 95th percentile response time.",
    "col_tokens": "Prompt plus completion tokens.",
    "col_cost": "Spend for this model on this block.",
    "col_cost_per_correct": "Spend divided by correct answers: what one right answer costs.",
    "col_api_errors": "Calls that failed at the API. Retry failed re-asks them.",
    "ar_ins_block": "The block whose items to inspect.",
    "ar_ins_item": "An item with its certified answer and how many models got it right.",
    "ar_ins_witness": "Draw the kernel's evidence on the item.",
    "ar_send_forensics": "Build counterfactual twins for this failure and open them in Forensics.",
    "ar_brief": "Add an AI-written executive summary. Needs a live key and sends only the computed "
    "metrics.",
    "ar_brief_model": "The model that writes the executive brief.",
    "ar_report_go": "Compute the statistics offline and write HTML, Markdown, JSON, CSV and an "
    "audit ZIP.",
}


def tip(key: str) -> str:
    """The tooltip for `key`. Raises KeyError for an unknown key, so a typo fails the page tests
    instead of showing an empty bubble."""
    return TOOLTIPS[key]


def tour_index(page: str) -> int:
    """0-based position of `page` in the tour."""
    return next(i for i, step in enumerate(TOUR_STEPS) if step["page"] == page)


def next_page(page: str) -> str | None:
    """The page after `page` in the tour, or None on the last step."""
    i = tour_index(page)
    return TOUR_STEPS[i + 1]["page"] if i + 1 < len(TOUR_STEPS) else None


def page_key(title: str) -> str | None:
    """Page key for a navigation title such as `Model Arena`."""
    return next((k for k, (name, _) in PAGES.items() if name == title), None)
