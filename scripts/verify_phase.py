#!/usr/bin/env python3
"""Knotwork phase gatekeeper.

    uv run python scripts/verify_phase.py --phase N          # core gates + phase-N gates
    uv run python scripts/verify_phase.py --phase N --upto   # core gates + phases 0..N gates
    uv run python scripts/verify_phase.py --list             # print the gate plan

A phase is DONE only when this script exits 0 for it. Results are written to
artifacts/gates/phase<N>.json (machine-readable) and printed as a table.
Every threshold below is normative and mirrored in SPEC.md section 14; change both or neither.

Exit codes: 0 all gates passed | 1 at least one gate failed | 2 usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ART = REPO / "artifacts"
GATES_DIR = ART / "gates"
CKPT = REPO / ".checkpoints"
PY = sys.executable
FONT = REPO / "assets" / "fonts" / "DejaVuSans.ttf"

# ----------------------------------------------------------------- normative thresholds
GEN_N = {"tangle": 30, "gauge": 20, "panel": 40}  # phase-2 certified base items per plugin
GEN_SEED = 11
MAX_FALSE_QUARANTINE = 0.10
MUTATE_N = 4
CAUSAL_MAX_PIXEL_FRAC = 0.05
GASLIGHT_FRAC = 0.15
BENCH_SPLIT_N = 600  # per plugin, split=test, --no-twins (probe/diversity/bench pool)
PROBE_MARGIN = 0.05
BALANCE_ALPHA = 0.01
MAX_DUP_RATE = 0.01
VENDI_MIN_RATIO = 2.0
REQUIRED_PROBES = {
    "tangle": ["straight_across", "nearest", "color_match", "gbdt_image_stats", "text_only"],
    "gauge": ["gbdt_image_stats", "text_only"],
    "panel": ["nearest", "color_match", "gbdt_image_stats", "text_only"],
}
HARBOR_ORACLE_SAMPLE = 20
MOCK_P_ABS_TOL = 0.03
MOCK_CI_SLACK = 0.015
STAIRCASE_LEVEL_TOL = 1.5
PORTAL_LIMIT = 1024
#: Official Innodata 11-slide template order (docs/SUBMISSION_CHECKLIST.md section C).
OFFICIAL_SLIDES = [
    "Cover",
    "Brief about the Idea",
    "Opportunity / Differentiation / USP",
    "List of Features Offered",
    "Process Flow / Use-Case Diagram",
    "Wireframes / Mock Diagrams",
    "Architecture Diagram",
    "Technologies Used",
    "Estimated Implementation Cost",
    "Hackathon Deliverables",
    "Thank You",
]
TEMPLATE_PHRASES = {
    3: "measures true reasoning performance while preventing test data leakage",
    8: "pytest or Harbor framework for evaluation",
}
MIN_FIGURES = 11
MIN_FIGURE_WIDTH = 1600

PHASE_TESTS: dict[int, list[str]] = {
    0: ["tests/conformance"],
    1: ["tests/kernel"],
    2: ["tests/plugins", "tests/pipeline"],
    3: ["tests/freshness", "tests/probes"],
    4: ["tests/export", "tests/scaffold"],
    5: ["tests/bench"],
    6: ["tests/apps", "tests/docs"],
}

PHASE_FILES: dict[int, list[str]] = {
    0: [
        "uv.lock",
        "assets/fonts/DejaVuSans.ttf",
        "src/knotwork/render/canvas.py",
        "tests/conformance/test_render_smoke.py",
    ],
    1: [
        f"src/knotwork/knotcore/{m}.py"
        for m in (
            "io",
            "ocr",
            "topo",
            "gauge",
            "logic",
            "witness",
            "legibility",
            "engine",
            "grading",
        )
    ],
    2: [
        *(
            f"src/knotwork/plugins/tangle/{m}.py"
            for m in ("braid", "layout", "render", "geom", "plugin")
        ),
        *(f"src/knotwork/plugins/gauge/{m}.py" for m in ("scale", "render", "geom", "plugin")),
        *(f"src/knotwork/plugins/panel/{m}.py" for m in ("compose", "programs", "plugin")),
        "src/knotwork/plugins/registry.py",
        *(
            f"src/knotwork/pipeline/{m}.py"
            for m in ("io", "certify", "generate", "twins", "mutate", "contact_sheet")
        ),
    ],
    3: [
        *(f"src/knotwork/freshness/{m}.py" for m in ("beacon", "emission", "ledger", "receipt")),
        *(
            f"src/knotwork/probes/{m}.py"
            for m in ("heuristics", "gbdt", "text_only", "stats", "run")
        ),
        "src/knotwork/net.py",
        "src/knotwork/pipeline/verify.py",
        "src/knotwork/pipeline/selftest.py",
        "tests/golden/golden.json",
    ],
    4: [
        *(f"src/knotwork/export/{m}.py" for m in ("harbor", "hf", "rlvr", "run")),
        "src/knotwork/scaffold/new_plugin.py",
        ".github/workflows/ci.yml",
    ],
    5: [
        *(
            f"src/knotwork/bench/{m}.py"
            for m in (
                "staircase",
                "prompts",
                "client_base",
                "client_mock",
                "client_replay",
                "client_openrouter",
                "budget",
                "stats",
                "run",
            )
        ),
        "tests/bench/fixtures/replay",
        "tests/bench/fixtures/data",
    ],
    6: [
        "apps/inspector.py",
        "src/knotwork/report/build.py",
        "src/knotwork/report/figures.py",
        "README.md",
        "VERIFICATION.md",
        "DATA_CARD.md",
        "docs/DECK_COPY.md",
        "docs/PORTAL_SUBMISSION.md",
        "docs/SUBMISSION_CHECKLIST.md",
        "samples/README.md",
        "samples/harbor",
        "samples/hf",
        "samples/logs",
    ],
}


class GateFailure(Exception):
    pass


@dataclass
class Result:
    gate: str
    ok: bool
    detail: str
    seconds: float


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise GateFailure(msg)


def _tail(text: str, n: int = 25) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-n:])


def run(
    cmd: list[str],
    *,
    timeout: int = 3600,
    env: dict[str, str] | None = None,
    cwd: Path = REPO,
    ok_codes: tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess[str]:
    full_env = {
        **os.environ,
        "KNOTWORK_NETWORK": "deny",
        "PYTHONHASHSEED": "0",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        **(env or {}),
    }
    full_env.pop("OPENROUTER_API_KEY", None)
    try:
        cp = subprocess.run(
            cmd,
            cwd=cwd,
            env=full_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GateFailure(f"command not found: {cmd[0]} ({exc})") from None
    except subprocess.TimeoutExpired:
        raise GateFailure(f"timeout after {timeout}s: {' '.join(cmd)}") from None
    if cp.returncode not in ok_codes:
        raise GateFailure(
            f"`{' '.join(cmd)}` exited {cp.returncode}\n{_tail(cp.stdout)}\n{_tail(cp.stderr)}"
        )
    return cp


def tool(name: str) -> list[str]:
    exe = shutil.which(name)
    if exe:
        return [exe]
    module = {"ruff": "ruff", "mypy": "mypy", "pytest": "pytest", "lint-imports": None}[name]
    if module is None:
        raise GateFailure(f"`{name}` not on PATH; run inside the uv environment (uv run ...)")
    return [PY, "-m", module]


def knot(
    *args: str, ok_codes: tuple[int, ...] = (0,), timeout: int = 7200
) -> subprocess.CompletedProcess[str]:
    return run([PY, "-m", "knotwork", *args], ok_codes=ok_codes, timeout=timeout)


def load_json(path: Path) -> Any:
    require(path.is_file(), f"missing JSON artifact: {path.relative_to(REPO)}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GateFailure(f"invalid JSON in {path}: {exc}") from None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL artifact: {path.relative_to(REPO)}")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def tree_digest(*paths: Path, extra: str = "") -> str:
    h = hashlib.sha256(extra.encode())
    for root in paths:
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for p in files:
            if "__pycache__" in p.parts:
                continue
            h.update(p.relative_to(REPO).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def wilson_upper(k: int, n: int, z: float = 1.959963984540054) -> float:
    if n == 0:
        return 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre + half) / denom


def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


# ===================================================================== core gates
def gate_loc() -> str:
    from knotwork.devtools.kernel_audit import EXECUTABLE_LOC_CAP, TOTAL_LOC_CAP, loc_report

    rep = loc_report()
    table = ", ".join(f"{k}={v}" for k, v in sorted(rep.per_file.items()))
    require(
        rep.executable <= EXECUTABLE_LOC_CAP,
        f"executable LOC {rep.executable} > {EXECUTABLE_LOC_CAP}: {table}",
    )
    require(rep.total <= TOTAL_LOC_CAP, f"total LOC {rep.total} > {TOTAL_LOC_CAP}: {table}")
    return f"{rep.summary()}; {table}"


def gate_ast_isolation() -> str:
    from knotwork.devtools.kernel_audit import scan_forbidden_imports

    v = scan_forbidden_imports()
    require(not v, "; ".join(f"{x.file}:{x.line} {x.module} ({x.reason})" for x in v))
    return "no forbidden imports in knotcore (AST scan)"


def gate_kernel_placeholders() -> str:
    from knotwork.devtools.kernel_audit import KNOTCORE_DIR, scan_placeholders

    hits = scan_placeholders(KNOTCORE_DIR)
    require(not hits, "; ".join(f"{f}:{n}" for f, n, _ in hits))
    return "no placeholder markers in knotcore"


def gate_lint_imports() -> str:
    cp = run(tool("lint-imports"))
    m = re.search(r"Contracts: (\d+) kept, (\d+) broken", cp.stdout)
    require(bool(m) and m is not None and m.group(2) == "0", _tail(cp.stdout))
    return f"import-linter: {m.group(0) if m else '?'}"


def gate_ruff() -> str:
    run([*tool("ruff"), "check", "."])
    run([*tool("ruff"), "format", "--check", "."])
    return "ruff check + ruff format --check clean"


def gate_mypy() -> str:
    run([*tool("mypy")], timeout=1800)
    return "mypy clean (strict on knotwork.knotcore)"


def gate_required_files(phases: list[int]) -> Callable[[], str]:
    def _gate() -> str:
        missing = [f for ph in phases for f in PHASE_FILES[ph] if not (REPO / f).exists()]
        missing += [d for ph in phases for d in PHASE_TESTS[ph] if not (REPO / d).is_dir()]
        require(not missing, "missing: " + ", ".join(missing))
        return f"{sum(len(PHASE_FILES[p]) + len(PHASE_TESTS[p]) for p in phases)} required paths present"

    return _gate


def gate_pytest(phases: list[int]) -> Callable[[], str]:
    def _gate() -> str:
        dirs = [d for ph in sorted(set(range(0, max(phases) + 1))) for d in PHASE_TESTS[ph]]
        require(all((REPO / d).is_dir() for d in dirs), f"missing test dirs among {dirs}")
        cp = run([*tool("pytest"), "-q", "-p", "no:cacheprovider", *dirs], timeout=5400)
        summary = _tail(cp.stdout, 1)
        require("failed" not in summary and "error" not in summary, summary)
        return f"pytest {' '.join(dirs)} -> {summary}"

    return _gate


# ===================================================================== phase 0
def gate_font() -> str:
    import yaml

    prof = yaml.safe_load((REPO / "configs/default_profile.yaml").read_text(encoding="utf-8"))
    want = prof["render"]["font_sha256"]
    got = hashlib.sha256(FONT.read_bytes()).hexdigest()
    require(got == want, f"font sha256 {got} != profile {want}")
    return "bundled font sha256 matches profile"


def gate_seed_determinism() -> str:
    a = knot("selftest", "--seeds-only").stdout
    b = knot("selftest", "--seeds-only").stdout
    require(
        a == b and json.loads(a)["ok"] is True, "seed known-answer vector differs across processes"
    )
    return "seed KAT identical across two fresh processes and equals pinned constants"


def gate_cli_surface() -> str:
    out = knot("--help").stdout
    cmds = [
        "generate",
        "verify",
        "mutate",
        "probe",
        "diversity",
        "export",
        "bench",
        "selftest",
        "report",
        "new-plugin",
    ]
    missing = [c for c in cmds if c not in out]
    require(not missing, f"CLI missing subcommands: {missing}")
    return "all 10 subcommands registered"


def gate_uv_lock() -> str:
    run(["uv", "lock", "--check"], timeout=600)
    return "uv.lock consistent with pyproject.toml"


# ===================================================================== phase 1
def gate_kernel_sha_parity() -> str:
    from knotwork.devtools.kernel_audit import kernel_sha256
    from knotwork.knotcore import engine  # type: ignore[attr-defined]

    got = engine.kernel_sha256()
    require(got == kernel_sha256(), "knotcore.engine.kernel_sha256() != devtools kernel_sha256()")
    return f"kernel sha256 {got[:16]}... parity ok"


# ===================================================================== phase 2
def _dataset(plugin: str) -> Path:
    return ART / "phase2" / f"data_{plugin}"


def gate_generate() -> str:
    notes = []
    for plugin, n in GEN_N.items():
        out = _dataset(plugin)
        shutil.rmtree(out, ignore_errors=True)
        knot(
            "generate",
            "--plugin",
            plugin,
            "--n",
            str(n),
            "--seed",
            str(GEN_SEED),
            "--out",
            str(out),
            "--workers",
            "0",
        )
        stats = load_json(out / "stats.json")
        index = load_jsonl(out / "index.jsonl")
        bases = [r for r in index if r["role"] == "base"]
        require(len(bases) == n, f"{plugin}: {len(bases)} base items != requested {n}")
        require(
            stats["false_quarantine_rate"] <= MAX_FALSE_QUARANTINE,
            f"{plugin}: false_quarantine_rate {stats['false_quarantine_rate']:.3f} > {MAX_FALSE_QUARANTINE}",
        )
        fams = {r["family"] for r in bases}
        require(
            set(stats["families_expected"]) <= fams,
            f"{plugin}: families missing {set(stats['families_expected']) - fams}",
        )
        for r in index[:5]:
            require(
                (out / r["dir"] / "task.json").is_file(),
                f"{plugin}: bundle dir missing for {r['bundle_id']}",
            )
        require(
            (out / "freshness_receipt.json").is_file(), f"{plugin}: freshness_receipt.json missing"
        )
        notes.append(f"{plugin}: {n} base, fq={stats['false_quarantine_rate']:.3f}")
    return "; ".join(notes)


def gate_determinism() -> str:
    a, b = ART / "phase2" / "det_a", ART / "phase2" / "det_b"
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)
        knot(
            "generate",
            "--plugin",
            "tangle",
            "--n",
            "6",
            "--seed",
            "3",
            "--out",
            str(d),
            "--workers",
            "0",
        )
    ia = [r["bundle_id"] for r in load_jsonl(a / "index.jsonl")]
    ib = [r["bundle_id"] for r in load_jsonl(b / "index.jsonl")]
    require(ia == ib, "same seed produced different bundle ids")
    c = ART / "phase2" / "det_c"
    shutil.rmtree(c, ignore_errors=True)
    knot(
        "generate",
        "--plugin",
        "tangle",
        "--n",
        "6",
        "--seed",
        "3",
        "--out",
        str(c),
        "--workers",
        "2",
    )
    ic = [r["bundle_id"] for r in load_jsonl(c / "index.jsonl")]
    require(ia == ic, "worker count changed bundle ids (parallelism must not affect output)")
    d = ART / "phase2" / "det_d"
    shutil.rmtree(d, ignore_errors=True)
    knot(
        "generate",
        "--plugin",
        "tangle",
        "--n",
        "6",
        "--seed",
        "4",
        "--out",
        str(d),
        "--workers",
        "0",
    )
    idd = [r["bundle_id"] for r in load_jsonl(d / "index.jsonl")]
    require(not set(ia) & set(idd), "different seeds produced overlapping bundle ids")
    return f"{len(ia)} bundle ids identical across runs and worker counts; disjoint for new seed"


def gate_twins() -> str:
    checked = 0
    for plugin in GEN_N:
        index = load_jsonl(_dataset(plugin) / "index.jsonl")
        by_id = {r["bundle_id"]: r for r in index}
        for base in (r for r in index if r["role"] == "base"):
            task = load_json(_dataset(plugin) / base["dir"] / "task.json")
            tw = task["twins"]
            require(
                tw["causal_id"] in by_id and tw["nuisance_id"] in by_id,
                f"{base['bundle_id']}: twin missing",
            )
            causal = load_json(_dataset(plugin) / by_id[tw["causal_id"]]["dir"] / "task.json")
            nuis = load_json(_dataset(plugin) / by_id[tw["nuisance_id"]]["dir"] / "task.json")
            frac = causal["twins"]["pixel_diff_frac"]
            require(
                frac is not None and frac < CAUSAL_MAX_PIXEL_FRAC,
                f"{base['bundle_id']}: causal diff {frac}",
            )
            require(
                causal["answer"]["canonical"] != task["answer"]["canonical"],
                f"{base['bundle_id']}: causal twin answer unchanged",
            )
            require(
                nuis["answer"]["canonical"] == task["answer"]["canonical"],
                f"{base['bundle_id']}: nuisance twin changed answer",
            )
            require(
                causal["twins"]["base_id"] == nuis["twins"]["base_id"] == base["bundle_id"],
                "twin base_id mismatch",
            )
            checked += 1
    return f"{checked} base items have valid causal (<{CAUSAL_MAX_PIXEL_FRAC:.0%} px, flipped) + nuisance (preserved) twins"


def gate_gaslight() -> str:
    bases = [r for r in load_jsonl(_dataset("panel") / "index.jsonl") if r["role"] == "base"]
    n_gas = sum(r["family"] == "T_GASLIGHT" for r in bases)
    n_ctl = sum(r["family"] == "P5_PREMISE_CONTROL" for r in bases)
    want = round_half_up(GASLIGHT_FRAC * len(bases))
    require(n_gas == want, f"T_GASLIGHT count {n_gas} != round_half_up(0.15*{len(bases)})={want}")
    require(n_ctl == want, f"P5_PREMISE_CONTROL count {n_ctl} != {want} (true-premise controls)")
    for r in bases:
        if r["family"] == "T_GASLIGHT":
            t = load_json(_dataset("panel") / r["dir"] / "task.json")
            require(t["question"]["premise"]["holds"] is False, "gaslight premise must be false")
            require(
                t["answer"]["value"]["premise_holds"] is False,
                "gaslight answer must reject premise",
            )
    return f"panel: {n_gas} gaslight + {n_ctl} true-premise controls of {len(bases)}"


def gate_mutate() -> str:
    out = ART / "phase2" / "mutate_report.json"
    knot("mutate", "--fault", "all", "--n", str(MUTATE_N), "--seed", "0", "--out", str(out))
    rep = load_json(out)
    from knotwork.knotcore.contracts import FAULT_EXPECTED

    bad = []
    for fault, code in FAULT_EXPECTED.items():
        row = rep["faults"].get(fault.value)
        if row is None:
            bad.append(f"{fault.value}: not run")
            continue
        if row["expected"] != code.value:
            bad.append(f"{fault.value}: expected field {row['expected']} != {code.value}")
        if not (row["injected"] == MUTATE_N and row["caught_exact"] == row["injected"]):
            bad.append(
                f"{fault.value}: caught_exact {row['caught_exact']}/{row['injected']} codes={row['codes_seen']}"
            )
    require(not bad, "; ".join(bad))
    require(rep["exact_catch_rate"] == 1.0, f"exact_catch_rate {rep['exact_catch_rate']}")
    fq = rep["clean"]["false_quarantine_rate"]
    require(fq <= MAX_FALSE_QUARANTINE, f"clean false-quarantine {fq} > {MAX_FALSE_QUARANTINE}")
    return f"M1-M10: 100% caught with exact code ({MUTATE_N} each); clean false-quarantine {fq:.3f}"


CONTACT_SHEET_MIN = (2048, 1536)
WITNESS_PANELS = ["beauty", "id_buffer", "kernel_overlay"]
GAP_CHECK_ONE_LINER = (
    "echo ok > .checkpoints/GAPCHECK_APPROVED && export OPENROUTER_API_KEY=sk-or-... && "
    "uv run python scripts/gap_check.py --data artifacts/phase2 --n 20 --slot vlm_b --budget-usd 0.50"
)


def gate_contact_sheet() -> str:
    from PIL import Image

    out = ART / "phase2" / "contact_sheet.png"
    run(
        [
            PY,
            "-m",
            "knotwork.pipeline.contact_sheet",
            "--data",
            str(ART / "phase2"),
            "--out",
            str(out),
        ]
    )
    require(out.is_file(), "contact sheet not written")
    w, h = Image.open(out).size
    require(
        w >= CONTACT_SHEET_MIN[0] and h >= CONTACT_SHEET_MIN[1], f"contact sheet too small: {w}x{h}"
    )
    meta = load_json(out.with_suffix(".json"))
    rows = {r["name"]: r for r in meta["rows"]}
    for plugin in GEN_N:
        require(plugin in rows, f"contact sheet missing row for {plugin}")
        kinds = [p["kind"] for p in rows[plugin]["panels"]]
        require(
            kinds[:3] == ["base", "causal", "nuisance"],
            f"{plugin} row must start base, causal, nuisance",
        )
    wit = [r for r in meta["rows"] if r["name"].startswith("pixel_witness:")]
    require(
        wit, "contact sheet missing the Pixel-Witness Proof strip (rows 'pixel_witness:<plugin>')"
    )
    for r in wit:
        kinds = [p["kind"] for p in r["panels"]]
        require(
            kinds == WITNESS_PANELS, f"{r['name']}: panels must be {WITNESS_PANELS}, got {kinds}"
        )
    covered = {r["name"].split(":", 1)[1] for r in wit}
    require(
        {"tangle", "panel"} <= covered,
        f"pixel-witness strips needed for tangle and panel, got {covered}",
    )
    faults = {f["fault"] for f in meta["faults"]}
    require(
        faults == {f"M{i}" for i in range(1, 11)},
        f"contact sheet must caption M1-M10, got {sorted(faults)}",
    )
    marker = ART / "phase2" / "CHECKPOINT_1_READY.md"
    marker.write_text(
        "# [HUMAN-CHECKPOINT-1]\n\n"
        "Open `artifacts/phase2/contact_sheet.png` and check:\n"
        "* Tangle, Gauge and Panel rows: base / causal twin / nuisance twin (crossing halos unambiguous,\n"
        "  labels crisp, needles readable, twins plausible).\n"
        "* Pixel-Witness Proof strips: beauty | id-buffer colormap | kernel overlay (BFS-traced path,\n"
        "  crossing halos, PCA needle axis). The overlay must sit exactly on the rendered ink.\n"
        "* Fault row: M1-M10 examples, each captioned with the E_* code the kernel raised.\n\n"
        "Optional live VLM gap check, max $0.50, 20 items, one model (only YOU run this; Claude never does):\n\n"
        f"    {GAP_CHECK_ONE_LINER}\n\n"
        "It writes `artifacts/phase2/gap_check.json` (accuracy by family). Near-100 % accuracy means the\n"
        "tasks may be too easy; near-chance on simple families means check the renders.\n\n"
        "Approve:  `echo ok > .checkpoints/CP1_APPROVED` (one line of notes is welcome).\n"
        "Reject:   write the problems into `BLOCKERS.md` and tell Claude to fix them.\n",
        encoding="utf-8",
    )
    return f"contact sheet {w}x{h}, {len(wit)} pixel-witness strips, 10 faults; wrote {marker.relative_to(REPO)}"


# ===================================================================== phase 3
def gate_cp1() -> str:
    require(
        (CKPT / "CP1_APPROVED").is_file(),
        "[HUMAN-CHECKPOINT-1] not approved: .checkpoints/CP1_APPROVED missing",
    )
    return "CP1 approved by human"


def gate_freshness() -> str:
    from knotwork.knotcore.contracts import FreshnessReceipt

    rec = FreshnessReceipt.model_validate(load_json(_dataset("tangle") / "freshness_receipt.json"))
    require(rec.source == "offline", f"offline run must have source=offline, got {rec.source}")
    for r in load_jsonl(_dataset("tangle") / "index.jsonl")[:10]:
        t = load_json(_dataset("tangle") / r["dir"] / "task.json")
        require(t["canary"] == rec.canary, "bundle canary != receipt canary")
        require(t["freshness_receipt_id"] == rec.receipt_id, "bundle receipt id mismatch")
    return f"receipt {rec.receipt_id[:12]}... valid; canary propagated"


def bench_split() -> Path:
    root = ART / "phase3" / "bench_split"
    for plugin in GEN_N:
        out = root / plugin
        gen_sources = [
            SRC / "knotwork" / d for d in ("knotcore", "plugins", "pipeline", "render", "freshness")
        ]
        digest = tree_digest(
            *gen_sources,
            REPO / "configs" / "default_profile.yaml",
            extra=f"{plugin}:{BENCH_SPLIT_N}:split=test:seed=101",
        )
        stamp = out / ".gate_digest"
        if stamp.is_file() and stamp.read_text(encoding="utf-8") == digest:
            continue
        shutil.rmtree(out, ignore_errors=True)
        knot(
            "generate",
            "--plugin",
            plugin,
            "--n",
            str(BENCH_SPLIT_N),
            "--seed",
            "101",
            "--split",
            "test",
            "--no-twins",
            "--workers",
            "0",
            "--out",
            str(out),
            timeout=14400,
        )
        stamp.write_text(digest, encoding="utf-8")
    return root


def gate_probes() -> str:
    root = bench_split()
    out = ART / "phase3" / "probe_report.json"
    knot("probe", "--data", str(root), "--out", str(out), "--seed", "0", "--folds", "5")
    rep = load_json(out)
    bad, notes = [], []
    for plugin, names in REQUIRED_PROBES.items():
        prow = rep["plugins"][plugin]
        chance = prow["pooled"]["chance"]
        for name in names:
            pr = prow["pooled"]["probes"].get(name)
            if pr is None:
                bad.append(f"{plugin}/{name}: missing")
                continue
            wu = wilson_upper(round(pr["acc"] * pr["n"]), pr["n"])
            if wu > chance + PROBE_MARGIN:
                bad.append(
                    f"{plugin}/{name}: wilson_upper {wu:.3f} > chance {chance:.3f}+{PROBE_MARGIN}"
                )
        for fam, frow in prow["families"].items():
            c = frow["chance"]
            for name, pr in frow["probes"].items():
                # per-family slices are small: allow the larger of the fixed margin and 1.96 SE
                margin = max(PROBE_MARGIN, 1.96 * math.sqrt(c * (1.0 - c) / max(1, pr["n"])))
                if pr["acc"] > c + margin:
                    bad.append(
                        f"{plugin}/{fam}/{name}: acc {pr['acc']:.3f} > chance {c:.3f}+{margin:.3f}"
                    )
        notes.append(f"{plugin}: chance {chance:.3f}")
    tests = rep["balance"]
    alpha = BALANCE_ALPHA / max(1, len(tests))
    bad += [
        f"balance {k}: p={v['p']:.4g} < {alpha:.4g}" for k, v in tests.items() if v["p"] < alpha
    ]
    bad += [
        f"dedup {k}: {v['dup_rate']:.4f}"
        for k, v in rep["dedup"].items()
        if v["dup_rate"] > MAX_DUP_RATE
    ]
    require(not bad, "; ".join(bad))
    return (
        "all shortcut probes <= chance+5% (Wilson-95 upper, pooled); balance and dedup ok; "
        + ", ".join(notes)
    )


def gate_diversity() -> str:
    root = bench_split()
    out = ART / "phase3" / "diversity_report.json"
    knot("diversity", "--data", str(root), "--out", str(out), "--n", "256", "--seed", "0")
    rep = load_json(out)
    bad, notes = [], []
    for plugin in GEN_N:
        row = rep["plugins"][plugin]
        ratio = row["vendi_ours"] / row["vendi_recolor"]
        notes.append(f"{plugin} {row['vendi_ours']:.1f}/{row['vendi_recolor']:.1f}={ratio:.2f}x")
        if ratio < VENDI_MIN_RATIO:
            bad.append(f"{plugin}: vendi ratio {ratio:.2f} < {VENDI_MIN_RATIO}")
    require(not bad, "; ".join(bad))
    return "Vendi ours/recolor: " + ", ".join(notes)


def gate_selftest() -> str:
    knot("selftest")
    rep = load_json(ART / "selftest" / "report.json")
    require(rep["ok"] is True, f"selftest report not ok: {rep}")
    return f"golden selftest ok ({len(rep.get('checks', []))} checks)"


def gate_verify_and_tamper() -> str:
    data = ART / "selftest" / "data"
    knot("verify", str(data), "--out", str(ART / "phase3" / "verify_clean.json"))
    clean = load_json(ART / "phase3" / "verify_clean.json")
    require(clean["ok"] is True and clean["n_ok"] == clean["n_bundles"] > 0, "clean verify failed")
    index = load_jsonl(data / "index.jsonl")
    src = data / index[0]["dir"]
    with tempfile.TemporaryDirectory() as tmp:
        for kind in ("pixel", "answer"):
            ds = Path(tmp) / kind
            dst = ds / index[0]["dir"]
            shutil.copytree(src, dst)
            (ds / "index.jsonl").write_text(json.dumps(index[0]) + "\n", encoding="utf-8")
            if kind == "pixel":
                import numpy as np
                from PIL import Image

                arr = np.array(Image.open(dst / "id_buffer.png")).astype(np.uint16)
                arr[arr.shape[0] // 2, arr.shape[1] // 2] ^= 1
                Image.fromarray(arr).save(dst / "id_buffer.png")
            else:
                t = json.loads((dst / "task.json").read_text(encoding="utf-8"))
                t["answer"]["canonical"] = t["answer"]["canonical"] + "X"
                (dst / "task.json").write_text(json.dumps(t), encoding="utf-8")
            rep_path = Path(tmp) / f"{kind}.json"
            knot("verify", str(ds), "--out", str(rep_path), ok_codes=(1,))
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            codes = {c for f in rep["failures"] for c in f["codes"]}
            require(
                rep["ok"] is False and "E_DIGEST_MISMATCH" in codes,
                f"{kind} tamper not detected: {codes}",
            )
    return f"verify: {clean['n_ok']}/{clean['n_bundles']} clean; pixel + answer tampering detected (E_DIGEST_MISMATCH)"


# ===================================================================== phase 4
def _wrong(answer: dict[str, Any]) -> Any:
    t, v, tol = answer["type"], answer["value"], answer.get("tolerance", 0.0)
    if t == "label":
        return "WRONG-" + str(v)
    if t == "int":
        return int(v) + 1
    if t == "float":
        return float(v) + 10 * tol + 7.0
    if t == "zone":
        return {"green": "red", "amber": "green", "red": "green"}[v]
    if t == "premise":
        return {"premise_holds": not v["premise_holds"], "value": v["value"]}
    raise GateFailure(f"unknown answer type {t}")


def gate_export() -> str:
    data = ART / "selftest" / "data"
    out = ART / "phase4" / "export"
    shutil.rmtree(out, ignore_errors=True)
    knot("export", "--data", str(data), "--format", "all", "--out", str(out))
    tasks = sorted(p for p in (out / "harbor").iterdir() if p.is_dir())
    require(tasks, "no Harbor tasks exported")
    bash = shutil.which("bash")
    checked = 0
    for task in tasks[:HARBOR_ORACLE_SAMPLE]:
        for rel in (
            "instruction.md",
            "task.toml",
            "environment/Dockerfile",
            "environment/image.png",
            "solution/solve.sh",
            "solution/answer.json",
            "tests/test.sh",
            "tests/verify.py",
            "tests/expected.json",
        ):
            require((task / rel).is_file(), f"{task.name}: missing {rel}")
        import tomllib

        toml = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
        require(
            toml.get("schema_version") == "1.4" and "verifier" in toml and "agent" in toml,
            f"{task.name}: bad task.toml",
        )
        expected = json.loads((task / "tests" / "expected.json").read_text(encoding="utf-8"))
        for mode in ("oracle", "wrong", "missing"):
            with tempfile.TemporaryDirectory() as tmp:
                app, logs = Path(tmp) / "app", Path(tmp) / "logs"
                app.mkdir()
                logs.mkdir()
                env = {
                    "KNOT_APP_DIR": str(app),
                    "KNOT_LOG_DIR": str(logs),
                    "KNOT_TESTS_DIR": str(task / "tests"),
                    "KNOT_SOLUTION_DIR": str(task / "solution"),
                }
                if mode == "oracle":
                    if bash:
                        run([bash, str(task / "solution" / "solve.sh")], env=env)
                    else:
                        shutil.copy(task / "solution" / "answer.json", app / "answer.json")
                elif mode == "wrong":
                    (app / "answer.json").write_text(
                        json.dumps({"answer": _wrong(expected)}), encoding="utf-8"
                    )
                cmd = (
                    [bash, str(task / "tests" / "test.sh")]
                    if bash
                    else [PY, str(task / "tests" / "verify.py")]
                )
                run(cmd, env=env)
                reward = json.loads((logs / "reward.json").read_text(encoding="utf-8"))["reward"]
                txt = float((logs / "reward.txt").read_text(encoding="utf-8").strip())
                want = 1.0 if mode == "oracle" else 0.0
                require(
                    reward == txt == want, f"{task.name} {mode}: reward {reward}/{txt} != {want}"
                )
        checked += 1
    return f"Harbor: {len(tasks)} tasks exported; oracle=1.0, wrong=0.0, missing=0.0 on {checked}"


def gate_hf_rlvr() -> str:
    import pyarrow.parquet as pq

    from knotwork.knotcore.grading import grade_prediction  # type: ignore[import-not-found]

    out = ART / "phase4" / "export"
    files = sorted((out / "hf" / "data").glob("*.parquet"))
    require(files, "no parquet shards")
    cols = {
        "bundle_id",
        "plugin",
        "family",
        "split",
        "image",
        "question",
        "options",
        "answer",
        "answer_json",
        "tolerance",
        "program_json",
        "certificate_json",
        "canary",
        "twin_role",
    }
    rows = 0
    for f in files:
        t = pq.read_table(f)
        missing = cols - set(t.column_names)
        require(not missing, f"{f.name}: missing columns {missing}")
        rows += t.num_rows
    lines = load_jsonl(out / "rlvr" / "tasks.jsonl")
    require(len(lines) == rows, f"rlvr rows {len(lines)} != parquet rows {rows}")
    for line in lines:
        ans = line["answer"]
        require(grade_prediction(ans["canonical"], ans) == 1.0, f"{line['id']}: oracle != 1.0")
        require(grade_prediction(_wrong(ans), ans) == 0.0, f"{line['id']}: wrong != 0.0")
        require(len(line["steps"]) >= 1, f"{line['id']}: empty process trace")
    return (
        f"HF parquet {rows} rows / {len(files)} shards; RLVR grade_prediction oracle=1.0 wrong=0.0"
    )


def gate_scaffold() -> str:
    from knotwork.devtools.kernel_audit import kernel_sha256

    before = kernel_sha256()
    with tempfile.TemporaryDirectory() as tmp:
        knot("new-plugin", "gatecheck_dots", "--dest", tmp)
        pkg = Path(tmp) / "gatecheck_dots"
        for rel in ("__init__.py", "plugin.py", "README.md", "tests/test_plugin.py"):
            require((pkg / rel).is_file(), f"scaffold missing {rel}")
        for p in pkg.rglob("*.py"):
            py_compile.compile(str(p), doraise=True)
        env = {"PYTHONPATH": os.pathsep.join([tmp, str(SRC)])}
        run(
            [*tool("pytest"), "-q", "-p", "no:cacheprovider", "--rootdir", tmp, str(pkg / "tests")],
            env=env,
        )
    require(kernel_sha256() == before, "new-plugin modified knotcore")
    return "scaffolded plugin compiles, passes its tests, knotcore unchanged"


def gate_ci_yaml() -> str:
    import yaml

    doc = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    text = json.dumps(doc)
    for needle in (
        "uv sync",
        "ruff check",
        "lint-imports",
        "pytest",
        "knot selftest",
        "verify_phase.py",
    ):
        require(needle in text, f"ci.yml missing step containing `{needle}`")
    require("ubuntu" in text, "ci.yml must run on ubuntu")
    return "ci.yml has uv sync, ruff, lint-imports, pytest, selftest, verify_phase"


# ===================================================================== phase 5
def gate_bench_mock() -> str:
    import yaml

    root = (
        bench_split()
        if (ART / "phase3" / "bench_split").exists()
        else REPO / "tests/bench/fixtures/data"
    )
    out = ART / "phase5" / "bench_mock"
    shutil.rmtree(out, ignore_errors=True)
    knot("bench", "--data", str(root), "--out", str(out), "--client", "mock", "--seed", "5")
    rep = load_json(out / "bench_report.json")
    require(
        rep["mode"] == "mock" and rep["network_calls"] == 0 and rep["cost_usd"] == 0.0,
        "mock run not offline/free",
    )
    truths = {
        m["slot"]: m
        for m in yaml.safe_load((REPO / "configs/models.yaml").read_text(encoding="utf-8"))[
            "mock_slots"
        ]
    }
    bad = []
    for slot, row in rep["models"].items():
        p_true = truths[slot]["p_cross"]
        pc = row["per_crossing"]
        lo, hi = pc["ci"]
        if not abs(pc["p_hat"] - p_true) <= max(MOCK_P_ABS_TOL, (hi - lo) / 2):
            bad.append(f"{slot}: p_hat {pc['p_hat']:.3f} vs true {p_true}")
        if not lo - MOCK_CI_SLACK <= p_true <= hi + MOCK_CI_SLACK:
            bad.append(
                f"{slot}: true p {p_true} outside CI [{lo:.3f}, {hi:.3f}] +/- {MOCK_CI_SLACK}"
            )
        for track, st in row["staircase"].items():
            if abs(st["threshold_level"] - st["expected_level"]) > STAIRCASE_LEVEL_TOL:
                bad.append(
                    f"{slot}/{track}: staircase {st['threshold_level']:.2f} vs {st['expected_level']:.2f}"
                )
        lo, hi = row["composition"]["abbott"]["gap_ci"]
        if not lo <= 0.0 <= hi:
            bad.append(
                f"{slot}: independent mock but composition gap CI {lo:.3f}..{hi:.3f} excludes 0"
            )
        for key in ("color_ablation", "track_switch", "twins", "gaslight", "parse_fail_rate"):
            if key not in row:
                bad.append(f"{slot}: missing {key}")
    require(not bad, "; ".join(bad))
    return f"mock VLMs: p recovered, staircase converges to 70.7% level, product law holds ({len(rep['models'])} slots)"


def gate_bench_replay_and_preflight() -> str:
    out = ART / "phase5" / "bench_replay"
    shutil.rmtree(out, ignore_errors=True)
    knot(
        "bench",
        "--data",
        str(REPO / "tests/bench/fixtures/data"),
        "--out",
        str(out),
        "--client",
        "replay",
        "--replay-dir",
        str(REPO / "tests/bench/fixtures/replay"),
        "--seed",
        "0",
        "--max-trials",
        "10",
    )
    rep = load_json(out / "bench_report.json")
    require(rep["mode"] == "replay" and rep["network_calls"] == 0, "replay mode touched network")
    pre = ART / "phase5" / "preflight"
    shutil.rmtree(pre, ignore_errors=True)
    knot(
        "bench",
        "--data",
        str(REPO / "tests/bench/fixtures/data"),
        "--out",
        str(pre),
        "--live",
        "--budget-usd",
        "8",
        "--dry-run",
    )
    est = load_json(pre / "preflight.json")
    require(est["estimated_usd"] <= 8.0, f"preflight estimate {est['estimated_usd']} > cap 8.00")
    marker = ART / "phase5" / "CHECKPOINT_2_READY.md"
    marker.write_text(
        "# [HUMAN-CHECKPOINT-2]\n\n"
        f"Preflight estimate: **${est['estimated_usd']:.2f}** for {est['calls']} calls "
        f"(hard cap $8.00). Per slot: {json.dumps(est.get('per_slot', {}))}\n\n"
        "To RUN the live 3-VLM benchmark: confirm model slugs in configs/models.yaml, export "
        "OPENROUTER_API_KEY, create `.checkpoints/CP2_APPROVED`, then run\n"
        "`uv run knot bench --data artifacts/phase3/bench_split --out artifacts/phase5/bench_live --live --budget-usd 8`\n"
        "and commit `bench_replay/`.\n\nTo SKIP: create `.checkpoints/CP2_SKIPPED`.\n",
        encoding="utf-8",
    )
    return f"replay offline ok; preflight ${est['estimated_usd']:.2f} <= $8.00; wrote {marker.relative_to(REPO)}"


# ===================================================================== phase 6
def gate_cp2() -> str:
    approved, skipped = (CKPT / "CP2_APPROVED").is_file(), (CKPT / "CP2_SKIPPED").is_file()
    require(
        approved or skipped,
        "[HUMAN-CHECKPOINT-2] undecided: create .checkpoints/CP2_APPROVED or CP2_SKIPPED",
    )
    if approved:
        rep = load_json(ART / "phase5" / "bench_live" / "bench_report.json")
        require(
            rep["mode"] == "live" and rep["cost_usd"] <= 8.0,
            f"live report invalid: {rep.get('cost_usd')}",
        )
        require(
            any((REPO / "bench_replay").rglob("*.json")),
            "bench_replay/ is empty; commit the replay cache",
        )
        return f"CP2 approved; live cost ${rep['cost_usd']:.2f}; replay cache committed"
    return "CP2 skipped (offline-only submission; mock + replay results used)"


def _portal_fields() -> dict[str, str]:
    text = (REPO / "docs/PORTAL_SUBMISSION.md").read_text(encoding="utf-8")
    return {
        m.group(1): m.group(2).strip()
        for m in re.finditer(r"<!-- FIELD:(\w+) -->\n(.*?)\n<!-- /FIELD -->", text, flags=re.S)
    }


def gate_docs() -> str:
    deck = (REPO / "docs/DECK_COPY.md").read_text(encoding="utf-8")
    slides = re.findall(r"^## Slide (\d+)\b", deck, flags=re.M)
    lines = re.findall(r"^(## Slide \d+\b[^\n]*)", deck, flags=re.M)
    for i, (title, line) in enumerate(zip(OFFICIAL_SLIDES, lines, strict=False), start=1):
        require(
            title.lower() in line.lower(),
            f"Slide {i} heading must contain official title '{title}': {line!r}",
        )
    bodies = re.split(r"^## Slide \d+\b[^\n]*$", deck, flags=re.M)[1:]
    for num, phrase in TEMPLATE_PHRASES.items():
        require(
            len(bodies) >= num and phrase.lower() in bodies[num - 1].lower(),
            f"Slide {num} must quote the template phrase '{phrase}'",
        )
    require(
        slides == [str(i) for i in range(1, 12)],
        f"DECK_COPY.md must have '## Slide 1'..'## Slide 11', got {slides}",
    )
    fields = _portal_fields()
    for key in ("challenge", "prototype_brief", "technology_stack", "github_repo", "demo_video"):
        require(key in fields, f"PORTAL_SUBMISSION.md missing FIELD:{key}")
    require(
        fields["challenge"] == "Visual Task Factory",
        "challenge field must be exactly 'Visual Task Factory'",
    )
    for key in ("prototype_brief", "technology_stack"):
        n = len(fields[key])
        require(0 < n <= PORTAL_LIMIT, f"{key} is {n} chars (limit {PORTAL_LIMIT})")
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for needle in (
        "uv sync",
        "knot selftest",
        "knot generate",
        "knot export --format harbor",
        "--offline",
    ):
        require(needle in readme, f"README.md missing `{needle}`")
    ver = (REPO / "VERIFICATION.md").read_text(encoding="utf-8")
    for ph in range(6):
        require(
            f"phase{ph}.json" in ver, f"VERIFICATION.md must cite artifacts/gates/phase{ph}.json"
        )
        g = load_json(GATES_DIR / f"phase{ph}.json")
        require(g["ok"] is True, f"artifacts/gates/phase{ph}.json is not ok")
    card = (REPO / "DATA_CARD.md").read_text(encoding="utf-8")
    for needle in ("KNOTWORK-CANARY", "Intended use", "Limitations", "License"):
        require(needle in card, f"DATA_CARD.md missing `{needle}`")
    return (
        f"deck 11 slides; brief {len(fields['prototype_brief'])}c, stack {len(fields['technology_stack'])}c "
        "<= 1024; README/VERIFICATION/DATA_CARD complete"
    )


def gate_figures() -> str:
    from PIL import Image

    figs = sorted((REPO / "docs/figures").glob("fig*.png"))
    require(
        len(figs) >= MIN_FIGURES, f"need >= {MIN_FIGURES} docs/figures/fig*.png, found {len(figs)}"
    )
    small = [f.name for f in figs if Image.open(f).size[0] < MIN_FIGURE_WIDTH]
    require(not small, f"figures narrower than {MIN_FIGURE_WIDTH}px: {small}")
    return f"{len(figs)} publication figures >= {MIN_FIGURE_WIDTH}px wide"


def gate_report() -> str:
    out = ART / "report" / "report.html"
    knot("report", "--artifacts", str(ART), "--out", str(out))
    html = out.read_text(encoding="utf-8")
    require(
        out.stat().st_size > 50_000 and "Knotwork" in html and "<img" in html,
        "report.html too small or missing figures",
    )
    require(
        "http://" not in html.replace("http://www.w3.org", ""),
        "report.html must be self-contained (no http:// refs)",
    )
    return f"report.html {out.stat().st_size // 1024} KiB, self-contained"


def gate_all_targets() -> str:
    import importlib

    from knotwork.cli import DISPATCH_TARGETS
    from knotwork.devtools.kernel_audit import scan_placeholders

    missing = []
    for cmd, (target, _, _) in DISPATCH_TARGETS.items():
        mod, func = target.split(":")
        try:
            if not callable(getattr(importlib.import_module(mod), func, None)):
                missing.append(cmd)
        except ModuleNotFoundError:
            missing.append(cmd)
    require(not missing, f"CLI targets not implemented: {missing}")
    hits = scan_placeholders(SRC) + scan_placeholders(REPO / "apps")
    require(not hits, "placeholders remain: " + "; ".join(f"{f}:{n}" for f, n, _ in hits[:10]))
    return "all 10 CLI targets implemented; zero placeholder markers in src/ and apps/"


# ===================================================================== plan
CORE: list[tuple[str, Callable[[], str]]] = [
    ("core.loc", gate_loc),
    ("core.ast_isolation", gate_ast_isolation),
    ("core.kernel_placeholders", gate_kernel_placeholders),
    ("core.lint_imports", gate_lint_imports),
    ("core.ruff", gate_ruff),
    ("core.mypy", gate_mypy),
]

PHASES: dict[int, list[tuple[str, Callable[[], str]]]] = {
    0: [
        ("p0.font", gate_font),
        ("p0.cli", gate_cli_surface),
        ("p0.seeds", gate_seed_determinism),
        ("p0.uv_lock", gate_uv_lock),
    ],
    1: [("p1.kernel_sha", gate_kernel_sha_parity)],
    2: [
        ("p2.generate", gate_generate),
        ("p2.determinism", gate_determinism),
        ("p2.twins", gate_twins),
        ("p2.gaslight", gate_gaslight),
        ("p2.mutate", gate_mutate),
        ("p2.contact_sheet", gate_contact_sheet),
    ],
    3: [
        ("p3.cp1", gate_cp1),
        ("p3.freshness", gate_freshness),
        ("p3.selftest", gate_selftest),
        ("p3.verify", gate_verify_and_tamper),
        ("p3.probes", gate_probes),
        ("p3.diversity", gate_diversity),
    ],
    4: [
        ("p4.export_harbor", gate_export),
        ("p4.hf_rlvr", gate_hf_rlvr),
        ("p4.scaffold", gate_scaffold),
        ("p4.ci", gate_ci_yaml),
    ],
    5: [
        ("p5.bench_mock", gate_bench_mock),
        ("p5.replay_preflight", gate_bench_replay_and_preflight),
    ],
    6: [
        ("p6.cp2", gate_cp2),
        ("p6.targets", gate_all_targets),
        ("p6.figures", gate_figures),
        ("p6.report", gate_report),
        ("p6.docs", gate_docs),
    ],
}


def plan(phase: int, upto: bool) -> list[tuple[str, Callable[[], str]]]:
    phases = list(range(0, phase + 1)) if upto else [phase]
    steps: list[tuple[str, Callable[[], str]]] = [
        ("core.required_files", gate_required_files(list(range(0, phase + 1)))),
        *CORE,
        ("core.pytest", gate_pytest(phases)),
    ]
    for ph in phases:
        steps += PHASES[ph]
    return steps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--phase", type=int, choices=range(0, 7))
    ap.add_argument("--upto", action="store_true", help="also re-run gates of all earlier phases")
    ap.add_argument("--list", action="store_true", help="print gate plan and exit")
    ap.add_argument("--keep-going", action="store_true", help="run remaining gates after a failure")
    args = ap.parse_args(argv)
    if args.list:
        for ph in range(7):
            print(f"phase {ph}: " + ", ".join(n for n, _ in plan(ph, False)))
        return 0
    if args.phase is None:
        ap.error("--phase is required")
    steps = plan(args.phase, args.upto)
    results: list[Result] = []
    failed = False
    for name, fn in steps:
        if failed and not args.keep_going:
            results.append(Result(name, False, "skipped (earlier gate failed)", 0.0))
            continue
        t0 = time.perf_counter()
        try:
            detail, ok = fn(), True
        except GateFailure as exc:
            detail, ok = str(exc), False
        except Exception as exc:
            detail, ok = f"{type(exc).__name__}: {exc}", False
        dt = time.perf_counter() - t0
        failed = failed or not ok
        results.append(Result(name, ok, detail, round(dt, 2)))
        flag = "PASS" if ok else "FAIL"
        print(
            f"[{flag}] {name:<24} {dt:7.1f}s  {detail.splitlines()[0] if detail else ''}",
            flush=True,
        )
        if not ok and "\n" in detail:
            print("        " + detail.replace("\n", "\n        "), flush=True)
    GATES_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": args.phase,
        "upto": args.upto,
        "ok": not failed,
        "results": [asdict(r) for r in results],
    }
    (GATES_DIR / f"phase{args.phase}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        f"\nPHASE {args.phase} GATE: {'PASSED' if not failed else 'FAILED'}  "
        f"-> artifacts/gates/phase{args.phase}.json"
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
