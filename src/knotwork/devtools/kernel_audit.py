"""Static audits of the trusted kernel: TCB invariants, LOC telemetry, placeholder markers.

Shared by tests/conformance/test_kernel_isolation.py and scripts/verify_phase.py so both layers
measure exactly the same thing.

LOC definition (SPEC.md section 2.2): a physical line counts iff it carries at least one Python
token other than COMMENT / NL / NEWLINE / INDENT / DEDENT / ENCODING / ENDMARKER and it is not part
of a module, class or function docstring.

The two LOC budgets (SPEC.md section 2.2) are informational telemetry (D-38): exceeding one
produces a warning, never a failure:
* EXECUTABLE_LOC_CAP (850): every knotcore module except the declarative schema file contracts.py.
* TOTAL_LOC_CAP (1350): all knotcore modules including contracts.py.

The trusted-computing-base contract IS enforced (`tcb_report`):
* imports are allow-listed: standard library, numpy, scipy, skimage, PIL, pydantic,
  rapidocr_onnxruntime and knotwork.knotcore itself;
* no module that opens sockets, spawns processes, loads native code or imports dynamically, and no
  eval / exec / compile calls;
* no access to `SceneIR.payload`, and no use of `SceneIR` outside the schema module;
* `kernel_sha256` is a pure function of file names and newline-normalised bytes.
"""

from __future__ import annotations

import ast
import hashlib
import io
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
KNOTCORE_DIR = REPO_ROOT / "src" / "knotwork" / "knotcore"
EXECUTABLE_LOC_CAP = 850
TOTAL_LOC_CAP = 1350
#: Declarative schema modules excluded from the executable cap (still counted in the total cap).
DECLARATIVE_MODULES = frozenset({"contracts.py"})

_SKIP_TOKENS = {
    tokenize.COMMENT,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENCODING,
    tokenize.ENDMARKER,
}

#: Knotwork packages the kernel may never import (it sits at the bottom of the layer stack).
FORBIDDEN_INTERNAL = (
    "knotwork.plugins",
    "knotwork.pipeline",
    "knotwork.freshness",
    "knotwork.probes",
    "knotwork.diversity",
    "knotwork.export",
    "knotwork.bench",
    "knotwork.report",
    "knotwork.scaffold",
    "knotwork.net",
    "knotwork.cli",
    "knotwork.devtools",
    "knotwork.apps",
)
#: Third-party / stdlib modules the kernel may never import (network, geometry truth, UI, ML).
FORBIDDEN_EXTERNAL = (
    "httpx",
    "requests",
    "urllib3",
    "urllib.request",
    "http.client",
    "socket",
    "socketserver",
    "ssl",
    "http",
    "urllib",
    "ftplib",
    "poplib",
    "imaplib",
    "smtplib",
    "xmlrpc",
    "webbrowser",
    "subprocess",
    "multiprocessing",
    "asyncio",
    "ctypes",
    "importlib",
    "shapely",
    "streamlit",
    "sklearn",
    "matplotlib",
    "pandas",
    "pyarrow",
    "random",
)
#: third-party top-level packages the kernel may import, besides the standard library and itself
ALLOWED_THIRD_PARTY = frozenset(
    {"numpy", "scipy", "skimage", "PIL", "pydantic", "rapidocr_onnxruntime"}
)
KERNEL_PACKAGE = "knotwork.knotcore"
_DYNAMIC_IMPORT_CALLS = {"__import__", "import_module"}
_DYNAMIC_CODE_CALLS = {"eval", "exec", "compile"}
# Built from fragments so this file does not flag itself when src/ is scanned.
_MARKERS = ("TO" + "DO", "FIX" + "ME", "X" + "XX", "HA" + "CK")
_PLACEHOLDER_RE = re.compile(r"\b(" + "|".join(_MARKERS) + r")\b|" + "NotImpl" + "ementedError")


@dataclass(frozen=True)
class ImportViolation:
    file: str
    line: int
    module: str
    reason: str


def _docstring_lines(tree: ast.AST) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                end = body[0].end_lineno or body[0].lineno
                lines.update(range(body[0].lineno, end + 1))
    return lines


def count_loc_source(source: str) -> int:
    tree = ast.parse(source)
    code_lines: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in _SKIP_TOKENS:
            continue
        code_lines.update(range(tok.start[0], tok.end[0] + 1))
    return len(code_lines - _docstring_lines(tree))


def count_loc_file(path: Path) -> int:
    return count_loc_source(path.read_text(encoding="utf-8"))


def kernel_files(kernel_dir: Path = KNOTCORE_DIR) -> list[Path]:
    found = (p for p in kernel_dir.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(found, key=lambda p: p.relative_to(kernel_dir).as_posix())


def kernel_loc(kernel_dir: Path = KNOTCORE_DIR) -> dict[str, int]:
    return {
        p.relative_to(kernel_dir).as_posix(): count_loc_file(p) for p in kernel_files(kernel_dir)
    }


@dataclass(frozen=True)
class LocReport:
    per_file: dict[str, int]
    executable: int
    total: int

    @property
    def ok(self) -> bool:
        """Within both budgets. Informational only (D-38)."""
        return not self.warnings()

    def warnings(self) -> list[str]:
        out = []
        if self.executable > EXECUTABLE_LOC_CAP:
            out.append(
                f"knotcore executable LOC {self.executable} exceeds the "
                f"{EXECUTABLE_LOC_CAP} review budget"
            )
        if self.total > TOTAL_LOC_CAP:
            out.append(f"knotcore total LOC {self.total} exceeds the {TOTAL_LOC_CAP} review budget")
        return out

    def summary(self) -> str:
        return (
            f"knotcore executable LOC {self.executable}/{EXECUTABLE_LOC_CAP} (excl. contracts.py), "
            f"total LOC {self.total}/{TOTAL_LOC_CAP}"
        )


def loc_report(kernel_dir: Path = KNOTCORE_DIR) -> LocReport:
    per = kernel_loc(kernel_dir)
    executable = sum(v for k, v in per.items() if k not in DECLARATIVE_MODULES)
    return LocReport(per_file=per, executable=executable, total=sum(per.values()))


def _module_name(path: Path, src_root: Path) -> str:
    rel = path.relative_to(src_root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative(module: str | None, level: int, current: str, is_pkg: bool) -> str:
    base = current.split(".")
    if not is_pkg:
        base = base[:-1]
    if level > 1:
        base = base[: len(base) - (level - 1)]
    return ".".join([*base, module] if module else base)


def _hits(name: str, prefixes: tuple[str, ...]) -> str | None:
    for p in prefixes:
        if name == p or name.startswith(p + "."):
            return p
    return None


def _disallowed(base: str) -> str | None:
    """Reason when `base` is outside the TCB import allow-list."""
    root = base.split(".")[0]
    if root == "knotwork":
        inside = base == KERNEL_PACKAGE or base.startswith(KERNEL_PACKAGE + ".")
        return None if inside else f"not allowed: {base}"
    if root in sys.stdlib_module_names or root in ALLOWED_THIRD_PARTY:
        return None
    return f"not allowed: {root}"


def scan_forbidden_imports(
    kernel_dir: Path = KNOTCORE_DIR, src_root: Path | None = None
) -> list[ImportViolation]:
    src_root = src_root or kernel_dir.parents[1]
    out: list[ImportViolation] = []
    for path in kernel_files(kernel_dir):
        current = _module_name(path, src_root)
        is_pkg = path.name == "__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = (
                    _resolve_relative(node.module, node.level, current, is_pkg)
                    if node.level
                    else (node.module or "")
                )
                names = [base] + [f"{base}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Call):
                fn = node.func
                fname = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
                if fname in _DYNAMIC_IMPORT_CALLS:
                    out.append(
                        ImportViolation(str(path), node.lineno, fname, "dynamic import forbidden")
                    )
                if isinstance(fn, ast.Name) and fname in _DYNAMIC_CODE_CALLS:
                    out.append(
                        ImportViolation(str(path), node.lineno, fname, "dynamic code forbidden")
                    )
            line = int(getattr(node, "lineno", 0))
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                bases = names if isinstance(node, ast.Import) else names[:1]
                for base in bases:
                    reason = _disallowed(base)
                    if reason and not (
                        _hits(base, FORBIDDEN_INTERNAL) or _hits(base, FORBIDDEN_EXTERNAL)
                    ):
                        out.append(ImportViolation(str(path), line, base, reason))
            for name in names:
                hit = _hits(name, FORBIDDEN_INTERNAL) or _hits(name, FORBIDDEN_EXTERNAL)
                if hit:
                    out.append(ImportViolation(str(path), line, name, f"forbidden: {hit}"))
    return sorted(set(out), key=lambda v: (v.file, v.line, v.module))


def _names_scene_ir(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        return any(a.name == "SceneIR" for a in node.names)
    name = node.id if isinstance(node, ast.Name) else getattr(node, "attr", "")
    return isinstance(node, (ast.Name, ast.Attribute)) and name == "SceneIR"


def scan_payload_access(kernel_dir: Path = KNOTCORE_DIR) -> list[ImportViolation]:
    """Any read of a `payload` attribute or key anywhere in the kernel, and any use of `SceneIR`
    outside the declarative schema module (which only declares the type)."""
    out: list[ImportViolation] = []
    for path in kernel_files(kernel_dir):
        schema = path.name in DECLARATIVE_MODULES
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            what = ""
            if isinstance(node, ast.Attribute) and node.attr == "payload":
                what = ".payload"
            elif isinstance(node, ast.Subscript):
                key = node.slice
                if isinstance(key, ast.Constant) and key.value == "payload":
                    what = "['payload']"
            elif isinstance(node, ast.Call):
                fn = node.func
                fname = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
                if fname in ("getattr", "get", "attrgetter", "itemgetter") and any(
                    isinstance(a, ast.Constant) and a.value == "payload" for a in node.args
                ):
                    what = f"{fname}('payload')"
            if not schema and not what and _names_scene_ir(node):
                what = "SceneIR"
            if what:
                out.append(
                    ImportViolation(
                        str(path), int(getattr(node, "lineno", 0)), what, "scene payload access"
                    )
                )
    return sorted(set(out), key=lambda v: (v.file, v.line, v.module))


@dataclass(frozen=True)
class TcbReport:
    violations: list[ImportViolation]
    kernel_sha256: str

    @property
    def ok(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        if self.ok:
            return (
                "TCB ok: allow-listed imports, no network or dynamic code, no payload access; "
                f"kernel_sha256 {self.kernel_sha256[:16]}"
            )
        return "; ".join(f"{v.file}:{v.line} {v.module} ({v.reason})" for v in self.violations)


def tcb_report(kernel_dir: Path = KNOTCORE_DIR, src_root: Path | None = None) -> TcbReport:
    found = scan_forbidden_imports(kernel_dir, src_root) + scan_payload_access(kernel_dir)
    first, second = kernel_sha256(kernel_dir), kernel_sha256(kernel_dir)
    if first != second:
        found.append(ImportViolation(str(kernel_dir), 0, "kernel_sha256", "not deterministic"))
    return TcbReport(violations=found, kernel_sha256=first)


def scan_placeholders(root: Path) -> list[tuple[str, int, str]]:
    """Lines under `root` that contain a banned placeholder marker (see _MARKERS) or the
    standard not-implemented exception name. The marker strings are assembled from fragments
    above so that this module never matches its own scan."""
    found: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _PLACEHOLDER_RE.search(line):
                found.append((str(path), i, line.strip()))
    return found


def kernel_sha256(kernel_dir: Path = KNOTCORE_DIR) -> str:
    """Digest of the kernel source; recorded in every Certificate.kernel_sha256."""
    h = hashlib.sha256()
    for path in kernel_files(kernel_dir):
        h.update(path.relative_to(kernel_dir).as_posix().encode())
        h.update(b"\0")
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))
        h.update(b"\0")
    return h.hexdigest()
