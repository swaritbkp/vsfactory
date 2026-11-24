"""Static audits of the trusted kernel: LOC budget, forbidden imports, placeholder markers.

Shared by tests/conformance/test_kernel_isolation.py and scripts/verify_phase.py so both layers
measure exactly the same thing.

LOC definition (SPEC.md section 2.2): a physical line counts iff it carries at least one Python
token other than COMMENT / NL / NEWLINE / INDENT / DEDENT / ENCODING / ENDMARKER and it is not part
of a module, class or function docstring.

Two caps are enforced (SPEC.md section 2.2):
* EXECUTABLE_LOC_CAP (850): every knotcore module except the declarative schema file contracts.py.
* TOTAL_LOC_CAP (1350): all knotcore modules including contracts.py.
"""

from __future__ import annotations

import ast
import hashlib
import io
import re
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
    "shapely",
    "streamlit",
    "sklearn",
    "matplotlib",
    "pandas",
    "pyarrow",
    "random",
)
_DYNAMIC_IMPORT_CALLS = {"__import__", "import_module"}
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
    return sorted(p for p in kernel_dir.rglob("*.py") if "__pycache__" not in p.parts)


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
        return self.executable <= EXECUTABLE_LOC_CAP and self.total <= TOTAL_LOC_CAP

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
            line = int(getattr(node, "lineno", 0))
            for name in names:
                hit = _hits(name, FORBIDDEN_INTERNAL) or _hits(name, FORBIDDEN_EXTERNAL)
                if hit:
                    out.append(ImportViolation(str(path), line, name, f"forbidden: {hit}"))
    return sorted(set(out), key=lambda v: (v.file, v.line, v.module))


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
