"""`knot new-plugin NAME --dest DIR` (Task 4.4): scaffold an untrusted TaskPlugin package.

Writes ``DIR/NAME/{__init__.py, plugin.py, README.md, tests/__init__.py, tests/test_plugin.py}`` from
the templates next to this module. Never touches knotcore (the gate checks the kernel sha256).
"""

from __future__ import annotations

import re
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parent / "templates"
NAME_RE = re.compile(r"[a-z][a-z0-9_]{2,30}")

INIT = '''"""{{name}} TaskPlugin package (scaffolded by `knot new-plugin`)."""

from __future__ import annotations

from .plugin import PLUGIN

__all__ = ["PLUGIN"]
'''
README = """# {{name}} -- Knotwork TaskPlugin

Scaffolded by `knot new-plugin {{name}}`. Untrusted plugin code: it samples, renders and claims;
the knotcore kernel re-derives every answer from the rendered pixels and certifies it.

* `plugin.py` -- `PLUGIN`, a `knotwork.knotcore.contracts.TaskPlugin` (8 methods).
* `tests/test_plugin.py` -- protocol conformance + kernel certification of clean scenes.

Next steps:
1. Replace the disc-counting scene with your own; answers must come from existing kernel ops.
2. Register it in `knotwork/plugins/registry.py` so `knot generate --plugin {{name}}` finds it.
3. Run `uv run pytest <this package>/tests -q` and `uv run knot generate --plugin {{name}} ...`.
"""


def _fill(text: str, name: str) -> str:
    return text.replace("{{name}}", name)


def scaffold(name: str, dest: Path) -> Path:
    if not NAME_RE.fullmatch(name):
        raise ValueError("plugin name must match [a-z][a-z0-9_]{2,30}")
    pkg = Path(dest) / name
    if pkg.exists():
        raise FileExistsError(f"{pkg} already exists")
    (pkg / "tests").mkdir(parents=True)
    files = {
        pkg / "__init__.py": INIT,
        pkg / "plugin.py": (TEMPLATES / "plugin.py.tmpl").read_text(encoding="utf-8"),
        pkg / "README.md": README,
        pkg / "tests" / "__init__.py": "",
        pkg / "tests" / "test_plugin.py": (TEMPLATES / "test_plugin.py.tmpl").read_text(
            encoding="utf-8"
        ),
    }
    for path, text in files.items():
        path.write_text(_fill(text, name), encoding="utf-8", newline="\n")
    return pkg


def run_new_plugin(name: str, dest: Path) -> int:
    try:
        pkg = scaffold(name, Path(dest))
    except (ValueError, FileExistsError) as exc:
        print(f"new-plugin: {exc}")
        return 2
    print(f"new-plugin: wrote {pkg}")
    return 0


__all__ = ["run_new_plugin", "scaffold"]
