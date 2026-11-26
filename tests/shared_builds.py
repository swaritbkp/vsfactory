"""Cross-worker cached render fixtures for the test suite.

Under pytest-xdist every worker is its own process, so a plain session fixture would render the same
scenes once per worker. `shared_build` builds a directory once per test session: the first worker to
take the lock builds it, the others wait for the done marker and reuse the files (read-only).
"""

from __future__ import annotations

import os
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import pytest

WAIT_S = 600.0
POLL_S = 0.1


def shared_build(factory: pytest.TempPathFactory, name: str, build: Callable[[Path], None]) -> Path:
    """Directory `name`, built exactly once per session by `build(directory)`."""
    if "PYTEST_XDIST_WORKER" not in os.environ:
        root = factory.mktemp(name)
        build(root)
        return root
    base = factory.getbasetemp().parent  # shared by all workers of this session
    root, lock = base / f"shared_{name}", base / f"shared_{name}.lock"
    done, failed = base / f"shared_{name}.done", base / f"shared_{name}.failed"
    deadline = time.monotonic() + WAIT_S
    while True:
        if done.exists():
            return root
        if failed.exists():
            pytest.fail(f"shared fixture {name} failed:\n{failed.read_text('utf-8')}")
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except (FileExistsError, PermissionError):
            if time.monotonic() > deadline:
                pytest.fail(f"timed out waiting for shared fixture {name}")
            time.sleep(POLL_S)
            continue
        try:
            root.mkdir(exist_ok=True)
            build(root)
        except BaseException:
            failed.write_text(traceback.format_exc(), encoding="utf-8")
            raise
        done.write_text("ok", encoding="utf-8")
        return root
