"""Repo-wide pytest configuration.

External network is hard-disabled for every test unless it carries @pytest.mark.network
(deselected by default). Loopback (127.0.0.0/8, ::1, "localhost") and AF_UNIX sockets stay allowed,
because asyncio event loops on Windows/WSL and Streamlit's AppTest use local socket pairs. DNS
resolution of non-loopback names is refused too, so nothing can leak a lookup.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from shared_builds import shared_build  # noqa: E402

PROFILE_PATH = REPO_ROOT / "configs" / "default_profile.yaml"
#: more xdist workers than this only slows the OCR-bound tests down
MAX_AUTO_WORKERS = 12
#: test tiers by path prefix, first match wins; everything else is tier1
TIERS: tuple[tuple[str, str], ...] = (
    ("tests/pipeline/test_selftest.py", "tier2"),
    # exhaustive multi-scene and process-pool checks: nightly. `test_tier1_representative.py`
    # keeps one real path through each of these stages in tier1.
    ("tests/pipeline/test_contact_sheet.py", "tier2"),
    ("tests/pipeline/test_mutate.py", "tier2"),
    ("tests/pipeline/test_twins.py", "tier2"),
    ("tests/pipeline/test_determinism.py", "tier2"),
    ("tests/pipeline/test_generate.py", "tier2"),
    ("tests/pipeline/test_verify.py", "tier2"),
    ("tests/export/test_harbor_oracle.py", "tier2"),
    ("tests/export/test_hf_rlvr.py", "tier2"),
    ("tests/probes/", "tier2"),
    # 10-render yield gate + Forensics loop: nightly; `test_rack_patch_fast.py` stays in tier1.
    ("tests/arena/test_rack_patch.py", "tier2"),
    ("tests/bench/test_stats_budget.py", "tier0"),
    ("tests/bench/", "tier2"),
    ("tests/conformance/test_contracts.py", "tier0"),
    ("tests/conformance/test_seeds.py", "tier0"),
    ("tests/conformance/test_kernel_isolation.py", "tier0"),
    ("tests/conformance/test_offline_guard.py", "tier0"),
    ("tests/conformance/test_schemas_in_sync.py", "tier0"),
    ("tests/kernel/test_grading.py", "tier0"),
    ("tests/kernel/test_io.py", "tier0"),
    ("tests/kernel/test_hardening.py", "tier0"),
    ("tests/plugins/test_braid.py", "tier0"),
    ("tests/freshness/", "tier0"),
)


def tier_of(relpath: str) -> str:
    return next((tier for prefix, tier in TIERS if relpath.startswith(prefix)), "tier1")


def pytest_configure(config: pytest.Config) -> None:
    """One native thread per test process and no nested-pool oversubscription under xdist."""
    workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "0") or 0)
    if workers:
        share = max(1, (os.cpu_count() or 2) // workers)
        os.environ.setdefault("KNOTWORK_AUTO_WORKERS", str(share))
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_auto_num_workers(config: pytest.Config) -> int:
    return min(os.cpu_count() or 1, MAX_AUTO_WORKERS)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if any(item.get_closest_marker(t) for t in ("tier0", "tier1", "tier2")):
            continue
        rel = Path(item.path).resolve().relative_to(REPO_ROOT).as_posix()
        item.add_marker(getattr(pytest.mark, tier_of(rel)))


@pytest.fixture(scope="session")
def datasets(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """data_tangle (n=3), data_gauge (n=2), data_panel (n=6, every family), all with twins;
    rendered once per session across all workers. Read-only for tests."""

    def build(root: Path) -> None:
        from knotwork.pipeline.generate import run_generate

        for plugin, n in (("tangle", 3), ("gauge", 2), ("panel", 6)):
            out = root / f"data_{plugin}"
            rc = run_generate(plugin, n, out, 5, "offline", PROFILE_PATH, "dev", True, None, 0)
            assert rc == 0, plugin

    return shared_build(tmp_path_factory, "datasets", build)


@pytest.fixture(scope="session")
def mutate_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """`knot mutate --fault all --n 1` with the faulty bundles kept on disk, shared by workers."""

    def build(root: Path) -> None:
        from knotwork.pipeline.io import write_json
        from knotwork.pipeline.mutate import run_mutate

        rc = run_mutate("all", 1, 0, root / "report.json", PROFILE_PATH, keep_dir=root / "kept")
        write_json(root / "rc.json", {"rc": rc})

    root = shared_build(tmp_path_factory, "mutate", build)
    from knotwork.pipeline.io import read_json

    return {
        "rc": read_json(root / "rc.json")["rc"],
        "report": read_json(root / "report.json"),
        "path": root / "report.json",
    }


class NetworkBlocked(RuntimeError):
    pass


_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex
_REAL_GETADDRINFO = socket.getaddrinfo
_LOOPBACK_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}


def is_loopback_host(host: object) -> bool:
    if host is None:
        return True  # getaddrinfo(None, port) resolves to the local wildcard/loopback
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    if not isinstance(host, str):
        return False
    name = host.strip("[]").split("%", 1)[0].lower()
    if name in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def _address_allowed(sock: socket.socket, address: object) -> bool:
    if getattr(socket, "AF_UNIX", None) is not None and sock.family == socket.AF_UNIX:
        return True
    if isinstance(address, tuple) and address:
        return is_loopback_host(address[0])
    return False


def _guarded_connect(self: socket.socket, address: object) -> None:
    if not _address_allowed(self, address):
        raise NetworkBlocked(f"network access is forbidden in offline tests: {address!r}")
    _REAL_CONNECT(self, address)  # type: ignore[arg-type]


def _guarded_connect_ex(self: socket.socket, address: object) -> int:
    if not _address_allowed(self, address):
        raise NetworkBlocked(f"network access is forbidden in offline tests: {address!r}")
    return _REAL_CONNECT_EX(self, address)  # type: ignore[arg-type]


def _guarded_getaddrinfo(host: object, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
    if not is_loopback_host(host):
        raise NetworkBlocked(f"DNS lookup is forbidden in offline tests: {host!r}")
    return _REAL_GETADDRINFO(host, *args, **kwargs)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _offline_guard(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOTWORK_NETWORK", "deny")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    if request.node.get_closest_marker("network"):
        monkeypatch.setenv("KNOTWORK_NETWORK", "beacon")
        return
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", _guarded_getaddrinfo)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def in_repo(monkeypatch: pytest.MonkeyPatch) -> Path:
    """chdir to the repo root so default relative config paths resolve."""
    monkeypatch.chdir(REPO_ROOT)
    return REPO_ROOT


os.environ.setdefault("KNOTWORK_NETWORK", "deny")
