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

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


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
