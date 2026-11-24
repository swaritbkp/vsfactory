"""The autouse offline guard allows loopback/AF_UNIX sockets and blocks every external host."""

from __future__ import annotations

import asyncio
import socket
import sys

import pytest


def _blocked(exc: BaseException) -> bool:
    return type(exc).__name__ == "NetworkBlocked" and "forbidden" in str(exc)


def test_loopback_tcp_connect_is_allowed() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        for host in ("127.0.0.1", "localhost"):
            client = socket.create_connection((host, port), timeout=5)
            conn, _ = server.accept()
            client.sendall(b"ping")
            assert conn.recv(4) == b"ping"
            conn.close()
            client.close()
    finally:
        server.close()


def test_socketpair_and_asyncio_loop_work() -> None:
    a, b = socket.socketpair()
    a.sendall(b"x")
    assert b.recv(1) == b"x"
    a.close()
    b.close()

    async def _noop() -> int:
        await asyncio.sleep(0)
        return 7

    assert asyncio.run(_noop()) == 7


@pytest.mark.skipif(sys.platform == "win32", reason="AF_UNIX path sockets are POSIX-only here")
def test_af_unix_is_allowed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = str(tmp_path / "s.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(1)
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(path)
    client.close()
    server.close()


@pytest.mark.parametrize(
    "address", [("93.184.216.34", 80), ("8.8.8.8", 53), ("2606:4700::1111", 443)]
)
def test_external_connect_is_blocked(address: tuple[str, int]) -> None:
    family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
    try:
        s = socket.socket(family, socket.SOCK_STREAM)
    except OSError:
        pytest.skip("address family not available on this host")
    try:
        with pytest.raises(Exception) as exc:
            s.connect(address)
        assert _blocked(exc.value)
        with pytest.raises(Exception) as exc2:
            s.connect_ex(address)
        assert _blocked(exc2.value)
    finally:
        s.close()


def test_external_create_connection_and_dns_are_blocked() -> None:
    with pytest.raises(Exception) as exc:
        socket.create_connection(("93.184.216.34", 80), timeout=1)
    assert _blocked(exc.value)
    with pytest.raises(Exception) as exc2:
        socket.getaddrinfo("example.com", 443)
    assert _blocked(exc2.value)
