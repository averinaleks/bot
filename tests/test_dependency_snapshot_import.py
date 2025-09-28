from __future__ import annotations

import http.client
import importlib.util
import socket
import sys
from pathlib import Path

import pytest


def test_dependency_snapshot_import_succeeds_without_requests(monkeypatch) -> None:
    """Ensure the dependency snapshot script has no hard dependency on ``requests``."""

    monkeypatch.setitem(sys.modules, "requests", None)

    module_name = "dependency_snapshot_no_requests"
    spec = importlib.util.spec_from_file_location(
        module_name, Path("scripts/submit_dependency_snapshot.py")
    )
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)

    assert hasattr(module, "submit_dependency_snapshot")
    assert callable(module.submit_dependency_snapshot)


def test_submit_with_headers_retries_on_retryable_status(monkeypatch, capsys):
    from scripts import submit_dependency_snapshot as snapshot

    class DummyResponse:
        def __init__(self, status: int, reason: str, body: bytes = b"") -> None:
            self.status = status
            self.reason = reason
            self._body = body
            self.closed = False

        def read(self) -> bytes:
            return self._body

        def close(self) -> None:
            self.closed = True

    class DummyConnection:
        def __init__(self, response: DummyResponse) -> None:
            self._response = response
            self.closed = False

        def request(self, method: str, path: str, body: bytes, headers: dict[str, str]) -> None:
            assert method == "POST"
            assert path == "/graphql"

        def getresponse(self) -> DummyResponse:
            return self._response

        def close(self) -> None:
            self.closed = True

    connections = [
        DummyConnection(DummyResponse(503, "Service Unavailable", b"temporary")),
        DummyConnection(DummyResponse(201, "Created", b"")),
    ]

    def fake_https_connection(host: str, *, port: int, timeout: int, context: object) -> DummyConnection:
        assert host == "example.com"
        assert port == 443
        assert timeout == snapshot._HTTP_TIMEOUT
        assert context is snapshot._SSL_CONTEXT
        try:
            return connections.pop(0)
        except IndexError:
            pytest.fail("HTTPSConnection called more times than expected")

    monkeypatch.setattr(http.client, "HTTPSConnection", fake_https_connection)

    snapshot._submit_with_headers(
        "https://example.com/graphql", b"{}", {"User-Agent": "test-agent"}
    )

    captured = capsys.readouterr()
    assert "Retrying" in captured.err
    assert "Dependency snapshot submitted" in captured.out


def test_submit_with_headers_raises_on_network_error(monkeypatch):
    from scripts import submit_dependency_snapshot as snapshot

    class FailingConnection:
        def __init__(self) -> None:
            self.closed = False

        def request(self, *_: object, **__: object) -> None:
            raise socket.timeout("timed out")

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        http.client,
        "HTTPSConnection",
        lambda *args, **kwargs: FailingConnection(),
    )

    with pytest.raises(snapshot.DependencySubmissionError) as excinfo:
        snapshot._submit_with_headers("https://example.com/graphql", b"{}", {})

    assert excinfo.value.status_code is None
    assert "timed out" in str(excinfo.value)
