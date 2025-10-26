from __future__ import annotations

import importlib.util
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


def test_submit_dependency_snapshot_skips_when_requests_missing(monkeypatch, capsys):
    """The script should report a friendly message when ``requests`` is unavailable."""

    from scripts import submit_dependency_snapshot as snapshot

    monkeypatch.setattr(snapshot, "requests", None, raising=False)
    monkeypatch.setattr(
        snapshot,
        "_REQUESTS_IMPORT_ERROR",
        ImportError("requests package is not installed"),
        raising=False,
    )

    snapshot.submit_dependency_snapshot()

    captured = capsys.readouterr()
    assert "requests" in captured.err
    assert "Skipping submission" in captured.err


def test_submit_with_headers_requires_requests(monkeypatch):
    """Submitting without ``requests`` should raise a descriptive error."""

    from scripts import submit_dependency_snapshot as snapshot

    monkeypatch.setattr(snapshot, "requests", None, raising=False)
    missing_error = ImportError("dependency missing")
    monkeypatch.setattr(
        snapshot,
        "_REQUESTS_IMPORT_ERROR",
        missing_error,
        raising=False,
    )

    with pytest.raises(snapshot.DependencySubmissionError) as excinfo:
        snapshot._submit_with_headers("https://example.com/api", b"{}", {})

    message = str(excinfo.value)
    assert "requests" in message
    assert excinfo.value.__cause__ is missing_error


def test_submit_with_headers_rejects_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import submit_dependency_snapshot as snapshot

    class DummyResponse:
        status_code = 302
        reason = "Found"
        headers = {"Location": "https://example.com/other"}
        content = b""

        def close(self) -> None:
            pass

    created_sessions: list["DummySession"] = []

    class DummySession:
        def __init__(self) -> None:
            self.trust_env = True
            self.proxies = {"http": "http://proxy"}
            self.verify = True
            self.captured_kwargs: dict[str, object] = {}

        def __enter__(self) -> "DummySession":
            created_sessions.append(self)
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def post(self, *_args: object, **kwargs: object) -> DummyResponse:
            self.captured_kwargs = dict(kwargs)
            return DummyResponse()

    monkeypatch.setattr(snapshot.requests, "Session", lambda: DummySession())

    with pytest.raises(snapshot.DependencySubmissionError) as excinfo:
        snapshot._submit_with_headers("https://example.com/api", b"{}", {})

    assert excinfo.value.status_code == 302
    assert created_sessions, "session should have been instantiated"
    session = created_sessions[-1]
    assert session.trust_env is False
    assert session.proxies == {}
    assert session.captured_kwargs.get("allow_redirects") is False


def test_submit_with_headers_uses_retry_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import submit_dependency_snapshot as snapshot

    class DummyResponse:
        def __init__(
            self,
            status_code: int,
            *,
            headers: dict[str, str] | None = None,
            reason: str = "",
        ) -> None:
            self.status_code = status_code
            self.headers = headers or {}
            self.reason = reason
            self.text = ""

        def close(self) -> None:
            return None

    responses = [
        DummyResponse(429, headers={"Retry-After": "5"}, reason="Too Many Requests"),
        DummyResponse(200, reason="OK"),
    ]
    calls: list[None] = []

    class DummySession:
        def __enter__(self) -> "DummySession":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def post(self, *_args: object, **_kwargs: object) -> DummyResponse:
            calls.append(None)
            return responses.pop(0)

    waits: list[float] = []

    def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(snapshot.requests, "Session", lambda: DummySession())
    monkeypatch.setattr(snapshot.time, "sleep", fake_sleep)

    snapshot._submit_with_headers("https://example.com/api", b"{}", {})

    assert len(calls) == 2
    assert waits == [pytest.approx(5.0)]


def test_submit_with_headers_falls_back_without_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import submit_dependency_snapshot as snapshot

    class DummyResponse:
        def __init__(self, status_code: int, *, reason: str = "") -> None:
            self.status_code = status_code
            self.headers: dict[str, str] = {}
            self.reason = reason
            self.text = ""

        def close(self) -> None:
            return None

    responses = [DummyResponse(429, reason="Too Many Requests"), DummyResponse(200, reason="OK")]
    calls: list[None] = []

    class DummySession:
        def __enter__(self) -> "DummySession":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def post(self, *_args: object, **_kwargs: object) -> DummyResponse:
            calls.append(None)
            return responses.pop(0)

    waits: list[float] = []

    def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(snapshot.requests, "Session", lambda: DummySession())
    monkeypatch.setattr(snapshot.time, "sleep", fake_sleep)

    snapshot._submit_with_headers("https://example.com/api", b"{}", {})

    assert len(calls) == 2
    assert waits == [pytest.approx(1.0)]
