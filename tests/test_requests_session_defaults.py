"""Tests for HTTP session helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import http_client


class DummyResponse:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class DummySession:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.trust_env = True
        self.proxies = {"http": "proxy"}
        self.verify = True
        self._closed = False

    def request(self, method: str, url: str, **kwargs):
        self.requests.append({"method": method, "url": url, **kwargs})
        return DummyResponse()

    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)

    def close(self) -> None:
        self._closed = True


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_get_requests_session_defaults(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    session = DummySession()

    def factory() -> DummySession:
        return session

    fake_requests = SimpleNamespace(Session=factory)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    with http_client.get_requests_session(timeout=5.5) as wrapped:
        request = getattr(wrapped, method.lower())
        response = request("https://example.test", data=b"payload")
        assert isinstance(response, DummyResponse)

    assert session._closed is True
    assert session.proxies == {}
    assert session.trust_env is False
    assert len(session.requests) == 1
    recorded = session.requests[0]
    assert recorded["timeout"] == 5.5
    assert recorded["allow_redirects"] is False
    assert recorded["data"] == b"payload"


def test_get_requests_session_timeout_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    session = DummySession()

    def factory() -> DummySession:
        return session

    fake_requests = SimpleNamespace(Session=factory)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    with http_client.get_requests_session(timeout=0) as wrapped:
        wrapped.get("https://example.test")

    assert session.requests[0]["timeout"] == http_client.DEFAULT_TIMEOUT


def test_get_requests_session_rejects_verify_false(monkeypatch: pytest.MonkeyPatch) -> None:
    session = DummySession()

    def factory() -> DummySession:
        return session

    fake_requests = SimpleNamespace(Session=factory)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    with pytest.raises(ValueError):
        with http_client.get_requests_session(verify=False):
            pass


def test_get_requests_session_accepts_custom_ca(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = DummySession()

    def factory() -> DummySession:
        return session

    fake_requests = SimpleNamespace(Session=factory)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    ca_file = tmp_path / "custom-ca.pem"
    ca_file.write_text("dummy", encoding="utf-8")

    with http_client.get_requests_session(verify=ca_file) as wrapped:
        wrapped.get("https://example.test")

    assert session.verify == str(ca_file)

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("invalid", http_client.DEFAULT_TIMEOUT),
        (-5, http_client.DEFAULT_TIMEOUT),
        (0.05, http_client.DEFAULT_TIMEOUT),
        ("nan", http_client.DEFAULT_TIMEOUT),
        ("inf", http_client.DEFAULT_TIMEOUT),
        (2, 2.0),
    ],
)
def test_coerce_timeout_values(raw: float | str, expected: float) -> None:
    assert http_client._coerce_timeout(raw, fallback=http_client.DEFAULT_TIMEOUT) == expected
