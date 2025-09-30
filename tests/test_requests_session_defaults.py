"""Tests for HTTP session helpers."""

from __future__ import annotations

import sys
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
