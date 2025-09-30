import logging
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Mapping

import pytest

import scripts.health_check as hc


class DummyResponse:
    def __init__(
        self, status_code: int = 200, headers: Mapping[str, str] | None = None
    ) -> None:
        self.status_code = status_code
        self.headers = dict(headers or {})

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise hc.requests.HTTPError(f"HTTP {self.status_code}")


def _call_check(base_url: str, endpoints: Iterable[str]) -> int:
    return hc.check_endpoints(base_url, list(endpoints))


def test_normalise_base_url_allows_loopback() -> None:
    allowed = hc._load_allowed_hosts()
    assert (
        hc._normalise_base_url("http://127.0.0.1:8080/api", allowed)
        == "http://127.0.0.1:8080/api"
    )


def test_normalise_base_url_rejects_http_non_local() -> None:
    allowed = hc._load_allowed_hosts()
    with pytest.raises(ValueError):
        hc._normalise_base_url("http://example.com", allowed)


def test_normalise_base_url_respects_allowed_hosts_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEALTH_CHECK_ALLOWED_HOSTS", "example.com")
    allowed = hc._load_allowed_hosts()
    assert (
        hc._normalise_base_url("https://example.com:8443/base", allowed)
        == "https://example.com:8443/base"
    )


def test_normalise_base_url_rejects_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEALTH_CHECK_ALLOWED_HOSTS", "example.com")
    allowed = hc._load_allowed_hosts()
    with pytest.raises(ValueError):
        hc._normalise_base_url("https://example.com/path?bad=1", allowed)


def test_check_endpoints_sanitises_logs(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    class DummyError(hc.requests.RequestException):
        pass

    class DummySession:
        def get(self, url: str) -> DummyResponse:
            raise DummyError("boom\nfail")

        def close(self) -> None:
            pass

    @contextmanager
    def fake_session(*_args: Any, **_kwargs: Any) -> Iterator[DummySession]:
        yield DummySession()

    monkeypatch.setattr(hc, "get_requests_session", fake_session)
    caplog.set_level(logging.ERROR)
    result = _call_check("http://localhost:8000", ["/bad\nendpoint"])
    assert result == 1
    record = caplog.records[0]
    message = record.getMessage()
    assert "http://localhost:8000/bad\\nendpoint" in message
    assert "boom\\nfail" in message


def test_check_endpoints_rejects_redirect(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    class DummySession:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, url: str) -> DummyResponse:
            self.calls += 1
            return DummyResponse(302, {"Location": "https://example.com/next"})

        def close(self) -> None:
            pass

    @contextmanager
    def fake_session(*_args: Any, **_kwargs: Any) -> Iterator[DummySession]:
        yield DummySession()

    monkeypatch.setattr(hc, "get_requests_session", fake_session)
    caplog.set_level(logging.ERROR)

    result = _call_check("http://localhost:8000", ["/redirect"])

    assert result == 1
    assert "redirect" in caplog.text.lower()
