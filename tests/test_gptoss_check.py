from pathlib import Path
import logging
import sys
import types

import httpx
import pytest

from gptoss_check.main import _load_skip_flag

_fake_client = types.ModuleType("gpt_client")


class _DummyError(Exception):
    pass


_fake_client.GPTClientError = _DummyError
_fake_client.query_gpt = lambda *args, **kwargs: None

sys.modules.setdefault("gpt_client", _fake_client)

import gptoss_check.check_code as check_code  # noqa: E402
from gptoss_check import main as gptoss_main  # noqa: E402


def test_skip_message(caplog, tmp_path):
    cfg = tmp_path / "gptoss_check.config"
    cfg.write_text("skip_gptoss_check=true")
    with caplog.at_level(logging.INFO):
        gptoss_main.main(config_path=cfg)
    assert "skipped" in caplog.text.lower()


def test_run_message(caplog, tmp_path, monkeypatch):
    cfg = tmp_path / "gptoss_check.config"
    cfg.write_text("skip_gptoss_check=false")

    called = []

    def fake_run():
        called.append(True)

    monkeypatch.setattr(check_code, "run", fake_run)
    monkeypatch.setattr(check_code, "wait_for_api", lambda *args, **kwargs: None)
    monkeypatch.setenv("GPT_OSS_API", "http://gptoss:8000")
    with caplog.at_level(logging.INFO):
        gptoss_main.main(config_path=cfg)
    assert "Running GPT-OSS check" in caplog.text
    assert "GPT-OSS check completed" in caplog.text
    assert called


def test_missing_config_triggers_check_and_warns(caplog, tmp_path, monkeypatch):
    cfg = tmp_path / "gptoss_check.config"  # file deliberately not created

    called = []

    def fake_run():
        called.append(True)

    monkeypatch.setattr(check_code, "run", fake_run)
    monkeypatch.setattr(check_code, "wait_for_api", lambda *args, **kwargs: None)
    monkeypatch.setenv("GPT_OSS_API", "http://gptoss:8000")
    with caplog.at_level(logging.INFO):
        gptoss_main.main(config_path=cfg)
    assert "не найден" in caplog.text
    assert "Running GPT-OSS check" in caplog.text
    assert called


def test_run_without_api(monkeypatch, caplog):
    monkeypatch.delenv("GPT_OSS_API", raising=False)
    with caplog.at_level(logging.WARNING):
        check_code.run()
    assert "GPT_OSS_API" in caplog.text


def test_run_handles_unexpected_query_error(monkeypatch, caplog):
    monkeypatch.setenv("GPT_OSS_API", "http://gptoss:8000")
    monkeypatch.setenv("CHECK_CODE_PATH", "gptoss_check/main.py")
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setattr(check_code, "wait_for_api", lambda *args, **kwargs: None)

    telegram_messages: list[str] = []
    monkeypatch.setattr(check_code, "send_telegram", telegram_messages.append)

    request = httpx.Request("POST", "http://gptoss:8000/v1/completions")
    response = httpx.Response(500, request=request)

    def boom(prompt: str) -> str:
        raise httpx.HTTPStatusError("server error", request=request, response=response)

    monkeypatch.setattr(check_code, "query", boom)

    with caplog.at_level(logging.ERROR):
        check_code.run()

    assert "Непредвиденная ошибка GPT-OSS" in caplog.text
    assert telegram_messages
    assert "Непредвиденная ошибка GPT-OSS" in telegram_messages[0]


def test_wait_for_api_http_error(monkeypatch):
    class DummyResponse:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError(
                "err",
                request=httpx.Request("GET", "http://test"),
                response=httpx.Response(404),
            )

    responses = []

    class DummyClient:
        def __init__(self) -> None:
            self.response = DummyResponse()
            responses.append(self.response)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            return self.response

        def post(self, url, json):
            return self.response

    calls = {"count": 0}

    def fake_time():
        calls["count"] += 1
        return 0 if calls["count"] < 3 else 100

    monkeypatch.setattr(check_code, "get_httpx_client", lambda **kw: DummyClient())
    monkeypatch.setattr(check_code.time, "time", fake_time)
    monkeypatch.setattr(check_code.time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError):
        check_code.wait_for_api("http://gptoss:8000", timeout=1)
    assert all(response.closed for response in responses)


def test_wait_for_api_prefers_health_endpoint(monkeypatch):
    called = []

    class DummyResponse:
        def close(self):
            called.append("closed")

        def raise_for_status(self):
            called.append("raise")

    class DummyClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            called.append(("get", url))
            return DummyResponse()

        def post(self, url, json):
            raise AssertionError("POST should not be called when health succeeds")

    monkeypatch.setattr(check_code, "get_httpx_client", lambda **kw: DummyClient())
    monkeypatch.setattr(check_code.time, "time", lambda: 0)
    monkeypatch.setattr(check_code.time, "sleep", lambda s: None)

    check_code.wait_for_api("http://gptoss:8000/api/v1/chat/completions", timeout=1)

    assert ("get", "http://gptoss:8000/api/v1/health") in called
    assert "raise" in called
    assert "closed" in called
    assert not any(isinstance(item, tuple) and item[0] == "post" for item in called)


def test_wait_for_api_falls_back_to_completions_endpoint(monkeypatch):
    called = []

    class DummyResponse:
        def __init__(self, should_raise: bool = False) -> None:
            self.should_raise = should_raise
            self.closed = False

        def close(self):
            self.closed = True
            called.append("closed")

        def raise_for_status(self):
            called.append("raise")
            if self.should_raise:
                raise httpx.HTTPStatusError(
                    "err",
                    request=httpx.Request("GET", "http://test"),
                    response=httpx.Response(404),
                )

    class GetClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            called.append(("get", url))
            return DummyResponse(should_raise=True)

    class PostClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json):
            called.append(("post", url))
            return DummyResponse()

    clients = [GetClient(), PostClient()]

    def fake_client(**_):
        return clients.pop(0)

    monkeypatch.setattr(check_code, "get_httpx_client", fake_client)
    monkeypatch.setattr(check_code.time, "time", lambda: 0)
    monkeypatch.setattr(check_code.time, "sleep", lambda s: None)

    check_code.wait_for_api("http://gptoss:8000/api", timeout=1)

    assert ("get", "http://gptoss:8000/api/v1/health") in called
    assert ("post", "http://gptoss:8000/api/v1/completions") in called
    assert called.count("closed") >= 2


def test_skip_flag_accepts_inline_comment(tmp_path: Path) -> None:
    cfg = tmp_path / "gptoss_check.config"
    cfg.write_text("skip_gptoss_check=true # comment\n", encoding="utf-8")
    assert _load_skip_flag(cfg) is True
