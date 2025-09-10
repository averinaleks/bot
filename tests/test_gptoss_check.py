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

import gptoss_check.check_code as check_code
from gptoss_check import main as gptoss_main


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


def test_wait_for_api_http_error(monkeypatch):
    class DummyResponse:
        def close(self):
            pass

        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "err", request=httpx.Request("GET", "http://test"), response=httpx.Response(404)
            )

    class DummyClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            return DummyResponse()

    calls = {"count": 0}

    def fake_time():
        calls["count"] += 1
        return 0 if calls["count"] == 1 else 100

    monkeypatch.setattr(check_code, "get_httpx_client", lambda **kw: DummyClient())
    monkeypatch.setattr(check_code.time, "time", fake_time)
    monkeypatch.setattr(check_code.time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError):
        check_code.wait_for_api("http://gptoss:8000", timeout=1)


def test_skip_flag_accepts_inline_comment(tmp_path: Path) -> None:
    cfg = tmp_path / "gptoss_check.config"
    cfg.write_text("skip_gptoss_check=true # comment\n", encoding="utf-8")
    assert _load_skip_flag(cfg) is True
