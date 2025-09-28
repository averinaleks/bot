import importlib
import importlib.util
from pathlib import Path
import json
import logging
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from services.stubs import create_httpx_stub

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


def test_query_gpt_uses_builtin_client_when_http_client_stub(monkeypatch):
    stub_module = types.ModuleType("http_client")
    httpx_stub = create_httpx_stub()

    class _StubContext:
        def __enter__(self):
            self.client = httpx_stub.Client()
            return self.client

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_get_httpx_client(*args, **kwargs):
        return _StubContext()

    stub_module.httpx = httpx_stub
    stub_module.get_httpx_client = fake_get_httpx_client

    monkeypatch.setitem(sys.modules, "http_client", stub_module)
    monkeypatch.delitem(sys.modules, "gpt_client", raising=False)
    reloaded = importlib.reload(check_code)
    try:
        assert reloaded._fallback_reason is not None

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                if self.path.rstrip("/") != "/v1/completions":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if length:
                    self.rfile.read(length)
                payload = json.dumps({"choices": [{"text": "stub-ok"}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, fmt, *args):  # pragma: no cover - silence server logs
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            monkeypatch.setenv("GPT_OSS_API", f"http://127.0.0.1:{server.server_port}")
            assert reloaded.query_gpt("ping") == "stub-ok"
        finally:
            server.shutdown()
            thread.join()
    finally:
        importlib.reload(check_code)


def test_wait_for_api_http_error(monkeypatch):
    class DummyResponse:
        def __init__(self, status_code: int) -> None:
            self.closed = False
            self.status_code = status_code

        def close(self) -> None:
            self.closed = True

        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError(
                "err",
                request=httpx.Request("GET", "http://test"),
                response=httpx.Response(self.status_code),
            )

    responses = []

    class DummyClient:
        def __init__(self) -> None:
            self.response = DummyResponse(status_code=503)
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
        def __init__(self, should_raise: bool = False, status_code: int = 200) -> None:
            self.should_raise = should_raise
            self.closed = False
            self.status_code = status_code

        def close(self):
            self.closed = True
            called.append("closed")

        def raise_for_status(self):
            called.append("raise")
            if self.should_raise:
                raise httpx.HTTPStatusError(
                    "err",
                    request=httpx.Request("GET", "http://test"),
                    response=httpx.Response(self.status_code),
                )

    class GetClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            called.append(("get", url))
            return DummyResponse(should_raise=True, status_code=503)

    class PostClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json):
            called.append(("post", url, json))
            return DummyResponse()

    clients = [GetClient(), PostClient()]

    def fake_client(**_):
        return clients.pop(0)

    monkeypatch.setattr(check_code, "get_httpx_client", fake_client)
    monkeypatch.setattr(check_code.time, "time", lambda: 0)
    monkeypatch.setattr(check_code.time, "sleep", lambda s: None)

    check_code.wait_for_api("http://gptoss:8000/api", timeout=1)

    assert ("get", "http://gptoss:8000/api/v1/health") in called
    assert ("post", "http://gptoss:8000/api/v1/completions", {"prompt": "ping"}) in called
    assert called.count("closed") >= 2


def test_wait_for_api_accepts_client_errors(monkeypatch):
    monkeypatch.setenv("GPT_OSS_MODEL", "test-model")
    called = []

    class DummyResponse:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code
            self.closed = False

        def close(self):
            self.closed = True
            called.append("closed")

        def raise_for_status(self):
            called.append("raise")
            raise httpx.HTTPStatusError(
                "err",
                request=httpx.Request("POST", "http://test"),
                response=httpx.Response(self.status_code),
            )

    class GetClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            called.append(("get", url))
            return DummyResponse(status_code=503)

    class PostClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json):
            called.append(("post", url, json))
            return DummyResponse(status_code=400)

    clients = [GetClient(), PostClient()]

    def fake_client(**_):
        return clients.pop(0)

    monkeypatch.setattr(check_code, "get_httpx_client", fake_client)
    monkeypatch.setattr(check_code.time, "time", lambda: 0)
    monkeypatch.setattr(check_code.time, "sleep", lambda s: None)

    check_code.wait_for_api("http://gptoss:8000/api", timeout=1)

    assert ("post", "http://gptoss:8000/api/v1/completions", {"prompt": "ping", "model": "test-model"}) in called
    assert "raise" in called
    assert called.count("closed") >= 2


def test_skip_flag_accepts_inline_comment(tmp_path: Path) -> None:
    cfg = tmp_path / "gptoss_check.config"
    cfg.write_text("skip_gptoss_check=true # comment\n", encoding="utf-8")
    assert _load_skip_flag(cfg) is True


def test_fallback_simple_response_json(monkeypatch):
    module_path = Path("gptoss_check/check_code.py")
    spec = importlib.util.spec_from_file_location(
        "gptoss_check.check_code_fallback",
        module_path,
    )
    assert spec is not None and spec.loader is not None

    monkeypatch.setitem(sys.modules, "http_client", None)
    monkeypatch.setitem(sys.modules, "httpx", None)

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)

    spec.loader.exec_module(module)

    response = module._SimpleResponse(
        200,
        {"Content-Type": "application/json"},
        b"{\"ok\": true}",
    )

    assert response.json() == {"ok": True}


def test_fallback_client_accepts_private_hostname(monkeypatch):
    module_path = Path("gptoss_check/check_code.py")
    spec = importlib.util.spec_from_file_location(
        "gptoss_check.check_code_private_host",
        module_path,
    )
    assert spec is not None and spec.loader is not None

    monkeypatch.setitem(sys.modules, "http_client", None)
    monkeypatch.setitem(sys.modules, "httpx", None)

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)

    spec.loader.exec_module(module)

    calls: dict[str, tuple[str | None, object | None]] = {}

    def fake_getaddrinfo(host: str, port: object, *args, **kwargs):  # type: ignore[override]
        calls["args"] = (host, port)
        return [
            (
                module.socket.AF_INET,
                module.socket.SOCK_STREAM,
                0,
                "",
                ("172.18.0.5", 0),
            )
        ]

    monkeypatch.setattr(module.socket, "getaddrinfo", fake_getaddrinfo)

    client = module._SimpleClient()
    assert client._is_local_hostname("gptoss") is True
    assert calls["args"][0] == "gptoss"
