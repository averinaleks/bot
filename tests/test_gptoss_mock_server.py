import threading
import time
from http.server import ThreadingHTTPServer

import httpx
import pytest

from scripts import gptoss_mock_server


@pytest.fixture()
def mock_gptoss_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), gptoss_mock_server._RequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    with httpx.Client(timeout=1) as client:
        for _ in range(20):
            try:
                client.get(f"http://127.0.0.1:{port}/v1/models").raise_for_status()
            except httpx.HTTPError:
                time.sleep(0.05)
            else:
                break
        else:
            server.shutdown()
            thread.join(timeout=1)
            server.server_close()
            pytest.fail("mock GPT-OSS server did not start")

    try:
        yield port
    finally:
        server.shutdown()
        thread.join(timeout=1)
        server.server_close()


def test_completions_endpoint_returns_text(mock_gptoss_server):
    port = mock_gptoss_server
    prompt = "print('debug')  # TODO"
    with httpx.Client(timeout=5) as client:
        response = client.post(
            f"http://127.0.0.1:{port}/v1/completions",
            json={"model": "test", "prompt": prompt},
        )
        response.raise_for_status()
        data = response.json()

    text = data["choices"][0]["text"]
    assert "Автоматический обзор" in text
    assert "отладочный print" in text
    assert "незавершённый комментарий" in text


def test_chat_completions_returns_message(mock_gptoss_server):
    port = mock_gptoss_server
    diff = "\n".join(
        [
            "diff --git a/test.py b/test.py",
            "+++ b/test.py",
            "@@",
            "+print('debug')",
            "+# FIXME: clean up",
        ]
    )
    with httpx.Client(timeout=5) as client:
        response = client.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": diff}]},
        )
        response.raise_for_status()
        data = response.json()

    message = data["choices"][0]["message"]["content"]
    assert "Автоматический обзор" in message
    assert message.count("В файле test.py") == 2


def test_signal_handlers_ignore_sighup_and_shutdown(monkeypatch):
    class DummyServer:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    registered: dict[int, object] = {}

    def fake_signal(sig: int, handler: object) -> None:
        registered[sig] = handler

    dummy_server = DummyServer()
    monkeypatch.setattr(gptoss_mock_server.signal, "signal", fake_signal)

    gptoss_mock_server._install_signal_handlers(dummy_server)

    sighup = getattr(gptoss_mock_server.signal, "SIGHUP", None)
    if sighup is not None:
        assert registered[sighup] is gptoss_mock_server.signal.SIG_IGN

    for sig_name in ("SIGTERM", "SIGINT"):
        sig = getattr(gptoss_mock_server.signal, sig_name, None)
        if sig is None:
            continue
        handler = registered[sig]
        handler(sig, None)
        assert dummy_server.shutdown_calls == 1
        dummy_server.shutdown_calls = 0
