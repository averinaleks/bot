import asyncio
import socket
import logging
import json

import pytest
import httpx
from typing import Any

from bot.gpt_client import (
    GPTClientError,
    GPTClientJSONError,
    GPTClientNetworkError,
    GPTClientResponseError,
    MAX_PROMPT_BYTES,
    MAX_RESPONSE_BYTES,
    _get_api_url_timeout,
    _load_allowed_hosts,
    _validate_api_url,
    _parse_gpt_response,
    query_gpt,
    query_gpt_async,
    query_gpt_json_async,
)


class DummyStream:
    def __init__(self, content=b"content", headers=None):
        self.content = content
        self.headers = headers or {"Content-Type": "application/json"}

    def raise_for_status(self):
        pass

    def iter_bytes(self):
        yield self.content

    async def aiter_bytes(self):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


QUERIES = [
    (query_gpt, httpx.AsyncClient),
    (query_gpt_async, httpx.AsyncClient),
]


@pytest.fixture(autouse=True)
def allow_test_hosts(monkeypatch):
    monkeypatch.setenv(
        "GPT_OSS_ALLOWED_HOSTS",
        "localhost,127.0.0.1,::1",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    yield


def test_load_allowed_hosts_filters_public(monkeypatch, caplog):
    monkeypatch.setenv("GPT_OSS_ALLOWED_HOSTS", "malicious.example,8.8.8.8")

    def fake_getaddrinfo(host, *_args, **_kwargs):
        if host == "malicious.example":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0))]
        if host == "8.8.8.8":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0))]
        raise AssertionError(host)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with caplog.at_level(logging.WARNING):
        hosts = _load_allowed_hosts()

    assert "malicious.example" not in hosts
    assert "8.8.8.8" not in hosts
    assert "localhost" in hosts
    assert "non-private IPs" in caplog.text


async def run_query(func, prompt):
    if asyncio.iscoroutinefunction(func):
        return await func(prompt)
    return await asyncio.to_thread(func, prompt)


def test_parse_gpt_response_success():
    content = json.dumps({"choices": [{"text": "ok"}]}).encode()
    assert _parse_gpt_response(content) == "ok"


def test_parse_gpt_response_message_content():
    content = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
    assert _parse_gpt_response(content) == "ok"


def test_parse_gpt_response_message_content_list():
    content = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"text": "hello"},
                            {"text": {"value": " world"}},
                            {"value": "!"},
                        ]
                    }
                }
            ]
        }
    ).encode()
    assert _parse_gpt_response(content) == "hello world!"


def test_parse_gpt_response_structured_content_parts():
    content = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "type": "output_text",
                                "text": [
                                    {"type": "output_text", "text": "multi"},
                                    {
                                        "type": "output_text",
                                        "text": {"value": "part"},
                                    },
                                ],
                            },
                            {"type": "output_text", "text": {"value": " response"}},
                        ]
                    }
                }
            ]
        }
    ).encode()
    assert _parse_gpt_response(content) == "multipart response"


def test_parse_gpt_response_delta_content_parts():
    content = json.dumps(
        {
            "choices": [
                {
                    "delta": {
                        "content": [
                            {
                                "type": "output_text",
                                "text": [
                                    {"type": "system", "text": "ignored"},
                                    {
                                        "type": "output_text",
                                        "text": {"value": "streamed"},
                                    },
                                ],
                            }
                        ]
                    }
                }
            ]
        }
    ).encode()
    assert _parse_gpt_response(content) == "streamed"


def test_parse_gpt_response_invalid_json():
    with pytest.raises(GPTClientJSONError):
        _parse_gpt_response(b"not json")


def test_parse_gpt_response_missing_field():
    content = json.dumps({"foo": "bar"}).encode()
    with pytest.raises(GPTClientResponseError):
        _parse_gpt_response(content)

@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_network_error(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "https://127.0.0.1")

    def fake_stream(self, *args, **kwargs):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(client_cls, "stream", fake_stream)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))],
    )
    sleep_path = "asyncio.sleep" if asyncio.iscoroutinefunction(func) else "time.sleep"
    if asyncio.iscoroutinefunction(func):
        async def no_sleep(*args, **kwargs):  # pragma: no cover - no sleep
            pass
        monkeypatch.setattr(sleep_path, no_sleep)
    else:
        monkeypatch.setattr(sleep_path, lambda *a, **k: None)
    with pytest.raises(GPTClientNetworkError):
        await run_query(func, "hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_non_json(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "https://127.0.0.1")

    def fake_stream(self, *args, **kwargs):
        return DummyStream(content=b"not json")

    monkeypatch.setattr(client_cls, "stream", fake_stream)
    with pytest.raises(GPTClientJSONError):
        await run_query(func, "hi")


@pytest.mark.asyncio
async def test_ipv4_mapped_ipv6_resolution(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://127.0.0.1")

    call_state = {"count": 0}

    def fake_socket_getaddrinfo(*args, **kwargs):
        call_state["count"] += 1
        if call_state["count"] == 1:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]
        return [
            (
                socket.AF_INET6,
                socket.SOCK_STREAM,
                6,
                "",
                ("::ffff:127.0.0.1", 0, 0, 0),
            )
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_socket_getaddrinfo)

    loop = asyncio.get_running_loop()

    async def fake_loop_getaddrinfo(*args, **kwargs):
        return [
            (
                socket.AF_INET6,
                socket.SOCK_STREAM,
                6,
                "",
                ("::ffff:127.0.0.1", 0, 0, 0),
            )
        ]

    monkeypatch.setattr(loop, "getaddrinfo", fake_loop_getaddrinfo)

    def fake_stream(self, *args, **kwargs):
        content = json.dumps({"choices": [{"text": "ok"}]}).encode()
        return DummyStream(content=content)

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)

    result = await query_gpt_async("hi")
    assert result == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_bad_content_type(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "https://127.0.0.1")

    def fake_stream(self, *args, **kwargs):
        content = json.dumps({"choices": [{"text": "ok"}]}).encode()
        return DummyStream(content=content, headers={"Content-Type": "text/plain"})

    monkeypatch.setattr(client_cls, "stream", fake_stream)
    with pytest.raises(GPTClientResponseError):
        await run_query(func, "hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_missing_fields(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "https://127.0.0.1")

    def fake_stream(self, *args, **kwargs):
        return DummyStream(content=json.dumps({"foo": "bar"}).encode())

    monkeypatch.setattr(client_cls, "stream", fake_stream)
    with pytest.raises(GPTClientResponseError):
        await run_query(func, "hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_insecure_url(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "http://127.0.0.1")
    with pytest.raises(GPTClientError):
        await run_query(func, "hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_allow_insecure_url(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "http://127.0.0.1")
    monkeypatch.setenv("ALLOW_INSECURE_GPT_URL", "1")

    def fake_stream(self, *args, **kwargs):
        content = json.dumps({"choices": [{"text": "ok"}]}).encode()
        return DummyStream(content=content)

    monkeypatch.setattr(client_cls, "stream", fake_stream)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))
        ],
    )

    result = await run_query(func, "hi")
    assert result in {"ok", "offline response"}


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_insecure_url_dns_failure(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "http://127.0.0.1")
    monkeypatch.setenv("TEST_MODE", "1")

    def fail_resolution(*args, **kwargs):
        raise socket.gaierror("resolution failed")

    monkeypatch.setattr(socket, "getaddrinfo", fail_resolution)

    with pytest.raises(GPTClientError):
        await run_query(func, "hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_localhost_allowed(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "http://localhost")

    def fake_stream(self, *args, **kwargs):
        content = json.dumps({"choices": [{"text": "ok"}]}).encode()
        return DummyStream(content=content)

    monkeypatch.setattr(client_cls, "stream", fake_stream)
    assert await run_query(func, "hi") == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
@pytest.mark.parametrize("ip", [
    "127.0.0.1",
    "10.0.0.1",
    "172.16.0.1",
    "192.168.1.1",
])
async def test_private_ip_allowed(monkeypatch, func, client_cls, ip):
    monkeypatch.setenv("GPT_OSS_API", f"http://{ip}")

    def fake_stream(self, *args, **kwargs):
        content = json.dumps({"choices": [{"text": "ok"}]}).encode()
        return DummyStream(content=content)

    monkeypatch.setattr(client_cls, "stream", fake_stream)
    assert await run_query(func, "hi") == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_public_ip_blocked(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "http://8.8.8.8")
    with pytest.raises(GPTClientError):
        await run_query(func, "hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
@pytest.mark.parametrize("ip", ["::1", "fc00::1"])
async def test_private_ipv6_allowed(monkeypatch, func, client_cls, ip):
    monkeypatch.setenv("GPT_OSS_API", f"http://[{ip}]")

    def fake_stream(self, *args, **kwargs):
        content = json.dumps({"choices": [{"text": "ok"}]}).encode()
        return DummyStream(content=content)

    monkeypatch.setattr(client_cls, "stream", fake_stream)
    assert await run_query(func, "hi") == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
@pytest.mark.parametrize("ip", [
    "2001:4860:4860::8888",
    "2606:4700:4700::1111",
])
async def test_public_ipv6_blocked(monkeypatch, func, client_cls, ip):
    monkeypatch.setenv("GPT_OSS_API", f"http://[{ip}]")
    with pytest.raises(GPTClientError):
        await run_query(func, "hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
@pytest.mark.parametrize("url", ["http://::1", "http://[::1%eth0]"])
async def test_invalid_ipv6(monkeypatch, func, client_cls, url):
    monkeypatch.setenv("GPT_OSS_API", url)
    with pytest.raises(GPTClientError):
        await run_query(func, "hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_invalid_url(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "bad-url")
    with pytest.raises(GPTClientError, match="scheme"):
        await run_query(func, "hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_invalid_url_no_host(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "https://")
    with pytest.raises(GPTClientError, match="hostname"):
        await run_query(func, "hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_no_env(monkeypatch, func, client_cls):
    monkeypatch.delenv("GPT_OSS_API", raising=False)
    with pytest.raises(GPTClientNetworkError):
        await run_query(func, "hi")


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_retry_success(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "https://127.0.0.1")
    calls = {"count": 0}

    def fake_stream(self, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.HTTPError("boom")
        content = json.dumps({"choices": [{"text": "ok"}]}).encode()
        return DummyStream(content=content)

    monkeypatch.setattr(client_cls, "stream", fake_stream)
    sleep_path = "asyncio.sleep" if asyncio.iscoroutinefunction(func) else "time.sleep"
    if asyncio.iscoroutinefunction(func):
        async def no_sleep(*args, **kwargs):  # pragma: no cover - no sleep
            pass
        monkeypatch.setattr(sleep_path, no_sleep)
    else:
        monkeypatch.setattr(sleep_path, lambda *a, **k: None)
    assert await run_query(func, "hi") == "ok"
    assert calls["count"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_retry_success_dns_fallback(monkeypatch, func, client_cls):
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("GPT_OSS_API", "https://127.0.0.1")
    stream_calls = {"count": 0}

    def fake_stream(self, *args, **kwargs):
        stream_calls["count"] += 1
        content = json.dumps({"choices": [{"text": "ok"}]}).encode()
        return DummyStream(content=content)

    monkeypatch.setattr(client_cls, "stream", fake_stream)

    def flaky_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if flaky_getaddrinfo.calls == 0:
            flaky_getaddrinfo.calls += 1
            raise socket.gaierror("resolution failed")
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.10", 0))
        ]

    flaky_getaddrinfo.calls = 0

    async def async_getaddrinfo(self, host, port, family=0, type=0, proto=0, flags=0):
        return flaky_getaddrinfo(host, port, family, type, proto, flags)

    monkeypatch.setattr(socket, "getaddrinfo", flaky_getaddrinfo)
    monkeypatch.setattr(asyncio.AbstractEventLoop, "getaddrinfo", async_getaddrinfo)

    assert await run_query(func, "hi") == "ok"
    assert stream_calls["count"] >= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_retry_failure(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "https://127.0.0.1")
    calls = {"count": 0}

    def fake_stream(self, *args, **kwargs):
        calls["count"] += 1
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(client_cls, "stream", fake_stream)
    sleep_path = "asyncio.sleep" if asyncio.iscoroutinefunction(func) else "time.sleep"
    if asyncio.iscoroutinefunction(func):
        async def no_sleep(*args, **kwargs):  # pragma: no cover - no sleep
            pass
        monkeypatch.setattr(sleep_path, no_sleep)
    else:
        monkeypatch.setattr(sleep_path, lambda *a, **k: None)
    with pytest.raises(GPTClientNetworkError):
        await run_query(func, "hi")
    assert calls["count"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("func, _", QUERIES)
async def test_prompt_too_long(monkeypatch, func, _, caplog):
    monkeypatch.setenv("GPT_OSS_API", "https://127.0.0.1")
    sent = {}

    def fake_stream(self, *args, **kwargs):
        sent["bytes"] = len(kwargs["json"]["prompt"].encode("utf-8"))
        content = json.dumps({"choices": [{"text": "ok"}]}).encode()
        return DummyStream(content=content)

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)
    long_prompt = "я" * (MAX_PROMPT_BYTES // 2 + 1)
    with caplog.at_level(logging.WARNING):
        result = await run_query(func, long_prompt)
    assert result == "ok"
    assert sent["bytes"] == MAX_PROMPT_BYTES
    assert "Prompt exceeds maximum length" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("func, client_cls", QUERIES)
async def test_response_too_long(monkeypatch, func, client_cls):
    monkeypatch.setenv("GPT_OSS_API", "https://127.0.0.1")

    def fake_stream(self, *args, **kwargs):
        return DummyStream(content=b"x" * (MAX_RESPONSE_BYTES + 1))

    monkeypatch.setattr(client_cls, "stream", fake_stream)
    with pytest.raises(GPTClientError):
        await run_query(func, "hi")


def test_validate_api_url_multiple_dns_results_public_blocked(monkeypatch):
    def fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        assert host == "foo.local"
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(GPTClientError):
        _validate_api_url("http://foo.local", {"foo.local"})


def test_validate_api_url_rejects_userinfo():
    with pytest.raises(GPTClientError) as excinfo:
        _validate_api_url(
            "https://user:pass@127.0.0.1", _load_allowed_hosts()
        )
    assert "must not contain embedded credentials" in str(excinfo.value)


def test_validate_api_url_rejects_public_ip_literal(monkeypatch):
    public_ip = "93.184.216.34"

    def fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        assert host == public_ip
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (public_ip, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(GPTClientError) as excinfo:
        _validate_api_url(f"https://{public_ip}", _load_allowed_hosts())

    assert "private" in str(excinfo.value)


def test_validate_api_url_private_ip_literal_allowed(monkeypatch):
    def fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        assert host == "192.168.1.5"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.5", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    host, ips = _validate_api_url("https://192.168.1.5", _load_allowed_hosts())

    assert host == "192.168.1.5"
    assert ips == {"192.168.1.5"}


def test_validate_api_url_insecure_allowed_with_env(monkeypatch, caplog):
    def fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        assert host == "foo.local"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setenv("ALLOW_INSECURE_GPT_URL", "1")
    monkeypatch.setattr("bot.gpt_client.ALLOW_INSECURE_GPT_URL", True)
    with caplog.at_level(logging.WARNING):
        host, ips = _validate_api_url("http://foo.local", {"foo.local"})

    assert host == "foo.local"
    assert ips == {"8.8.8.8"}
    assert "Using insecure GPT_OSS_API URL" in caplog.text


def test_query_gpt_private_fqdn_allowed(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "http://foo.local")
    monkeypatch.setenv("GPT_OSS_ALLOWED_HOSTS", "foo.local")
    called = {"value": False}

    def fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        called["value"] = True
        assert host == "foo.local"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 0))]

    async def fake_async_getaddrinfo(
        self, host, port, family=0, type=0, proto=0, flags=0
    ):
        return fake_getaddrinfo(host, port, family, type, proto, flags)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(asyncio.AbstractEventLoop, "getaddrinfo", fake_async_getaddrinfo)

    def fake_stream(self, *args, **kwargs):
        content = json.dumps({"choices": [{"text": "ok"}]}).encode()
        return DummyStream(content=content)

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)
    assert query_gpt("hi") == "ok"
    assert called["value"]


def test_query_gpt_dns_error(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "http://foo.local")
    with pytest.raises(GPTClientError):
        query_gpt("hi")


@pytest.mark.asyncio
async def test_query_gpt_async_context(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://127.0.0.1")

    def fake_stream(self, *args, **kwargs):
        content = json.dumps({"choices": [{"text": "ok"}]}).encode()
        return DummyStream(content=content)

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)
    assert query_gpt("hi") == "ok"


def test_get_api_url_timeout_non_positive(monkeypatch, caplog):
    monkeypatch.setenv("GPT_OSS_API", "https://127.0.0.1")
    monkeypatch.setenv("GPT_OSS_TIMEOUT", "0")
    with caplog.at_level(logging.WARNING):
        _, timeout, _, _ = _get_api_url_timeout()
    assert timeout == 5.0
    assert "Non-positive GPT_OSS_TIMEOUT value" in caplog.text


def test_get_api_url_timeout_invalid(monkeypatch, caplog):
    monkeypatch.setenv("GPT_OSS_API", "https://127.0.0.1")
    monkeypatch.setenv("GPT_OSS_TIMEOUT", "abc")
    with caplog.at_level(logging.WARNING):
        _, timeout, _, _ = _get_api_url_timeout()
    assert timeout == 5.0
    assert "Invalid GPT_OSS_TIMEOUT value 'abc'; defaulting to 5.0" in caplog.text


def test_get_api_url_timeout_respects_base_path(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "http://localhost/api")
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))],
    )
    url, timeout, hostname, ips = _get_api_url_timeout()
    assert url == "http://localhost/api/v1/completions"
    assert hostname == "localhost"


@pytest.mark.asyncio
async def test_query_gpt_json_async(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://127.0.0.1")

    def fake_stream(self, *args, **kwargs):
        content = json.dumps({"choices": [{"text": '{"signal": "buy", "tp_mult": 1, "sl_mult": 1}'}]}).encode()
        return DummyStream(content=content)

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)
    result = await query_gpt_json_async("hi")
    assert result["signal"] == "buy"


@pytest.mark.asyncio
async def test_query_gpt_json_async_missing_fields(monkeypatch, caplog):
    monkeypatch.setenv("GPT_OSS_API", "https://127.0.0.1")

    def fake_stream(self, *args, **kwargs):
        content = json.dumps({"choices": [{"text": '{"signal": "buy"}'}]}).encode()
        return DummyStream(content=content)

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)
    with caplog.at_level(logging.ERROR):
        result = await query_gpt_json_async("hi")
    assert result == {"signal": "hold"}
    assert "Missing fields in GPT response" in caplog.text


@pytest.mark.asyncio
async def test_query_gpt_json_async_invalid_payload(monkeypatch, caplog):
    monkeypatch.setenv("GPT_OSS_API", "https://127.0.0.1")

    def fake_stream(self, *args, **kwargs):
        content = json.dumps({"choices": [{"text": "not json"}]}).encode()
        return DummyStream(content=content)

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)
    with caplog.at_level(logging.ERROR):
        result = await query_gpt_json_async("hi")
    assert result == {"signal": "hold"}
    assert "Invalid JSON from GPT OSS API" in caplog.text


def _install_dummy_openai(monkeypatch, recorder):
    class DummyResponse:
        def __init__(self, data):
            self._data = data

        def model_dump(self):
            return self._data

    class DummyCompletions:
        def __init__(self, record):
            self._record = record

        def create(self, **kwargs):
            self._record["request"] = kwargs
            return DummyResponse({"choices": [{"message": {"content": "ok"}}]})

    class DummyChat:
        def __init__(self, record):
            self.completions = DummyCompletions(record)

    class DummyOpenAI:
        def __init__(self, **kwargs):
            recorder["client_kwargs"] = kwargs
            self.chat = DummyChat(recorder)

    monkeypatch.setattr("bot.gpt_client.OpenAI", DummyOpenAI)


def test_query_gpt_openai_fallback(monkeypatch):
    monkeypatch.delenv("GPT_OSS_API", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    recorder: dict[str, Any] = {}
    _install_dummy_openai(monkeypatch, recorder)

    result = query_gpt("привет")

    assert result == "ok"
    assert recorder["client_kwargs"].get("timeout") == 5.0
    assert recorder["request"]["messages"][0]["content"] == "привет"


def test_query_gpt_openai_scheme(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "openai://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    recorder: dict[str, Any] = {}
    _install_dummy_openai(monkeypatch, recorder)

    result = query_gpt("test")

    assert result == "ok"
    assert recorder["client_kwargs"].get("base_url") == "https://api.openai.com/v1"


def test_query_gpt_openai_invalid_max_tokens(monkeypatch, caplog):
    monkeypatch.delenv("GPT_OSS_API", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MAX_TOKENS", "oops")
    recorder: dict[str, Any] = {}
    _install_dummy_openai(monkeypatch, recorder)

    with caplog.at_level(logging.WARNING):
        result = query_gpt("text")

    assert result == "ok"
    assert "Invalid OPENAI_MAX_TOKENS value 'oops'; ignoring" in caplog.text
    assert "max_tokens" not in recorder["request"]


@pytest.mark.asyncio
async def test_query_gpt_no_env_leak(monkeypatch):
    monkeypatch.setenv("GPT_OSS_API", "https://127.0.0.1")
    captured: dict[str, bool | None] = {}

    class DummyClient:
        def __init__(self, *a, **kw):
            captured["trust_env"] = kw.get("trust_env")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def stream(self, *args, **kwargs):
            content = json.dumps({"choices": [{"text": '{"signal": "buy", "tp_mult": 1, "sl_mult": 1}'}]}).encode()
            return DummyStream(content=content)

    monkeypatch.setattr(httpx, "AsyncClient", DummyClient)
    await query_gpt_json_async("hi")
    assert captured.get("trust_env") is False
