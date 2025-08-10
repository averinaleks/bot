import pytest
import requests

from bot.gpt_client import (
    GPTClientJSONError,
    GPTClientNetworkError,
    GPTClientResponseError,
    query_gpt,
)


class DummyResponse:
    def __init__(self, json_data=None, json_exc=None):
        self._json_data = json_data
        self._json_exc = json_exc

    def raise_for_status(self):
        pass

    def json(self):
        if self._json_exc:
            raise self._json_exc
        return self._json_data


def test_query_gpt_network_error(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.RequestException("boom")

    monkeypatch.setattr(requests, "post", fake_post)
    with pytest.raises(GPTClientNetworkError):
        query_gpt("hi")


def test_query_gpt_non_json(monkeypatch):
    def fake_post(*args, **kwargs):
        return DummyResponse(json_exc=ValueError("no json"))

    monkeypatch.setattr(requests, "post", fake_post)
    with pytest.raises(GPTClientJSONError):
        query_gpt("hi")


def test_query_gpt_missing_fields(monkeypatch):
    def fake_post(*args, **kwargs):
        return DummyResponse(json_data={"foo": "bar"})

    monkeypatch.setattr(requests, "post", fake_post)
    with pytest.raises(GPTClientResponseError):
        query_gpt("hi")
