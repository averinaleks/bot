"""Tests for offline environment helpers."""

from __future__ import annotations

import os

import pytest

from services import offline


@pytest.fixture(autouse=True)
def restore_env():
    """Ensure environment variables created during a test are removed afterwards."""

    snapshot = dict(os.environ)
    yield
    for key in set(os.environ) - set(snapshot):
        del os.environ[key]
    for key, value in snapshot.items():
        os.environ[key] = value


def test_ensure_offline_env_supports_callable(monkeypatch):
    monkeypatch.setattr(offline, "OFFLINE_MODE", True)
    monkeypatch.delenv("OFFLINE_TEST_TOKEN", raising=False)

    generated: list[str] = []

    def _factory() -> str:
        value = offline.generate_placeholder_credential("unit-test")
        generated.append(value)
        return value

    applied = offline.ensure_offline_env({"OFFLINE_TEST_TOKEN": _factory})

    assert applied == ["OFFLINE_TEST_TOKEN"]
    assert os.environ["OFFLINE_TEST_TOKEN"] == generated[0]

    second = offline.ensure_offline_env({"OFFLINE_TEST_TOKEN": lambda: "ignored"})
    assert second == []
    assert os.environ["OFFLINE_TEST_TOKEN"] == generated[0]


def test_generate_placeholder_credential_entropy():
    token_one = offline.generate_placeholder_credential("entropy-check")
    token_two = offline.generate_placeholder_credential("entropy-check")

    assert token_one != token_two
    assert token_one.startswith("offline-entropy-check-")
    assert token_two.startswith("offline-entropy-check-")
