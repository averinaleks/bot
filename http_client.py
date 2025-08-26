"""Utilities for creating HTTP clients with default timeouts."""

from __future__ import annotations

import os

import httpx
import requests

DEFAULT_TIMEOUT = float(os.getenv("MODEL_DOWNLOAD_TIMEOUT", "30"))


def get_requests_session(timeout: float = DEFAULT_TIMEOUT) -> requests.Session:
    """Return a :class:`requests.Session` with a default timeout."""
    session = requests.Session()
    original = session.request

    def request(method: str, url: str, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return original(method, url, **kwargs)

    session.request = request  # type: ignore[assignment]
    return session


def get_httpx_client(timeout: float = DEFAULT_TIMEOUT, **kwargs) -> httpx.Client:
    """Return an :class:`httpx.Client` with a default timeout."""
    kwargs.setdefault("timeout", timeout)
    return httpx.Client(**kwargs)
