"""Utilities for creating HTTP clients with default timeouts."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from functools import wraps
from typing import Generator

import httpx
import requests

DEFAULT_TIMEOUT_STR = os.getenv("MODEL_DOWNLOAD_TIMEOUT", "30")
try:
    DEFAULT_TIMEOUT = float(DEFAULT_TIMEOUT_STR)
except ValueError:
    logging.warning(
        "Invalid MODEL_DOWNLOAD_TIMEOUT '%s'; using default timeout 30s",
        DEFAULT_TIMEOUT_STR,
    )
    DEFAULT_TIMEOUT = 30.0


@contextmanager
def get_requests_session(
    timeout: float = DEFAULT_TIMEOUT,
) -> Generator[requests.Session, None, None]:
    """Return a :class:`requests.Session` with a default timeout."""
    session = requests.Session()
    original = session.request

    @wraps(original)
    def request(method: str, url: str, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return original(method, url, **kwargs)

    session.request = request  # type: ignore[assignment]
    try:
        yield session
    finally:
        session.close()


@contextmanager
def get_httpx_client(
    timeout: float = DEFAULT_TIMEOUT, **kwargs
) -> Generator[httpx.Client, None, None]:
    """Return an :class:`httpx.Client` with a default timeout."""
    kwargs.setdefault("timeout", timeout)
    client = httpx.Client(**kwargs)
    try:
        yield client
    finally:
        client.close()
