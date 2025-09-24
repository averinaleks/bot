#!/usr/bin/env python
"""Simple health check for API endpoints."""
import sys
from typing import Iterable

import logging
from ipaddress import ip_address
from urllib.parse import urlparse

import requests  # type: ignore[import-untyped]


logger = logging.getLogger(__name__)


def _is_local_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    lowered = hostname.lower()
    if lowered == "localhost":
        return True
    try:
        parsed = ip_address(lowered)
    except ValueError:
        return False
    return parsed.is_loopback or parsed.is_private


def check_endpoints(base_url: str, endpoints: Iterable[str]) -> int:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        logger.error("Unsupported base URL scheme: %s", parsed.scheme or "<missing>")
        return 1
    if parsed.scheme == "http" and not _is_local_host(parsed.hostname):
        logger.error(
            "Refusing to perform insecure HTTP health checks for non-local host %s",
            parsed.hostname or "<missing>",
        )
        return 1

    for endpoint in endpoints:
        url = f"{base_url.rstrip('/')}{endpoint}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            logger.info("%s -> %s", url, response.status_code)
        except requests.RequestException as exc:
            logger.error("Health check failed for %s: %s", url, exc)
            return 1
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    base_url = "http://localhost:8000"
    endpoints = ["/health", "/ping"]
    return check_endpoints(base_url, endpoints)


if __name__ == "__main__":
    sys.exit(main())
