#!/usr/bin/env python
"""Simple health check for API endpoints."""
import sys
from typing import Iterable

import logging
import requests


logger = logging.getLogger(__name__)


def check_endpoints(base_url: str, endpoints: Iterable[str]) -> int:
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
