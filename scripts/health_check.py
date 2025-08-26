#!/usr/bin/env python
"""Simple health check for API endpoints."""
import sys
import time
from typing import Iterable

import requests


def check_endpoints(base_url: str, endpoints: Iterable[str]) -> int:
    for endpoint in endpoints:
        url = f"{base_url.rstrip('/')}{endpoint}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            print(f"{url} -> {response.status_code}")
        except Exception as exc:  # noqa: BLE001
            print(f"Health check failed for {url}: {exc}")
            return 1
    return 0


def main() -> int:
    base_url = "http://localhost:8000"
    endpoints = ["/health", "/ping"]
    return check_endpoints(base_url, endpoints)


if __name__ == "__main__":
    sys.exit(main())
