#!/usr/bin/env python
"""Simple health check for API endpoints."""
import logging
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Set

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

from ipaddress import ip_address
from urllib.parse import urlparse, urlunparse

import requests  # type: ignore[import-untyped]

from services.logging_utils import sanitize_log_value


logger = logging.getLogger(__name__)


_ALLOWED_HOSTS_ENV = "HEALTH_CHECK_ALLOWED_HOSTS"
_DEFAULT_ALLOWED_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})


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


def _load_allowed_hosts() -> Set[str]:
    hosts: Set[str] = set(_DEFAULT_ALLOWED_HOSTS)
    raw = os.getenv(_ALLOWED_HOSTS_ENV)
    if not raw:
        return hosts
    for part in raw.split(","):
        candidate = part.strip()
        if not candidate:
            continue
        if candidate.startswith("[") and candidate.endswith("]"):
            candidate = candidate[1:-1]
        hosts.add(candidate.lower())
    return hosts


def _normalise_base_url(base_url: str, allowed_hosts: Set[str]) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            f"Unsupported base URL scheme: {parsed.scheme or '<missing>'}"
        )
    if parsed.username or parsed.password:
        raise ValueError("Base URL must not include credentials")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Base URL must include a hostname")

    lowered = hostname.lower()
    is_local = _is_local_host(hostname)
    if lowered not in allowed_hosts and not is_local:
        raise ValueError(f"Base URL host {hostname!r} is not permitted")

    if parsed.scheme == "http" and not is_local:
        raise ValueError(
            f"Refusing insecure HTTP health checks for non-local host {hostname!r}"
        )

    if parsed.query or parsed.fragment:
        raise ValueError("Base URL must not include query or fragment components")

    if parsed.path and parsed.path != "/":
        collapsed = re.sub(r"/+/", "/", parsed.path)
        if not collapsed.startswith("/"):
            collapsed = f"/{collapsed}"
        path = collapsed.rstrip("/")
    else:
        path = ""

    netloc = hostname
    if parsed.port:
        if not (0 < parsed.port < 65536):
            raise ValueError("Base URL port is out of range")
        netloc = f"{hostname}:{parsed.port}"

    normalised = urlunparse((parsed.scheme, netloc, path, "", "", ""))
    return normalised.rstrip("/") or normalised


def check_endpoints(base_url: str, endpoints: Iterable[str]) -> int:
    allowed_hosts = _load_allowed_hosts()
    try:
        normalised = _normalise_base_url(base_url, allowed_hosts)
    except ValueError as exc:
        logger.error("%s", sanitize_log_value(str(exc)))
        return 1

    base = normalised.rstrip("/")
    for endpoint in endpoints:
        suffix = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        url = f"{base}{suffix}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            logger.info(
                "%s -> %s",
                sanitize_log_value(url),
                response.status_code,
            )
        except requests.RequestException as exc:
            logger.error(
                "Health check failed for %s: %s",
                sanitize_log_value(url),
                sanitize_log_value(str(exc)),
            )
            return 1
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    base_url = "http://localhost:8000"
    endpoints = ["/health", "/ping"]
    return check_endpoints(base_url, endpoints)


if __name__ == "__main__":
    sys.exit(main())
