#!/usr/bin/env python
"""Simple health check for API endpoints."""
import logging
import os
import re
import sys
import time
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
from http_client import get_requests_session


logger = logging.getLogger(__name__)


_ALLOWED_HOSTS_ENV = "HEALTH_CHECK_ALLOWED_HOSTS"
_BASE_URL_ENV = "HEALTH_CHECK_BASE_URL"
_ENDPOINTS_ENV = "HEALTH_CHECK_ENDPOINTS"
_MAX_ATTEMPTS_ENV = "HEALTH_CHECK_MAX_ATTEMPTS"
_DELAY_SECONDS_ENV = "HEALTH_CHECK_DELAY_SECONDS"
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


def _safe_int_env(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Env %s=%s is not an integer, using default %s",
            name,
            sanitize_log_value(raw),
            default,
        )
        return default
    return max(minimum, value)


def _safe_float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Env %s=%s is not a number, using default %s",
            name,
            sanitize_log_value(raw),
            default,
        )
        return default
    return value if value > minimum else minimum


def check_endpoints(
    base_url: str,
    endpoints: Iterable[str],
    *,
    max_attempts: int = 5,
    delay_seconds: float = 2.0,
) -> int:
    allowed_hosts = _load_allowed_hosts()
    try:
        normalised = _normalise_base_url(base_url, allowed_hosts)
    except ValueError as exc:
        logger.error("%s", sanitize_log_value(str(exc)))
        return 1

    base = normalised.rstrip("/")
    with get_requests_session(timeout=10.0, verify=True) as session:
        for endpoint in endpoints:
            suffix = endpoint if endpoint.startswith("/") else f"/{endpoint}"
            url = f"{base}{suffix}"
            parsed = urlparse(url)
            if parsed.scheme == "http" and not _is_local_host(parsed.hostname):
                logger.error(
                    "Refusing insecure HTTP health check for non-local endpoint %s",
                    sanitize_log_value(url),
                )
                return 1

            attempts_left = max(1, max_attempts)
            while attempts_left:
                try:
                    response = session.get(url)
                    response.raise_for_status()
                except requests.RequestException as exc:
                    attempts_left -= 1
                    if attempts_left <= 0:
                        logger.error(
                            "Health check failed for %s: %s",
                            sanitize_log_value(url),
                            sanitize_log_value(str(exc)),
                        )
                        return 1
                    logger.warning(
                        "Attempt %s for %s failed: %s – retrying in %.1fs",
                        max_attempts - attempts_left,
                        sanitize_log_value(url),
                        sanitize_log_value(str(exc)),
                        delay_seconds,
                    )
                    time.sleep(max(0.0, delay_seconds))
                    continue

                logger.info(
                    "%s -> %s",
                    sanitize_log_value(url),
                    response.status_code,
                )
                break
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    base_url = os.getenv(_BASE_URL_ENV, "http://localhost:8000")
    raw_endpoints = os.getenv(_ENDPOINTS_ENV)
    if raw_endpoints:
        endpoints = [
            part.strip()
            for part in raw_endpoints.split(",")
            if part.strip()
        ]
        if not endpoints:
            endpoints = ["/health", "/ping"]
    else:
        endpoints = ["/health", "/ping"]
    max_attempts = _safe_int_env(_MAX_ATTEMPTS_ENV, 5, minimum=1)
    delay_seconds = _safe_float_env(_DELAY_SECONDS_ENV, 2.0, minimum=0.0)
    return check_endpoints(
        base_url,
        endpoints,
        max_attempts=max_attempts,
        delay_seconds=delay_seconds,
    )


if __name__ == "__main__":
    sys.exit(main())
