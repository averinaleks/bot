#!/usr/bin/env python3
"""Generate and submit a dependency snapshot to GitHub."""
from __future__ import annotations

import http.client
import json
import os
import re
import socket
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, TypedDict

from urllib.parse import quote, urlparse

MANIFEST_PATTERNS = (
    "requirements*.txt",
    "requirements*.in",
    "requirements*.out",
)
_REQUIREMENT_RE = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[^\]]+\])?==(?P<version>[^\s]+)")
_DEFAULT_API_VERSION = "2022-11-28"
_RETRYABLE_STATUS_CODES = {500, 502, 503, 504}
_TOKEN_PREFIXES = ("ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_")

_SKIPPED_PACKAGES = {"ccxtpro"}


class DependencySubmissionError(RuntimeError):
    """Raised when submitting a dependency snapshot fails."""

    def __init__(self, status_code: int | None, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.status_code = status_code
        if cause is not None:
            self.__cause__ = cause


def _iter_requirement_files(root: Path) -> Iterable[Path]:
    for pattern in MANIFEST_PATTERNS:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                yield path


def _normalise_name(name: str) -> str:
    return name.replace("_", "-").lower()


def _derive_scope(manifest_name: str) -> str:
    lowered = manifest_name.lower()
    if any(token in lowered for token in ("dev", "test", "ci", "health")):
        return "development"
    return "runtime"


class ResolvedDependency(TypedDict):
    package_url: str
    relationship: str
    scope: str
    dependencies: list[str]


class ManifestFile(TypedDict):
    source_location: str


class Manifest(TypedDict):
    name: str
    file: ManifestFile
    resolved: Dict[str, ResolvedDependency]


def _encode_version_for_purl(version: str) -> str:
    """Return a dependency version encoded for use inside a purl."""

    safe_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-~"
    return quote(version, safe=safe_chars)


def _parse_requirements(path: Path) -> Dict[str, ResolvedDependency]:
    scope = _derive_scope(path.name)
    resolved: Dict[str, ResolvedDependency] = OrderedDict()
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r", "--", "-c")):
            # Skip include/constraint directives, hashes and pip options.
            continue

        while line.endswith("\\"):
            line = line[:-1].rstrip()

        requirement_part = line.split("#", 1)[0].strip()
        if not requirement_part:
            continue

        requirement_part = requirement_part.split(";", 1)[0].strip()
        if not requirement_part:
            continue

        match = _REQUIREMENT_RE.match(requirement_part)
        if not match:
            continue
        name = match.group("name")
        version = match.group("version")
        if not name or not version:
            continue
        # Remove extras if present, e.g. package[extra]==1.0.0
        if "[" in name and "]" in name:
            name = name.split("[", 1)[0]
        package_name = _normalise_name(name)
        if package_name in _SKIPPED_PACKAGES:
            continue
        package_url = f"pkg:pypi/{package_name}@{_encode_version_for_purl(version)}"
        resolved[package_url] = {
            "package_url": package_url,
            "relationship": "direct",
            "scope": scope,
            "dependencies": [],
        }
    return resolved


def _build_manifests(root: Path) -> Dict[str, Manifest]:
    manifests: Dict[str, Manifest] = OrderedDict()
    for manifest in _iter_requirement_files(root):
        resolved = _parse_requirements(manifest)
        if not resolved:
            continue
        manifests[str(manifest)] = {
            "name": manifest.name,
            "file": {"source_location": str(manifest)},
            "resolved": resolved,
        }
    return manifests


def _env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        print(f"Missing required environment variable: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def _api_base_url() -> str:
    api_url = os.getenv("GITHUB_API_URL") or "https://api.github.com"
    return api_url.rstrip("/")


def _https_components(url: str) -> tuple[str, int, str]:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise DependencySubmissionError(
            None, "Запрос зависимостей разрешён только по HTTPS"
        )
    if parsed.username or parsed.password:
        raise DependencySubmissionError(
            None, "URL snapshot не должен содержать учетные данные"
        )
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return parsed.hostname, parsed.port or 443, path


def _job_metadata(repository: str, run_id: str, correlator: str) -> dict[str, str]:
    job: dict[str, str] = {"id": run_id, "correlator": correlator}
    server_url = os.getenv("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    if run_id.isdigit():
        job["html_url"] = f"{server_url}/{repository}/actions/runs/{run_id}"
    return job


def _auth_schemes(token: str) -> list[str]:
    override = os.getenv("DEPENDENCY_SNAPSHOT_AUTH_SCHEME")
    if override:
        return [override]
    if token.startswith(_TOKEN_PREFIXES):
        return ["Bearer", "token"]
    return ["token", "Bearer"]


def _submit_with_headers(url: str, body: bytes, headers: dict[str, str]) -> None:
    host, port, path = _https_components(url)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        connection = http.client.HTTPSConnection(host, port, timeout=30)
        try:
            connection.request("POST", path, body=body, headers=headers)
            response = connection.getresponse()
            status_code = int(response.status or 0)
            reason = response.reason or ""
            payload = response.read()
        except socket.timeout as exc:
            message = str(exc) or "timed out"
            if attempt < 3:
                wait_time = 2 ** (attempt - 1)
                print(
                    f"Network timeout '{message}'. Retrying in {wait_time} s...",
                    file=sys.stderr,
                )
                time.sleep(wait_time)
                last_error = exc
                continue
            last_error = exc
            break
        except (OSError, http.client.HTTPException) as exc:
            message = str(exc) or exc.__class__.__name__
            if attempt < 3:
                wait_time = 2 ** (attempt - 1)
                print(
                    f"Network error '{message}'. Retrying in {wait_time} s...",
                    file=sys.stderr,
                )
                time.sleep(wait_time)
                last_error = exc
                continue
            last_error = exc
            break
        finally:
            connection.close()

        if status_code in _RETRYABLE_STATUS_CODES and attempt < 3:
            wait_time = 2 ** (attempt - 1)
            print(
                f"Received retryable error HTTP {status_code}. Retrying in {wait_time} s...",
                file=sys.stderr,
            )
            time.sleep(wait_time)
            last_error = DependencySubmissionError(
                status_code,
                f"Retryable HTTP {status_code}: {reason}",
            )
            continue

        if status_code >= 400:
            message = payload.decode(errors="replace") or reason
            print(
                f"Failed to submit dependency snapshot: HTTP {status_code}: {message}",
                file=sys.stderr,
            )
            raise DependencySubmissionError(
                status_code,
                f"GitHub отклонил snapshot зависимостей: HTTP {status_code}: {message}",
            )

        print(f"Dependency snapshot submitted: HTTP {status_code}")
        return

    if last_error is not None:
        message = str(last_error) or last_error.__class__.__name__
        raise DependencySubmissionError(None, message, last_error)


def submit_dependency_snapshot() -> None:
    repository = _env("GITHUB_REPOSITORY")
    token = _env("GITHUB_TOKEN")
    sha = _env("GITHUB_SHA")
    ref = _env("GITHUB_REF")

    manifests = _build_manifests(Path("."))
    if not manifests:
        print("No dependency manifests found.")
        return

    workflow = os.getenv("GITHUB_WORKFLOW", "dependency-graph")
    job_name = os.getenv("GITHUB_JOB", "submit")
    run_id = os.getenv("GITHUB_RUN_ID", str(int(datetime.now(timezone.utc).timestamp())))
    run_attempt = os.getenv("GITHUB_RUN_ATTEMPT", "1")
    correlator = f"{workflow}-{job_name}"

    payload = {
        "version": 0,
        "sha": sha,
        "ref": ref,
        "job": _job_metadata(repository, run_id, correlator),
        "detector": {
            "name": "requirements-parser",
            "version": "1.0.0",
            "url": "https://github.com/averinaleks/bot",
        },
        "metadata": {
            "dependency_count": sum(len(entry["resolved"]) for entry in manifests.values()),
            "run_attempt": run_attempt,
        },
        "scanned": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "manifests": manifests,
    }

    base_url = _api_base_url()
    url = f"{base_url}/repos/{repository}/dependency-graph/snapshots"
    parsed_url = urlparse(url)
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise RuntimeError("Запрос зависимостей разрешён только по HTTPS")
    body = json.dumps(payload).encode()
    headers_base = {
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "X-GitHub-Api-Version": os.getenv("GITHUB_API_VERSION", _DEFAULT_API_VERSION),
        "User-Agent": "dependency-snapshot-script",
    }

    schemes = _auth_schemes(token)
    last_error: Exception | None = None
    for index, scheme in enumerate(schemes):
        headers = dict(headers_base, Authorization=f"{scheme} {token}")
        try:
            _submit_with_headers(url, body, headers)
            return
        except DependencySubmissionError as exc:
            if (
                exc.status_code in {401, 403}
                and index < len(schemes) - 1
            ):
                next_scheme = schemes[index + 1]
                print(
                    f"Authentication with scheme '{scheme}' failed (HTTP {exc.status_code}). Trying '{next_scheme}'.",
                    file=sys.stderr,
                )
                last_error = exc
                continue
            last_error = exc
            break

    if last_error is not None:
        if isinstance(last_error, DependencySubmissionError):
            if last_error.status_code == 401:
                print(
                    "Dependency snapshot submission skipped из-за ошибки авторизации токена (HTTP 401).",
                    file=sys.stderr,
                )
                return
            if last_error.status_code in {403, 404}:
                print(
                    "Dependency snapshot submission skipped из-за ограниченных прав доступа.",
                    file=sys.stderr,
                )
                return
        if isinstance(last_error, DependencySubmissionError):
            if last_error.status_code is None:
                print(
                    "Dependency snapshot submission skipped из-за сетевой ошибки.",
                    file=sys.stderr,
                )
                return
        raise last_error


if __name__ == "__main__":
    submit_dependency_snapshot()
