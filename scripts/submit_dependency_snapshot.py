#!/usr/bin/env python3
"""Generate and submit a dependency snapshot to GitHub."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, TypedDict

from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

MANIFEST_PATTERNS = ("requirements*.txt",)
_REQUIREMENT_RE = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[^\]]+\])?==(?P<version>[^\s]+)")
_DEFAULT_API_VERSION = "2022-11-28"
_RETRYABLE_STATUS_CODES = {500, 502, 503, 504}
_TOKEN_PREFIXES = ("ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_")


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
        package_url = f"pkg:pypi/{package_name}@{version}"
        resolved[package_name] = {
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


def _build_request(url: str, body: bytes, headers: dict[str, str]) -> Request:
    return Request(url, data=body, headers=headers, method="POST")


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
    last_error: Exception | None = None
    for attempt in range(1, 4):
        request = _build_request(url, body, headers)
        try:
            with urlopen(request, timeout=30) as response:
                status_code = int(response.getcode() or 0)
                print(f"Dependency snapshot submitted: HTTP {status_code}")
                return
        except HTTPError as exc:
            message = exc.read().decode(errors="replace") if exc.fp else exc.reason
            if exc.code in _RETRYABLE_STATUS_CODES and attempt < 3:
                wait_time = 2 ** (attempt - 1)
                print(
                    f"Received retryable error HTTP {exc.code}. Retrying in {wait_time} s...",
                    file=sys.stderr,
                )
                time.sleep(wait_time)
                last_error = exc
                continue
            print(
                f"Failed to submit dependency snapshot: HTTP {exc.code}: {message}",
                file=sys.stderr,
            )
            raise DependencySubmissionError(
                exc.code,
                f"GitHub отклонил snapshot зависимостей: HTTP {exc.code}: {message}",
                exc,
            )
        except URLError as exc:
            if attempt < 3:
                wait_time = 2 ** (attempt - 1)
                print(
                    f"Network error '{exc.reason}'. Retrying in {wait_time} s...",
                    file=sys.stderr,
                )
                time.sleep(wait_time)
                last_error = exc
                continue
            raise DependencySubmissionError(
                None,
                f"Не удалось отправить snapshot зависимостей: {exc.reason}",
                exc,
            )

    if last_error is not None:
        raise last_error


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
        if isinstance(last_error, DependencySubmissionError) and last_error.status_code in {403, 404}:
            print(
                "Dependency snapshot submission skipped из-за ограниченных прав доступа.",
                file=sys.stderr,
            )
            return
        raise last_error


if __name__ == "__main__":
    submit_dependency_snapshot()
