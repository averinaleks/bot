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

MANIFEST_PATTERNS = ("requirements*.txt", "requirements*.in", "requirements*.out")
_REQUIREMENT_RE = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[^\]]+\])?==(?P<version>[^\s]+)")
_DEFAULT_API_VERSION = "2022-11-28"
_RETRYABLE_STATUS_CODES = {500, 502, 503, 504}


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


def _build_request(url: str, body: bytes, headers: dict[str, str]) -> Request:
    return Request(url, data=body, headers=headers, method="POST")


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
    job = os.getenv("GITHUB_JOB", "submit")
    run_id = os.getenv("GITHUB_RUN_ID", str(int(datetime.now(timezone.utc).timestamp())))
    run_attempt = os.getenv("GITHUB_RUN_ATTEMPT", "1")

    payload = {
        "version": 0,
        "sha": sha,
        "ref": ref,
        "job": {
            "correlator": f"{workflow}-{job}",
            "id": f"{run_id}-{run_attempt}",
        },
        "detector": {
            "name": "requirements-parser",
            "version": "1.0.0",
            "url": "https://github.com/averinaleks/bot",
        },
        "metadata": {
            "dependency_count": sum(len(entry["resolved"]) for entry in manifests.values()),
        },
        "manifests": manifests,
    }

    base_url = _api_base_url()
    url = f"{base_url}/repos/{repository}/dependency-graph/snapshots"
    parsed_url = urlparse(url)
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise RuntimeError("Запрос зависимостей разрешён только по HTTPS")
    body = json.dumps(payload).encode()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "X-GitHub-Api-Version": os.getenv("GITHUB_API_VERSION", _DEFAULT_API_VERSION),
        "User-Agent": "dependency-snapshot-script",
    }

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
            raise RuntimeError("GitHub отклонил snapshot зависимостей") from exc
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
            raise RuntimeError(
                f"Не удалось отправить snapshot зависимостей: {exc.reason}"
            ) from exc

    if last_error is not None:
        raise last_error


if __name__ == "__main__":
    submit_dependency_snapshot()
