#!/usr/bin/env python3
"""Generate and submit a dependency snapshot to GitHub."""
from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, TypedDict
from urllib.error import HTTPError
from urllib.request import Request, urlopen

MANIFEST_PATTERNS = ("requirements*.txt", "requirements*.in", "requirements*.out")


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
            # Skip include/constraint directives.
            continue
        requirement_part = line.split(";", 1)[0].strip()
        if not requirement_part or "==" not in requirement_part:
            continue
        name, version = (segment.strip() for segment in requirement_part.split("==", 1))
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

    url = f"https://api.github.com/repos/{repository}/dependency-graph/snapshots"
    body = json.dumps(payload).encode()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "User-Agent": "dependency-snapshot-script",
    }
    request = Request(url, data=body, headers=headers, method="POST")

    try:
        with urlopen(request) as response:
            print(f"Dependency snapshot submitted: HTTP {response.status}")
    except HTTPError as error:
        message = error.read().decode() if error.fp else error.reason
        print(f"Failed to submit dependency snapshot: HTTP {error.code}: {message}", file=sys.stderr)
        raise


if __name__ == "__main__":
    submit_dependency_snapshot()
