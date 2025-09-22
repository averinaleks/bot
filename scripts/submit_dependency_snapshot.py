#!/usr/bin/env python3
"""Generate and submit a dependency snapshot to GitHub."""
from __future__ import annotations

import fnmatch
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
_EXCLUDED_DIR_NAMES = {
    ".git",
    ".hg",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "env",
    "node_modules",
    "site-packages",
    "venv",
}
_ALLOWED_HIDDEN_DIR_NAMES = {".github"}
_REQUIREMENT_RE = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[^\]]+\])?==(?P<version>[^\s]+)")
_DEFAULT_API_VERSION = "2022-11-28"
_RETRYABLE_STATUS_CODES = {500, 502, 503, 504}
_TOKEN_PREFIXES = ("ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_")

_SKIPPED_PACKAGES = {"ccxtpro"}


def _should_skip_manifest(name: str, available: set[str]) -> bool:
    """Return ``True`` when the manifest is redundant and can be dropped."""

    path = Path(name)
    if path.suffix == ".out":
        candidate = path.with_suffix(".txt").as_posix()
        if candidate in available:
            return True
    return False


class MissingEnvironmentVariableError(RuntimeError):
    """Raised when a required GitHub environment variable is missing."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Missing required environment variable: {name}")
        self.name = name


class DependencySubmissionError(RuntimeError):
    """Raised when submitting a dependency snapshot fails."""

    def __init__(self, status_code: int | None, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.status_code = status_code
        if cause is not None:
            self.__cause__ = cause


def _should_include_dir(dirname: str) -> bool:
    if dirname in _EXCLUDED_DIR_NAMES:
        return False
    if dirname.startswith(".") and dirname not in _ALLOWED_HIDDEN_DIR_NAMES:
        return False
    return True


def _iter_requirement_files(root: Path) -> Iterable[Path]:
    matches: list[Path] = []
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            dirname for dirname in dirnames if _should_include_dir(dirname)
        )
        for filename in filenames:
            if not any(
                fnmatch.fnmatch(filename, pattern) for pattern in MANIFEST_PATTERNS
            ):
                continue
            path = Path(current_root, filename)
            if not path.is_file():
                continue
            matches.append(path)

    seen: set[Path] = set()
    for path in sorted(matches, key=lambda item: item.relative_to(root).as_posix()):
        if path in seen:
            continue
        seen.add(path)
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


class ResolvedDependencies(OrderedDict[str, ResolvedDependency]):
    """Mapping of dependency identifiers with support for alias lookups."""

    def __init__(self) -> None:
        super().__init__()
        self._aliases: dict[str, str] = {}

    def _resolve_alias(self, key: str) -> str:
        alias = self._aliases.get(key)
        if alias is not None:
            return alias
        normalised = _normalise_name(key)
        alias = self._aliases.get(normalised)
        if alias is not None:
            return alias
        return key

    def add(
        self,
        original_name: str,
        base_name: str,
        package_url: str,
        dependency: ResolvedDependency,
    ) -> None:
        if not super().__contains__(package_url):
            super().__setitem__(package_url, dependency)

        alias_candidates = {
            original_name,
            base_name,
            _normalise_name(original_name),
            _normalise_name(base_name),
        }
        for alias in alias_candidates:
            if alias:
                self._aliases[alias] = package_url

    def __getitem__(self, key: str) -> ResolvedDependency:  # type: ignore[override]
        return super().__getitem__(self._resolve_alias(key))

    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        if isinstance(key, str):
            key = self._resolve_alias(key)
        return super().__contains__(key)

    def get(  # type: ignore[override]
        self, key: str, default: ResolvedDependency | None = None
    ) -> ResolvedDependency | None:
        return super().get(self._resolve_alias(key), default)


def _encode_version_for_purl(version: str) -> str:
    """Return a dependency version encoded for use inside a purl."""

    safe_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-~"
    return quote(version, safe=safe_chars)


def _parse_requirements(path: Path) -> Dict[str, ResolvedDependency]:
    scope = _derive_scope(path.name)
    resolved = ResolvedDependencies()
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

        raw_name = requirement_part.split("==", 1)[0].strip()
        match = _REQUIREMENT_RE.match(requirement_part)
        if not match:
            continue
        matched_name = match.group("name")
        version = match.group("version")
        if not matched_name or not version:
            continue
        # Remove extras if present, e.g. package[extra]==1.0.0
        base_name = matched_name
        if "[" in base_name and "]" in base_name:
            base_name = base_name.split("[", 1)[0]
        package_name = _normalise_name(base_name)
        if package_name in _SKIPPED_PACKAGES:
            continue
        package_url = f"pkg:pypi/{package_name}@{_encode_version_for_purl(version)}"
        dependency: ResolvedDependency = {
            "package_url": package_url,
            "relationship": "direct",
            "scope": scope,
            "dependencies": [],
        }
        resolved.add(raw_name, base_name, package_url, dependency)
    return resolved


def _build_manifests(root: Path) -> Dict[str, Manifest]:
    manifests: Dict[str, Manifest] = OrderedDict()
    for manifest in _iter_requirement_files(root):
        resolved = _parse_requirements(manifest)
        if not resolved:
            continue
        try:
            relative_path = manifest.relative_to(root)
        except ValueError:
            # Fallback for unexpected paths outside of the provided root.
            relative_path = Path(manifest.name)

        relative_str = (
            relative_path.as_posix()
            if isinstance(relative_path, Path)
            else str(relative_path)
        )

        manifests[relative_str] = {
            "name": manifest.name,
            "file": {"source_location": relative_str},
            "resolved": resolved,
        }

    if manifests:
        available = set(manifests.keys())
        manifests = OrderedDict(
            (
                name,
                manifest,
            )
            for name, manifest in manifests.items()
            if not _should_skip_manifest(name, available)
        )
    return manifests


def _env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise MissingEnvironmentVariableError(name)
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


def _log_unexpected_error(exc: Exception) -> None:
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        raise
    print(
        "Dependency snapshot submission skipped из-за непредвиденной ошибки.",
        file=sys.stderr,
    )
    message = str(exc).strip() or exc.__class__.__name__
    print(message, file=sys.stderr)


def _normalise_run_attempt(raw_value: str | None) -> int:
    """Return a validated run attempt number suitable for submission."""

    if not raw_value:
        return 1
    try:
        value = int(raw_value)
    except ValueError:
        print(
            "Invalid GITHUB_RUN_ATTEMPT value. Using fallback value 1.",
            file=sys.stderr,
        )
        return 1
    if value < 1:
        print(
            "GITHUB_RUN_ATTEMPT must be >= 1. Using fallback value 1.",
            file=sys.stderr,
        )
        return 1
    return value


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


def _report_dependency_submission_error(error: DependencySubmissionError) -> None:
    status_code = error.status_code
    if status_code == 401:
        print(
            "Dependency snapshot submission skipped из-за ошибки авторизации токена (HTTP 401).",
            file=sys.stderr,
        )
        return
    if status_code in {403, 404}:
        print(
            "Dependency snapshot submission skipped из-за ограниченных прав доступа.",
            file=sys.stderr,
        )
        return
    if status_code == 422:
        print(
            "Dependency snapshot submission skipped из-за ошибки валидации данных (HTTP 422).",
            file=sys.stderr,
        )
        return
    if status_code in _RETRYABLE_STATUS_CODES:
        print(
            "Dependency snapshot submission skipped из-за временной ошибки сервера GitHub.",
            file=sys.stderr,
        )
        return
    if status_code == 413:
        print(
            "Dependency snapshot submission skipped из-за превышения допустимого размера snapshot (HTTP 413).",
            file=sys.stderr,
        )
        return
    if status_code is None:
        print(
            "Dependency snapshot submission skipped из-за сетевой ошибки.",
            file=sys.stderr,
        )
        return
    print(
        "Dependency snapshot submission skipped из-за ошибки GitHub API.",
        file=sys.stderr,
    )
    message = str(error).strip()
    if status_code:
        print(
            f"Получен код ответа HTTP {status_code}: {message}",
            file=sys.stderr,
        )
    elif message:
        print(message, file=sys.stderr)


def submit_dependency_snapshot() -> None:
    try:
        repository = _env("GITHUB_REPOSITORY")
        token = _env("GITHUB_TOKEN")
        sha = _env("GITHUB_SHA")
        ref = _env("GITHUB_REF")
    except MissingEnvironmentVariableError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "Dependency snapshot submission skipped из-за отсутствия переменных окружения.",
            file=sys.stderr,
        )
        return

    try:
        manifests = _build_manifests(Path("."))
    except Exception as exc:
        _log_unexpected_error(exc)
        return
    if not manifests:
        print("No dependency manifests found.")
        return

    workflow = os.getenv("GITHUB_WORKFLOW", "dependency-graph")
    job_name = os.getenv("GITHUB_JOB", "submit")
    run_id = os.getenv("GITHUB_RUN_ID", str(int(datetime.now(timezone.utc).timestamp())))
    run_attempt = _normalise_run_attempt(os.getenv("GITHUB_RUN_ATTEMPT"))
    correlator = f"{workflow}:{job_name}"

    job_metadata = _job_metadata(repository, run_id, correlator)
    job_metadata["correlator"] = f"{correlator}:attempt-{run_attempt}"

    payload = {
        "version": 0,
        "sha": sha,
        "ref": ref,
        "job": job_metadata,
        "detector": {
            "name": "requirements-parser",
            "version": "1.0.0",
            "url": "https://github.com/averinaleks/bot",
        },
        "scanned": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "manifests": manifests,
        "metadata": {
            "run_attempt": run_attempt,
            "job": job_name,
            "workflow": workflow,
        },
    }

    try:
        base_url = _api_base_url()
        url = f"{base_url}/repos/{repository}/dependency-graph/snapshots"
        body = json.dumps(payload).encode()
        headers_base = {
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": os.getenv("GITHUB_API_VERSION", _DEFAULT_API_VERSION),
            "User-Agent": "dependency-snapshot-script",
        }

        schemes = _auth_schemes(token)
        last_error: DependencySubmissionError | None = None
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
                        (
                            f"Authentication with scheme '{scheme}' failed "
                            f"(HTTP {exc.status_code}). Trying '{next_scheme}'."
                        ),
                        file=sys.stderr,
                    )
                    last_error = exc
                    continue
                last_error = exc
                break

        if last_error is not None:
            _report_dependency_submission_error(last_error)
        return
    except DependencySubmissionError as exc:
        _report_dependency_submission_error(exc)
        return
    except Exception as exc:
        _log_unexpected_error(exc)
        return


if __name__ == "__main__":
    submit_dependency_snapshot()
