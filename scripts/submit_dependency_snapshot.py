#!/usr/bin/env python3
"""Generate and submit a dependency snapshot to GitHub."""
from __future__ import annotations

import fnmatch
import json
import os
import re
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, TypedDict
from urllib.parse import quote, urlparse

_REQUESTS_IMPORT_ERROR: ImportError | None
try:  # pragma: no cover - import guard exercised in tests
    import requests  # type: ignore
except ImportError as import_error:  # pragma: no cover - requests may be absent in CI
    requests = None  # type: ignore[assignment]
    _REQUESTS_IMPORT_ERROR = import_error
else:
    _REQUESTS_IMPORT_ERROR = None


# Ensure the repository root is on ``sys.path`` so that sibling modules can be
# imported when this script is executed directly via ``python
# scripts/submit_dependency_snapshot.py``.  GitHub Actions invokes the script in
# this manner and, without the adjustment, Python would attempt to resolve the
# ``scripts`` package relative to ``scripts/`` instead of the project root.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

if __package__ in {None, ""}:
    # Allow absolute ``scripts`` imports when the module is executed as a
    # script (``python scripts/submit_dependency_snapshot.py``).  Without this
    # adjustment only the ``scripts`` directory is on ``sys.path`` which makes
    # ``import scripts`` fail.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.github_path_resolver import resolve_github_path  # noqa: E402

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
_TOKEN_PAYLOAD_KEYS = ("token", "github_token", "access_token")
_NULL_STRINGS = {"null", "none", "undefined", '""', "''"}
_PAYLOAD_SHA_KEYS = (
    "after",
    "sha",
    "head_sha",
    "commit_oid",
    "commit_sha",
    "after_sha",
    "head_commit_oid",
    "head_commit_sha",
)
_PAYLOAD_REF_KEYS = (
    "ref",
    "ref_name",
    "branch",
    "head_ref",
)
_PAYLOAD_BASE_REF_KEYS = (
    "base_ref",
    "base_branch",
)
_WORKFLOW_RUN_REPOSITORY_KEYS = ("head_repository", "repository")
_DEVELOPMENT_TOKEN_HINTS = ("ci",)

_SKIPPED_PACKAGES = {"ccxtpro"}


def _safe_path_fragment(path: Path) -> str:
    """Return a sanitised POSIX representation for log messages."""

    text = path.as_posix()
    return text.replace("\n", " ").replace("\r", " ")


def _normalise_optional_string(value: str | None) -> str:
    if not value:
        return ""
    candidate = value.strip()
    if candidate.lower() in _NULL_STRINGS:
        return ""
    return candidate


def _extract_payload_value(payload: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        candidate = payload.get(key)
        if isinstance(candidate, str):
            normalised = _normalise_optional_string(candidate)
            if normalised:
                return normalised
    return ""


def _extract_payload_token(payload: Mapping[str, object] | None) -> str:
    if not isinstance(payload, Mapping):
        return ""
    return _extract_payload_value(payload, *_TOKEN_PAYLOAD_KEYS)


def _as_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, MutableMapping):
        return value
    if isinstance(value, Mapping):
        return value
    return None


def _extract_workflow_run(payload: Mapping[str, object] | None) -> Mapping[str, object] | None:
    if not isinstance(payload, Mapping):
        return None
    candidate = payload.get("workflow_run")
    return _as_mapping(candidate)


def _extract_dependency_graph_payload(
    payload: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    if not isinstance(payload, Mapping):
        return None
    candidate = payload.get("dependency_graph")
    return _as_mapping(candidate)


def _extract_workflow_run_repository(payload: Mapping[str, object]) -> str:
    for key in _WORKFLOW_RUN_REPOSITORY_KEYS:
        repository = payload.get(key)
        if isinstance(repository, str):
            normalised = _normalise_optional_string(repository)
            if normalised:
                return normalised
        mapping = _as_mapping(repository)
        if mapping:
            value = mapping.get("full_name")
            if isinstance(value, str):
                normalised = _normalise_optional_string(value)
                if normalised:
                    return normalised
            name = mapping.get("name")
            if isinstance(name, str):
                owner = mapping.get("owner")
                owner_login = ""
                if isinstance(owner, str):
                    owner_login = _normalise_optional_string(owner)
                else:
                    owner_mapping = _as_mapping(owner)
                    if owner_mapping:
                        login = owner_mapping.get("login")
                        if isinstance(login, str):
                            owner_login = _normalise_optional_string(login)
                        if not owner_login:
                            owner_name = owner_mapping.get("name")
                            if isinstance(owner_name, str):
                                owner_login = _normalise_optional_string(owner_name)
                if owner_login:
                    candidate = _normalise_optional_string(name)
                    if candidate:
                        return f"{owner_login}/{candidate}"
    return ""


def _extract_workflow_run_sha(payload: Mapping[str, object]) -> str:
    sha = _extract_payload_value(payload, "head_sha")
    if sha:
        return sha
    head_commit = _as_mapping(payload.get("head_commit"))
    if head_commit:
        sha = _extract_payload_value(head_commit, "id", "sha")
        if sha:
            return sha
    return ""


def _extract_workflow_run_ref(payload: Mapping[str, object]) -> str:
    branch = _extract_payload_value(payload, "head_branch")
    if branch:
        return _normalise_ref_value(branch)
    head_ref = _extract_payload_value(payload, "head_ref")
    if head_ref:
        return _normalise_ref_value(head_ref)
    return ""


_EVENT_PAYLOAD_EVENTS = {
    "repository_dispatch",
    "workflow_run",
    "workflow_call",
    "dynamic",
    "dependency_graph",
}
_SHA_PATTERN = re.compile(r"^[0-9a-f]{40,64}$", re.IGNORECASE)


def _resolve_git_directory(root: Path = REPOSITORY_ROOT) -> Path | None:
    """Return the resolved ``.git`` directory for *root* or ``None`` if missing."""

    git_entry = root / ".git"
    if git_entry.is_dir():
        return git_entry
    if git_entry.is_file():
        try:
            content = git_entry.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if content.startswith("gitdir:"):
            target = content.split(":", 1)[1].strip()
            candidate = (git_entry.parent / target).resolve()
            if candidate.exists():
                return candidate
    return None


def _read_git_ref(git_dir: Path, ref: str) -> str | None:
    """Return the commit hash for *ref* from loose or packed references."""

    ref_path = git_dir / ref
    try:
        data = ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        data = ""
    if data and _SHA_PATTERN.match(data):
        return data

    packed_refs = git_dir / "packed-refs"
    try:
        packed_content = packed_refs.read_text(encoding="utf-8")
    except OSError:
        return None

    for line in packed_content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("^"):
            continue
        if " " not in line:
            continue
        sha, name = line.split(" ", 1)
        sha = sha.strip()
        name = name.strip()
        if name == ref and _SHA_PATTERN.match(sha):
            return sha
    return None


def _read_head(git_dir: Path) -> tuple[str | None, str | None]:
    """Return ``(sha, ref)`` for ``HEAD`` where ``ref`` may be ``None``."""

    head_path = git_dir / "HEAD"
    try:
        raw = head_path.read_text(encoding="utf-8").strip()
    except OSError:
        return (None, None)

    if raw.startswith("ref:"):
        ref = raw.split(":", 1)[1].strip()
        sha = _read_git_ref(git_dir, ref)
        return (sha, ref)
    if _SHA_PATTERN.match(raw):
        return (raw, None)
    return (None, None)


def _iter_remote_refs(git_dir: Path) -> Iterable[tuple[str, str]]:
    """Yield remote reference names and hashes from loose and packed refs."""

    refs: dict[str, str] = {}
    refs_root = git_dir / "refs"
    if refs_root.exists():
        for path in refs_root.rglob("*"):
            if not path.is_file():
                continue
            try:
                sha = path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if not _SHA_PATTERN.match(sha):
                continue
            ref_name = "refs/" + path.relative_to(refs_root).as_posix()
            refs.setdefault(ref_name, sha)

    packed_refs = git_dir / "packed-refs"
    try:
        packed_content = packed_refs.read_text(encoding="utf-8")
    except OSError:
        packed_content = ""
    for line in packed_content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("^"):
            continue
        if " " not in line:
            continue
        sha, name = line.split(" ", 1)
        sha = sha.strip()
        name = name.strip()
        if _SHA_PATTERN.match(sha):
            refs.setdefault(name, sha)

    for name, sha in refs.items():
        if name.startswith("refs/remotes/"):
            yield name, sha


def _discover_git_sha() -> str | None:
    """Return the current commit hash from the local repository."""

    git_dir = _resolve_git_directory()
    if git_dir is None:
        return None
    sha, ref = _read_head(git_dir)
    if sha:
        return sha
    if ref:
        return _read_git_ref(git_dir, ref)
    return None


def _discover_git_ref() -> str | None:
    """Return a ref name for the current ``HEAD`` commit."""

    git_dir = _resolve_git_directory()
    if git_dir is None:
        return None
    sha, ref = _read_head(git_dir)
    if ref:
        return ref
    if not sha:
        return None

    origin_prefix = "refs/remotes/origin/"
    for name, value in _iter_remote_refs(git_dir):
        if not name.startswith(origin_prefix):
            continue
        if value == sha:
            branch = name.removeprefix(origin_prefix)
            if branch:
                return f"refs/heads/{branch}"
    return None


def _load_event_payload() -> dict[str, Any] | None:
    path = resolve_github_path(
        os.getenv("GITHUB_EVENT_PATH"),
        description="GitHub event payload path",
    )
    if path is None:
        return None
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"::warning::Unable to read GitHub event payload: {exc}",
            file=sys.stderr,
        )
        return None
    if isinstance(data, dict):
        return data
    return None


def _should_skip_manifest(
    name: str, available: set[str], available_lower: set[str]
) -> bool:
    """Return ``True`` when the manifest is redundant and can be dropped."""

    path = Path(name)
    if path.suffix.lower() in {".out", ".in"}:
        candidate = path.with_suffix(".txt").as_posix()
        candidate_lower = candidate.lower()
        if candidate in available or candidate_lower in available_lower:
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


def _is_safe_child_path(base: Path, candidate: Path, *, entry_type: str) -> bool:
    """Return ``True`` when *candidate* is a safe descendant of *base*."""

    safe_candidate = _safe_path_fragment(candidate)
    try:
        if candidate.is_symlink():
            print(
                f"::warning::Skipping {entry_type} {safe_candidate}: symbolic links are not processed",
                file=sys.stderr,
            )
            return False
    except OSError as exc:
        print(
            f"::warning::Skipping {entry_type} {safe_candidate}: unable to inspect entry ({exc})",
            file=sys.stderr,
        )
        return False

    try:
        resolved = candidate.resolve(strict=False)
    except OSError as exc:
        print(
            f"::warning::Skipping {entry_type} {safe_candidate}: unable to resolve path ({exc})",
            file=sys.stderr,
        )
        return False

    try:
        resolved.relative_to(base)
    except ValueError:
        print(
            f"::warning::Skipping {entry_type} {safe_candidate}: resolved path escapes repository root",
            file=sys.stderr,
        )
        return False

    return True


def _iter_requirement_files(root: Path) -> Iterable[Path]:
    matches: list[Path] = []
    resolved_root = root.resolve(strict=False)
    for current_root, dirnames, filenames in os.walk(root):
        current_path = Path(current_root)
        filtered_dirnames: list[str] = []
        for dirname in sorted(dirnames):
            if not _should_include_dir(dirname):
                continue
            candidate_dir = current_path / dirname
            if not _is_safe_child_path(
                resolved_root, candidate_dir, entry_type="directory"
            ):
                continue
            filtered_dirnames.append(dirname)
        dirnames[:] = filtered_dirnames
        for filename in filenames:
            filename_lower = filename.lower()
            if not any(
                fnmatch.fnmatch(filename, pattern)
                or fnmatch.fnmatch(filename_lower, pattern)
                for pattern in MANIFEST_PATTERNS
            ):
                continue
            path = current_path / filename
            if not _is_safe_child_path(resolved_root, path, entry_type="file"):
                continue
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
    tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", lowered)
        if token
    ]

    for token in tokens:
        if token.startswith("dev") or token.startswith("test"):
            return "development"
        if token.startswith("health"):
            return "development"
        if token in _DEVELOPMENT_TOKEN_HINTS or any(
            token.startswith(prefix) and len(token) <= len(prefix) + 2
            for prefix in _DEVELOPMENT_TOKEN_HINTS
        ):
            return "development"

    return "runtime"


def _normalise_ref_value(ref: str) -> str:
    ref = ref.strip()
    if not ref:
        return ref
    if ref.startswith("refs/"):
        return ref
    if ref.startswith("heads/") or ref.startswith("tags/"):
        return f"refs/{ref}"
    return f"refs/heads/{ref}"


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

    def _register_alias(self, alias: str, canonical: str) -> None:
        if not alias:
            return
        self._aliases[alias] = canonical
        normalised = _normalise_name(alias)
        self._aliases[normalised] = canonical

    def add(
        self,
        original_name: str,
        base_name: str,
        package_url: str,
        dependency: ResolvedDependency,
        *,
        extra_aliases: Iterable[str] = (),
    ) -> None:
        canonical_name = _normalise_name(base_name) or _normalise_name(original_name)
        if not canonical_name:
            canonical_name = package_url

        existing = super().get(canonical_name)
        if existing is None or existing.get("package_url") != package_url:
            super().__setitem__(canonical_name, dependency)

        alias_candidates = {
            original_name,
            base_name,
            canonical_name,
            package_url,
            _normalise_name(original_name),
            _normalise_name(base_name),
        }
        alias_candidates.update(extra_aliases)

        for alias in alias_candidates:
            if not alias:
                continue
            self._register_alias(alias, canonical_name)

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


def _read_manifest_text(path: Path) -> str | None:
    """Return the decoded contents of a requirements file or ``None`` on error."""

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        relative = path.as_posix()
        print(
            (
                f"Skipping manifest '{relative}' due to encoding error: {exc}. "
                "File contents must be valid UTF-8."
            ),
            file=sys.stderr,
        )
        return None
    except OSError as exc:
        relative = path.as_posix()
        print(
            (
                f"Skipping manifest '{relative}' due to filesystem error: {exc}. "
                "Unable to read the file contents."
            ),
            file=sys.stderr,
        )
        return None


def _parse_requirements(path: Path) -> Dict[str, ResolvedDependency]:
    scope = _derive_scope(path.name)
    resolved = ResolvedDependencies()
    contents = _read_manifest_text(path)
    if contents is None:
        return resolved

    for raw_line in contents.splitlines():
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
        normalised_requirement = requirement_part
        if "==" in requirement_part:
            name_part, version_part = requirement_part.split("==", 1)
            normalised_requirement = f"{_normalise_name(name_part)}=={version_part}"

        resolved.add(
            raw_name,
            base_name,
            package_url,
            dependency,
            extra_aliases=(requirement_part, normalised_requirement),
        )
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
        available_lower = {name.lower() for name in available}
        manifests = OrderedDict(
            (
                name,
                manifest,
            )
            for name, manifest in manifests.items()
            if not _should_skip_manifest(name, available, available_lower)
        )
    return manifests


def _env(name: str) -> str:
    value = _normalise_optional_string(os.getenv(name))
    if not value:
        raise MissingEnvironmentVariableError(name)
    return value


def _api_base_url() -> str:
    api_url = _normalise_optional_string(os.getenv("GITHUB_API_URL")) or "https://api.github.com"
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
    server_url = (
        _normalise_optional_string(os.getenv("GITHUB_SERVER_URL"))
        or "https://github.com"
    ).rstrip("/")
    if run_id.isdigit():
        job["html_url"] = f"{server_url}/{repository}/actions/runs/{run_id}"
    return job


def _auth_schemes(token: str) -> list[str]:
    override = _normalise_optional_string(os.getenv("DEPENDENCY_SNAPSHOT_AUTH_SCHEME"))
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

    raw_value = _normalise_optional_string(raw_value)
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
    _https_components(url)

    if requests is None:
        message = "requests package is required to submit dependency snapshots"
        if _REQUESTS_IMPORT_ERROR is not None:
            message = f"{message}: {_REQUESTS_IMPORT_ERROR}"
        raise DependencySubmissionError(None, message, _REQUESTS_IMPORT_ERROR)

    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            with requests.Session() as session:  # type: ignore[union-attr]
                session.trust_env = False
                session.proxies = {}
                response = session.post(
                    url,
                    data=body,
                    headers=headers,
                    timeout=30,
                    allow_redirects=False,
                )
                status_code = int(response.status_code)
                reason = response.reason or ""
                redirect_location = response.headers.get("Location", "")
                try:
                    if hasattr(response, 'text'):
                        payload_text = response.text
                    else:
                        payload_bytes = getattr(response, 'content', b'')
                        if isinstance(payload_bytes, bytes):
                            payload_text = payload_bytes.decode('utf-8', errors='replace')
                        else:
                            payload_text = str(payload_bytes)
                finally:
                    response.close()

            if 300 <= status_code < 400:
                target = redirect_location or reason or "redirect"
                raise DependencySubmissionError(
                    status_code,
                    f"Failed to submit dependency snapshot: HTTP {status_code}: unexpected redirect to {target}",
                )

            if status_code in _RETRYABLE_STATUS_CODES:
                if attempt < 3:
                    wait_time = 2 ** (attempt - 1)
                    print(
                        "GitHub вернул временную ошибку при отправке snapshot. "
                        f"Повторяем попытку через {wait_time} с...",
                        file=sys.stderr,
                    )
                    time.sleep(wait_time)
                    continue
                raise DependencySubmissionError(
                    status_code,
                    f"GitHub отклонил snapshot зависимостей: HTTP {status_code}: {payload_text.strip() or reason}",
                )

            if status_code >= 400:
                message = payload_text.strip() or reason or ""
                raise DependencySubmissionError(
                    status_code,
                    f"GitHub отклонил snapshot зависимостей: HTTP {status_code}: {message}",
                )

            print(f"Dependency snapshot submitted: HTTP {status_code}")
            return
        except requests.exceptions.Timeout as exc:  # type: ignore[union-attr]
            message = str(exc).strip() or "timed out"
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
        except requests.exceptions.RequestException as exc:  # type: ignore[union-attr]
            message = str(exc).strip() or exc.__class__.__name__
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

    if last_error is not None:
        message = str(last_error).strip() or last_error.__class__.__name__
        raise DependencySubmissionError(None, message, last_error)



_ORIGINAL_SUBMIT_WITH_HEADERS = _submit_with_headers


def _report_dependency_submission_error(error: DependencySubmissionError) -> None:
    status_code = error.status_code
    message = str(error).strip()
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
        if message:
            print(message, file=sys.stderr)
        return
    print(
        "Dependency snapshot submission skipped из-за ошибки GitHub API.",
        file=sys.stderr,
    )
    if status_code:
        detail = message or error.__class__.__name__
        print(
            f"Получен код ответа HTTP {status_code}: {detail}",
            file=sys.stderr,
        )
    elif message:
        print(message, file=sys.stderr)


def submit_dependency_snapshot() -> None:
    submit_func = _submit_with_headers

    if requests is None:
        message = "Dependency snapshot submission skipped: requests package is unavailable. Skipping submission."
        print(message, file=sys.stderr)
        if _REQUESTS_IMPORT_ERROR is not None:
            details = str(_REQUESTS_IMPORT_ERROR).strip()
            if details:
                print(details, file=sys.stderr)
        return

    payload = _load_event_payload()
    event_name = (
        _normalise_optional_string(os.getenv("GITHUB_EVENT_NAME")) or ""
    ).lower()
    allow_event_payload = event_name in _EVENT_PAYLOAD_EVENTS

    client_payload: Mapping[str, object] | None = None
    if allow_event_payload and isinstance(payload, dict):
        raw_client = payload.get("client_payload")
        if isinstance(raw_client, Mapping):
            client_payload = raw_client

    workflow_run_payload = (
        _extract_workflow_run(payload) if allow_event_payload else None
    )
    client_workflow_run_payload = (
        _extract_workflow_run(client_payload) if client_payload is not None else None
    )
    dependency_graph_payload = (
        _extract_dependency_graph_payload(payload) if allow_event_payload else None
    )

    repository = _normalise_optional_string(os.getenv("GITHUB_REPOSITORY"))
    payload_used = False

    if not repository and allow_event_payload:
        for candidate_payload in (
            client_payload,
            client_workflow_run_payload,
            workflow_run_payload,
            dependency_graph_payload,
            payload if isinstance(payload, Mapping) else None,
        ):
            if not isinstance(candidate_payload, Mapping):
                continue
            repository_candidate = _extract_workflow_run_repository(candidate_payload)
            if repository_candidate:
                repository = repository_candidate
                payload_used = True
                break

    if not repository:
        message = "Missing required environment variable: GITHUB_REPOSITORY"
        print(message, file=sys.stderr)
        print(
            "Dependency snapshot submission skipped из-за отсутствия переменных окружения.",
            file=sys.stderr,
        )
        return

    token_env = _normalise_optional_string(os.getenv("GITHUB_TOKEN"))
    sha = _normalise_optional_string(os.getenv("GITHUB_SHA"))
    ref = _normalise_optional_string(os.getenv("GITHUB_REF"))
    payload_used_local = payload_used

    token_override = _extract_payload_token(client_payload)
    if not token_override and allow_event_payload:
        token_override = _extract_payload_token(payload)
    if not token_override and dependency_graph_payload is not None:
        token_override = _extract_payload_token(dependency_graph_payload)
    token = token_override or token_env

    try:
        if not token:
            raise MissingEnvironmentVariableError("GITHUB_TOKEN")
    except MissingEnvironmentVariableError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "Dependency snapshot submission skipped из-за отсутствия переменных окружения.",
            file=sys.stderr,
        )
        return

    if not sha or not ref:
        if allow_event_payload and client_payload is not None:
            if not sha:
                sha_candidate = _extract_payload_value(
                    client_payload, *_PAYLOAD_SHA_KEYS
                )
                if sha_candidate:
                    sha = sha_candidate
                    payload_used_local = True
            if not ref:
                    ref_candidate = _extract_payload_value(
                        client_payload, *_PAYLOAD_REF_KEYS
                    )
                    if ref_candidate:
                        ref = _normalise_ref_value(ref_candidate)
                        payload_used_local = True
        if allow_event_payload and dependency_graph_payload is not None and not sha:
            sha_candidate = _extract_payload_value(
                dependency_graph_payload, *_PAYLOAD_SHA_KEYS
            )
            if sha_candidate:
                sha = sha_candidate
                payload_used_local = True
        if allow_event_payload and not sha and isinstance(payload, Mapping):
            sha_candidate = _extract_payload_value(payload, *_PAYLOAD_SHA_KEYS)
            if sha_candidate:
                sha = sha_candidate
                payload_used_local = True
        if allow_event_payload and not sha and client_workflow_run_payload is not None:
            sha_candidate = _extract_workflow_run_sha(client_workflow_run_payload)
            if sha_candidate:
                sha = sha_candidate
                payload_used_local = True
        if allow_event_payload and dependency_graph_payload is not None and not ref:
            ref_candidate = _extract_payload_value(
                dependency_graph_payload, *_PAYLOAD_REF_KEYS
            )
            if ref_candidate:
                ref = _normalise_ref_value(ref_candidate)
                payload_used_local = True
            elif dependency_graph_payload is not None:
                base_candidate = _extract_payload_value(
                    dependency_graph_payload, *_PAYLOAD_BASE_REF_KEYS
                )
                if base_candidate:
                    ref = _normalise_ref_value(base_candidate)
                    payload_used_local = True
        if allow_event_payload and not ref and isinstance(payload, Mapping):
            ref_candidate = _extract_payload_value(payload, *_PAYLOAD_REF_KEYS)
            if ref_candidate:
                ref = _normalise_ref_value(ref_candidate)
                payload_used_local = True
            else:
                base_candidate = _extract_payload_value(
                    payload, *_PAYLOAD_BASE_REF_KEYS
                )
                if base_candidate:
                    ref = _normalise_ref_value(base_candidate)
                    payload_used_local = True
        if allow_event_payload and not ref and client_workflow_run_payload is not None:
            ref_candidate = _extract_workflow_run_ref(client_workflow_run_payload)
            if ref_candidate:
                ref = ref_candidate
                payload_used_local = True
        if allow_event_payload and not sha and workflow_run_payload is not None:
            sha_candidate = _extract_workflow_run_sha(workflow_run_payload)
            if sha_candidate:
                sha = sha_candidate
                payload_used_local = True
        if allow_event_payload and not ref and workflow_run_payload is not None:
            ref_candidate = _extract_workflow_run_ref(workflow_run_payload)
            if ref_candidate:
                ref = ref_candidate
                payload_used_local = True

    if not sha:
        sha = _discover_git_sha() or ""
    if not ref:
        ref = _discover_git_ref() or ""

    sha = _normalise_optional_string(sha)  # Ensure fallbacks did not produce sentinels
    ref = _normalise_optional_string(ref)

    try:
        if not sha:
            raise MissingEnvironmentVariableError("GITHUB_SHA")
        if not ref:
            raise MissingEnvironmentVariableError("GITHUB_REF")
    except MissingEnvironmentVariableError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "Dependency snapshot submission skipped из-за отсутствия переменных окружения.",
            file=sys.stderr,
        )
        return

    if payload_used_local:
        print("Using event payload to populate snapshot metadata.", flush=True)

    try:
        manifests = _build_manifests(Path("."))
    except Exception as exc:
        _log_unexpected_error(exc)
        return
    if not manifests:
        print("No dependency manifests found.")
        return

    workflow = (
        _normalise_optional_string(os.getenv("GITHUB_WORKFLOW"))
        or "dependency-graph"
    )
    job_name = _normalise_optional_string(os.getenv("GITHUB_JOB")) or "submit"
    run_id = _normalise_optional_string(os.getenv("GITHUB_RUN_ID")) or str(
        int(datetime.now(timezone.utc).timestamp())
    )
    run_attempt = _normalise_run_attempt(os.getenv("GITHUB_RUN_ATTEMPT"))
    correlator = f"{workflow}:{job_name}"

    job_metadata = _job_metadata(repository, run_id, correlator)
    job_metadata["correlator"] = f"{correlator}:attempt-{run_attempt}"

    metadata_workflow = (
        _normalise_optional_string(os.getenv("DEPENDENCY_SNAPSHOT_WORKFLOW"))
        or "dependency-graph"
    )
    metadata_job = (
        _normalise_optional_string(os.getenv("DEPENDENCY_SNAPSHOT_JOB")) or "submit"
    )

    metadata = {
        "run_attempt": str(run_attempt),
        "job": str(metadata_job),
        "workflow": str(metadata_workflow),
    }

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
        "metadata": metadata,
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
                submit_func(url, body, headers)
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
