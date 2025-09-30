"""Helpers for generating pull request diffs for the GPT-OSS workflow.

The GitHub Actions workflow previously relied on shell pipelines composed of
``curl`` and ``jq`` to fetch pull request metadata and prepare a filtered diff.
That approach introduced two recurring sources of instability:

* ``jq`` had to be installed via ``apt`` on every run which occasionally failed
  because of transient package mirror issues.
* Any HTTP hiccup immediately caused the entire step to fail, even though the
  workflow can gracefully degrade by simply skipping the review.

This module re-implements the diff preparation logic in Python so the workflow
no longer depends on external packages.  Errors are surfaced as GitHub Actions
annotations and the script always exits with ``0`` to avoid failing the overall
job when the diff cannot be produced.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import ssl
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPSHandler, Request, build_opener

if __package__ in {None, ""}:
    package_root = Path(__file__).resolve().parent.parent
    package_root_str = str(package_root)
    if package_root_str not in sys.path:
        sys.path.insert(0, package_root_str)

from scripts.github_paths import resolve_github_path


_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_GIT_REF_RE = re.compile(r"^(?!-)(?!.*\.\.)(?!.*//)(?!.*@\{)(?!.*\\0)[\w./-]+$")
_INVALID_PATH_CHARS = {"\x00", "\n", "\r"}
_SAFE_PATH_CHARS_RE = re.compile(r"^[A-Za-z0-9_./*?\[\]-]+$")
_ALLOWED_PATHSPEC_PREFIXES = ("", ":(glob)")
_ALLOWED_GIT_OPTIONS = {"--no-tags"}
_ALLOWED_GITHUB_API_HOSTS = {"api.github.com"}


@dataclass
class PullRequestInfo:
    """Subset of pull request metadata required to prepare a diff."""

    base_sha: str
    base_ref: str


@dataclass(slots=True)
class DiffComputation:
    """Result of running ``git diff`` for the requested changes."""

    content: str
    has_diff: bool


@dataclass(slots=True)
class GitCompletedProcess:
    """Lightweight replacement for :class:`subprocess.CompletedProcess`."""

    args: tuple[str, ...]
    returncode: int
    stdout: str | None = None
    stderr: str | None = None


class GitCommandError(RuntimeError):
    """Raised when a git invocation returns a non-zero exit status."""

    def __init__(
        self,
        returncode: int,
        cmd: Sequence[str],
        stdout: str | None,
        stderr: str | None,
    ) -> None:
        message = f"Command {tuple(cmd)!r} exited with status {returncode}"
        super().__init__(message)
        self.returncode = returncode
        self.cmd = tuple(cmd)
        self.stdout = stdout
        self.stderr = stderr


def _validate_git_sha(sha: str) -> str:
    """Return *sha* if it is a valid 40-character hexadecimal commit hash."""

    if not _SHA_RE.match(sha):
        raise RuntimeError("Получен некорректный SHA коммита")
    return sha


def _validate_git_ref(ref: str) -> str:
    """Ensure *ref* is a safe Git refname.

    The validation mirrors the rules enforced by ``git-check-ref-format``.  It
    rejects refnames containing dangerous sequences such as ``..``, ``//`` or
    ``@{`` which could be interpreted specially by Git.
    """

    if not ref:
        raise RuntimeError("Не указан base_ref для diff")
    if not _GIT_REF_RE.match(ref) or ref.endswith(".lock") or ref.endswith("/"):
        raise RuntimeError("Получен недопустимый git ref")
    return ref


def _validate_path_argument(path: str) -> str:
    """Return a sanitised Git pathspec argument."""

    if not path:
        raise RuntimeError("Путь для diff не может быть пустым")
    if path[0] == "-":
        raise RuntimeError("Путь не должен начинаться с '-' (интерпретируется как опция)")
    if any(ord(char) < 32 or char in _INVALID_PATH_CHARS for char in path):
        raise RuntimeError("Путь содержит недопустимые символы")

    prefix = ""
    remainder = path
    for allowed in _ALLOWED_PATHSPEC_PREFIXES:
        if allowed and remainder.startswith(allowed):
            prefix = allowed
            remainder = remainder[len(allowed) :]
            if remainder.startswith("/"):
                remainder = remainder[1:]
            break
    else:
        if ":" in path:
            raise RuntimeError("Путь содержит недопустимый префикс pathspec")

    if not remainder:
        raise RuntimeError("Путь pathspec должен содержать сегменты после префикса")
    if remainder[0] == "-":
        raise RuntimeError("Путь не должен начинаться с '-' (интерпретируется как опция)")
    if "\\" in remainder:
        raise RuntimeError("Путь не должен использовать обратные слэши")
    if Path(remainder).is_absolute():
        raise RuntimeError("Путь должен быть относительным")
    if not _SAFE_PATH_CHARS_RE.match(remainder):
        raise RuntimeError("Путь содержит недопустимые символы")

    posix = PurePosixPath(remainder)
    if any(part in {"", ".", ".."} for part in posix.parts):
        raise RuntimeError("Путь содержит недопустимые сегменты")

    return f"{prefix}{remainder}" if prefix else remainder


def _write_github_output(**outputs: str | bool) -> None:
    """Append key/value pairs to ``GITHUB_OUTPUT`` if available."""

    path = resolve_github_path(
        os.getenv("GITHUB_OUTPUT"),
        allow_missing=True,
        description="GITHUB_OUTPUT",
    )
    if path is None:
        return

    try:
        with path.open("a", encoding="utf-8") as handle:
            for key, value in outputs.items():
                if isinstance(value, bool):
                    value = "true" if value else "false"
                handle.write(f"{key}={value}\n")
    except OSError as exc:  # pragma: no cover - extremely unlikely on CI
        print(f"::warning::Не удалось записать GITHUB_OUTPUT: {exc}", file=sys.stderr)

def _perform_https_request(
    url: str, headers: dict[str, str], timeout: float
) -> tuple[int, str, bytes]:
    """Return status, reason and payload for a validated HTTPS request."""

    parsed = urlparse(url)
    hostname = parsed.hostname
    if parsed.scheme != "https" or hostname is None:
        raise RuntimeError("Разрешены только HTTPS-запросы к GitHub API")
    if parsed.username or parsed.password:
        raise RuntimeError("URL GitHub API не должен содержать учетные данные")

    sanitised = parsed._replace(path=parsed.path or "/", fragment="")
    hostname = (sanitised.hostname or "").lower()
    if hostname not in _ALLOWED_GITHUB_API_HOSTS:
        raise RuntimeError(
            "Разрешены только запросы к api.github.com"
        )

    port = sanitised.port or 443
    if port != 443:
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname
    sanitised = sanitised._replace(netloc=netloc)

    request = Request(
        sanitised.geturl(),
        headers=headers,
        method="GET",
    )

    opener = build_opener(HTTPSHandler(context=ssl.create_default_context()))
    try:
        with opener.open(request, timeout=timeout) as response:
            status = int(getattr(response, "status", response.getcode()))
            reason = getattr(response, "reason", "") or ""
            payload = response.read()
    except HTTPError as exc:
        status = int(getattr(exc, "code", 0) or 0)
        reason = getattr(exc, "reason", "") or ""
        payload = exc.read() if hasattr(exc, "read") else b""
        return status, reason, payload
    except URLError as exc:
        message = getattr(exc.reason, "strerror", None) or exc.reason or exc
        raise RuntimeError(
            f"HTTP запрос {sanitised.geturl()} завершился ошибкой: {message}"
        ) from exc

    return status, reason, payload


def _api_request(url: str, token: str | None, timeout: float = 10.0) -> dict:
    """Perform a GitHub API request and return the decoded JSON payload."""

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "gptoss-review-workflow/1.0 (+https://github.com/averinaleks/bot)",
    }
    if token:
        headers["Authorization"] = f"token {token}"

    status, reason, payload = _perform_https_request(url, headers, timeout)
    if status >= 400:
        raise RuntimeError(
            f"HTTP запрос {url} завершился ошибкой: {status} {reason}"
        )

    try:
        return json.loads(payload.decode("utf-8"))
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("GitHub API вернул некорректный JSON") from exc


def _fetch_pull_request(
    repo: str, pr_number: str, token: str | None, timeout: float = 10.0
) -> PullRequestInfo:
    """Return base branch information for the requested pull request."""

    if not repo or not pr_number:
        raise RuntimeError("Не указан номер PR или репозиторий")

    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    payload = _api_request(url, token, timeout=timeout)

    base = payload.get("base") or {}
    base_sha = base.get("sha")
    base_ref = base.get("ref")
    if not isinstance(base_sha, str) or not isinstance(base_ref, str):
        raise RuntimeError("GitHub API не вернул base.sha/base.ref")

    return PullRequestInfo(base_sha=base_sha, base_ref=base_ref)


def _run_git(args: Sequence[str], *, capture_output: bool = False) -> GitCompletedProcess:
    """Execute a validated git command and return the completed process."""

    argv = list(args)
    if not argv or argv[0] != "git":
        raise RuntimeError("Команда должна начинаться с 'git'")
    if len(argv) < 2:
        raise RuntimeError("Не указана подкоманда git")

    subcommand = argv[1]
    if subcommand == "fetch":
        if len(argv) < 4:
            raise RuntimeError("Недостаточно аргументов для 'git fetch'")
        options = argv[2:-2]
        for option in options:
            if option not in _ALLOWED_GIT_OPTIONS:
                raise RuntimeError(f"Недопустимая опция git fetch: {option}")
        remote = argv[-2]
        if remote != "origin":
            raise RuntimeError("Разрешён только fetch из origin")
        _validate_git_ref(argv[-1])
    elif subcommand == "cat-file":
        if len(argv) != 4 or argv[2] != "-e":
            raise RuntimeError("Недопустимая команда git cat-file")
        object_spec = argv[3]
        commit, sep, suffix = object_spec.partition("^{")
        _validate_git_sha(commit)
        if sep and suffix.rstrip("}") != "commit":
            raise RuntimeError("Поддерживается только проверка существования коммита")
    elif subcommand == "diff":
        if len(argv) < 3:
            raise RuntimeError("Недостаточно аргументов для 'git diff'")
        range_arg = argv[2]
        if "..." not in range_arg:
            raise RuntimeError("Диапазон diff должен содержать '...'")
        base_sha, _, rest = range_arg.partition("...")
        if rest != "HEAD":
            raise RuntimeError("diff допускает сравнение только с HEAD")
        _validate_git_sha(base_sha)
        if "--" in argv[3:]:
            marker_index = argv.index("--", 3)
            for path in argv[marker_index + 1 :]:
                _validate_path_argument(path)
    else:
        raise RuntimeError(f"Недопустимая команда git: {subcommand}")

    return _execute_git(argv, capture_output=capture_output)


def _execute_git(argv: Sequence[str], *, capture_output: bool) -> GitCompletedProcess:
    """Spawn git via :mod:`asyncio` and return a completed process."""

    async def _run() -> GitCompletedProcess:
        stdout_pipe = asyncio.subprocess.PIPE if capture_output else None
        stderr_pipe = asyncio.subprocess.PIPE if capture_output else None

        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=stdout_pipe,
            stderr=stderr_pipe,
        )

        stdout_bytes, stderr_bytes = await process.communicate()
        if capture_output:
            stdout_text = (stdout_bytes or b"").decode("utf-8", "replace")
            stderr_text = (stderr_bytes or b"").decode("utf-8", "replace")
        else:
            stdout_text = None
            stderr_text = None

        returncode = process.returncode
        if returncode is None:  # pragma: no cover - process should always set returncode
            raise RuntimeError("git процесс завершился без кода возврата")
        if returncode != 0:
            raise GitCommandError(returncode, argv, stdout_text, stderr_text)

        return GitCompletedProcess(
            args=tuple(argv),
            returncode=returncode,
            stdout=stdout_text,
            stderr=stderr_text,
        )

    try:
        return asyncio.run(_run())
    except RuntimeError as exc:
        if "asyncio.run() cannot be called" not in str(exc):
            raise
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_run())
        finally:
            loop.close()


def _commit_exists(sha: str) -> bool:
    """Return ``True`` if *sha* resolves to a commit in the local repository."""

    try:
        _run_git(
            ["git", "cat-file", "-e", f"{_validate_git_sha(sha)}^{{commit}}"],
            capture_output=False,
        )
    except GitCommandError:
        return False
    return True


def _ensure_base_available(base_ref: str, base_sha: str) -> None:
    """Ensure that the base reference is available locally."""

    safe_ref = _validate_git_ref(base_ref)

    if _commit_exists(base_sha):
        return

    try:
        _run_git(["git", "fetch", "--no-tags", "origin", safe_ref])
    except GitCommandError as exc:
        if _commit_exists(base_sha):
            return
        raise RuntimeError(f"git fetch origin {base_ref} завершился с ошибкой") from exc


def _build_diff_args(base_sha: str, paths: Iterable[str]) -> list[str]:
    safe_sha = _validate_git_sha(base_sha)
    path_list = [_validate_path_argument(path) for path in paths]
    args = ["git", "diff", f"{safe_sha}...HEAD"]
    if path_list:
        args.append("--")
        args.extend(path_list)
    return args


def _compute_diff(
    base_sha: str,
    paths: Sequence[str],
    *,
    truncate: int,
) -> DiffComputation:
    """Return ``git diff`` output limited to ``truncate`` characters."""

    if not base_sha:
        raise RuntimeError("Не указан base_sha для diff")
    if truncate < 0:
        raise RuntimeError("Параметр truncate должен быть неотрицательным")

    try:
        result = _run_git(
            _build_diff_args(base_sha, paths),
            capture_output=True,
        )
    except GitCommandError as exc:
        raise RuntimeError("git diff завершился с ошибкой") from exc

    content = result.stdout or ""
    if truncate and len(content) > truncate:
        content = content[: truncate - 3] + "..."

    has_diff = bool(content.strip())
    return DiffComputation(content=content, has_diff=has_diff)


def prepare_diff(
    repo: str,
    pr_number: str,
    token: str | None,
    *,
    base_sha: str | None = None,
    base_ref: str | None = None,
    paths: Sequence[str] | None = None,
    truncate: int = 200_000,
) -> DiffComputation:
    """High level helper used by the GitHub Actions workflow."""

    if base_sha is None or base_ref is None:
        info = _fetch_pull_request(repo, pr_number, token)
        base_sha = base_sha or info.base_sha
        base_ref = base_ref or info.base_ref

    if base_sha is None or base_ref is None:
        raise RuntimeError("Не удалось определить base_sha/base_ref")

    safe_sha = _validate_git_sha(base_sha)
    _ensure_base_available(base_ref, safe_sha)
    monitored_paths: Sequence[str] = paths or [":(glob)**/*.py"]
    diff = _compute_diff(safe_sha, monitored_paths, truncate=truncate)
    if not isinstance(diff, DiffComputation):
        raise RuntimeError("git diff вернул неожиданный результат")
    return diff


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate diff for GPT-OSS review")
    parser.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY", ""))
    parser.add_argument("--pr-number", default=os.getenv("PR_NUMBER", ""))
    parser.add_argument("--token", default=os.getenv("GITHUB_TOKEN", ""))
    parser.add_argument("--output", default="diff.patch")
    parser.add_argument(
        "--path",
        dest="paths",
        action="append",
        help="Путь/паттерн для фильтрации diff (можно указать несколько раз)",
    )
    parser.add_argument("--base-sha")
    parser.add_argument("--base-ref")
    parser.add_argument("--truncate", type=int, default=200_000)

    args, unknown = parser.parse_known_args(argv)
    if unknown:
        raise ValueError("Неизвестные аргументы: " + " ".join(unknown))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
    except ValueError as exc:
        print(f"::warning::{exc}", file=sys.stderr)
        _write_github_output(has_diff=False)
        return 0
    except SystemExit as exc:  # pragma: no cover - argparse --help
        code = getattr(exc, "code", 0)
        if code not in (0, None):
            print(
                f"::warning::Парсер аргументов завершился с кодом {code}",
                file=sys.stderr,
            )
            _write_github_output(has_diff=False)
        return 0

    output_path = Path(args.output)
    paths = args.paths or None

    try:
        result = prepare_diff(
            args.repo,
            args.pr_number,
            args.token or None,
            base_sha=args.base_sha,
            base_ref=args.base_ref,
            paths=paths,
            truncate=args.truncate,
        )
    except RuntimeError as exc:
        print(f"::warning::{exc}", file=sys.stderr)
        _write_github_output(has_diff=False)
        return 0
    except Exception as exc:  # pragma: no cover - defensive guard
        print(
            f"::error::Неожиданная ошибка при подготовке diff: {exc}",
            file=sys.stderr,
        )
        _write_github_output(has_diff=False)
        return 0

    if not result.has_diff:
        print("::notice::Изменений для обзора не найдено", file=sys.stderr)
        try:
            if output_path.exists():
                output_path.unlink()
        except OSError:
            pass
        _write_github_output(has_diff=False)
        return 0

    try:
        output_path.write_text(result.content, encoding="utf-8")
    except OSError as exc:
        print(f"::warning::Не удалось записать diff: {exc}", file=sys.stderr)
        _write_github_output(has_diff=False)
        return 0

    _write_github_output(has_diff=True)
    return 0


def cli(argv: Sequence[str] | None = None) -> int:
    """Guard ``main`` so the workflow never fails because of SystemExit."""

    try:
        return main(argv)
    except SystemExit as exc:  # pragma: no cover - defensive guard
        code = exc.code
        if code not in (0, None):
            print(
                f"::warning::Скрипт завершился с кодом {code}. Возвращаю 0.",
                file=sys.stderr,
            )
            _write_github_output(has_diff=False)
        return 0
    except BaseException as exc:  # pragma: no cover - defensive guard
        print(
            f"::error::Критическое исключение в prepare_gptoss_diff: {exc}",
            file=sys.stderr,
        )
        _write_github_output(has_diff=False)
        return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(cli())
