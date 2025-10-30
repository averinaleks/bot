"""Helper for validating pull request readiness in the GPT-OSS workflow.

The previous implementation embedded the logic directly into the workflow
using a shell script with inline Python.  That approach made debugging
failures difficult – any unexpected API response caused the step to fail
hard with a non-zero exit code.  This module centralises the behaviour in a
tested Python entry point that always exits successfully while emitting clear
GitHub Actions annotations.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import ssl
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPSHandler, Request, build_opener

if __package__ in {None, ""}:
    package_root = Path(__file__).resolve().parent.parent
    package_root_str = str(package_root)
    if package_root_str not in sys.path:
        sys.path.insert(0, package_root_str)

try:  # pragma: no branch - ensures graceful fallback if helpers are missing
    from scripts.github_path_resolver import resolve_github_path as _resolve_github_path
except Exception:  # pragma: no cover - executed only when helpers missing
    try:
        from scripts.github_path_resolver_fallback import (
            resolve_github_path as _resolve_github_path,
        )
    except Exception:  # pragma: no cover - last resort, mirrors fallback logic
        def _resolve_github_path(
            raw_path: str | None,
            *,
            allow_missing: bool = False,
            description: str = "GitHub provided path",
        ) -> Path | None:
            if not raw_path:
                return None
            try:
                return Path(raw_path).resolve(strict=not allow_missing)
            except OSError as exc:
                print(
                    f"::warning::Не удалось определить {description} '{raw_path}': {exc}",
                    file=sys.stderr,
                )
                return None

resolve_github_path = _resolve_github_path

try:  # pragma: no cover - exercised when helpers ship without the module
    from scripts._filesystem import write_secure_text  # noqa: E402
except Exception:  # pragma: no cover - fallback for legacy commits
    import errno as _errno
    import os as _os
    import stat as _stat

    def write_secure_text(
        path: Path,
        content: str,
        *,
        append: bool = False,
        permissions: int = 0o600,
        encoding: str = "utf-8",
        dir_permissions: int | None = 0o700,
        allow_special_files: bool = False,
    ) -> None:
        if dir_permissions is not None:
            parent = path.parent
            if parent and parent != Path("."):
                parent.mkdir(parents=True, exist_ok=True, mode=dir_permissions)

        flags = _os.O_WRONLY | _os.O_CREAT
        flags |= _os.O_APPEND if append else _os.O_TRUNC

        fd = _os.open(path, flags, permissions)
        try:
            if hasattr(_os, "fchmod"):
                _os.fchmod(fd, permissions)
            else:  # pragma: no cover - Windows compatibility
                _os.chmod(path, permissions)
            info = _os.fstat(fd)
            if not _stat.S_ISREG(info.st_mode):
                if allow_special_files and _stat.S_ISFIFO(info.st_mode):
                    pass
                else:
                    raise OSError(_errno.EPERM, "target file must be a regular file")
            try:
                link_info = _os.lstat(path)
            except OSError:
                link_info = None
            if link_info is not None and _stat.S_ISLNK(link_info.st_mode):
                raise OSError(_errno.EPERM, "refusing to write through symlink")
            mode = "a" if append else "w"
            with _os.fdopen(fd, mode, encoding=encoding, closefd=False) as handle:
                handle.write(content)
        finally:
            _os.close(fd)


_DEFAULT_TIMEOUT = 10.0
_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(slots=True)
class PRStatus:
    """Result of evaluating whether a PR is eligible for review."""

    skip: bool
    head_sha: str
    notices: list[str]


def _write_github_output(skip: bool, head_sha: str) -> None:
    """Append outputs for downstream workflow steps if possible."""

    path = resolve_github_path(
        os.getenv("GITHUB_OUTPUT"),
        allow_missing=True,
        description="GITHUB_OUTPUT",
    )
    if path is None:
        return

    try:
        write_secure_text(
            path,
            f"skip={'true' if skip else 'false'}\nhead_sha={head_sha}\n",
            append=True,
            allow_special_files=True,
        )
    except OSError as exc:  # pragma: no cover - extremely rare on GitHub runners
        print(f"::warning::Не удалось записать GITHUB_OUTPUT: {exc}", file=sys.stderr)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check pull request status")
    parser.add_argument("--repo", default=os.getenv("REPOSITORY", ""))
    parser.add_argument("--pr-number", default=os.getenv("PR_NUMBER", ""))
    parser.add_argument("--token", default=os.getenv("GITHUB_TOKEN", ""))
    parser.add_argument("--timeout", type=float, default=_DEFAULT_TIMEOUT)

    args, unknown = parser.parse_known_args(argv)
    if unknown:
        raise ValueError("Неизвестные аргументы: " + " ".join(unknown))
    return args


def _fetch_pull_request(url: str, token: str, timeout: float) -> Any:
    """Return decoded JSON payload for the GitHub pull request API."""

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "gptoss-review-workflow/1.0 (+https://github.com/averinaleks/bot)",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, headers=headers)
    opener = build_opener(HTTPSHandler(context=ssl.create_default_context()))

    try:
        with opener.open(request, timeout=timeout) as response:
            payload = response.read()
    except HTTPError as exc:
        status = int(getattr(exc, "code", 0) or 0)
        reason = getattr(exc, "reason", "") or ""
        raise RuntimeError(
            f"HTTP запрос {url} завершился ошибкой: {status} {reason}"
        ) from exc
    except URLError as exc:
        message = getattr(exc.reason, "strerror", None) or getattr(exc.reason, "args", [None])[0]
        message = message or exc.reason or exc
        raise RuntimeError(f"HTTP запрос {url} завершился ошибкой: {message}") from exc
    except (TimeoutError, socket.timeout) as exc:
        message = str(exc) or "timed out"
        raise RuntimeError(f"HTTP запрос {url} завершился ошибкой: {message}") from exc
    except OSError as exc:
        message = getattr(exc, "strerror", None) or str(exc)
        raise RuntimeError(f"HTTP запрос {url} завершился ошибкой: {message}") from exc

    try:
        return json.loads(payload.decode("utf-8"))
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("GitHub API вернул некорректный JSON") from exc


def _evaluate_payload(payload: Any, repository: str) -> PRStatus:
    """Inspect GitHub API payload and determine whether to skip the review."""

    notices: list[str] = []
    head_sha = ""

    if not isinstance(payload, dict):
        return PRStatus(skip=True, head_sha="", notices=["Ответ GitHub API имеет неожиданный формат"])

    state = str(payload.get("state") or "").strip()
    draft_flag = payload.get("draft")
    locked_flag = payload.get("locked")

    head_repo = ""
    head_block = payload.get("head") or {}
    if isinstance(head_block, dict):
        repo_block = head_block.get("repo") or {}
        if isinstance(repo_block, dict):
            head_repo = str(repo_block.get("full_name") or "").strip()
        sha_value = head_block.get("sha")
        if isinstance(sha_value, str):
            head_sha = sha_value.strip()

    skip = False

    if state != "open":
        skip = True
        if state:
            notices.append(f"PR находится в состоянии {state!r}")
        else:
            notices.append("GitHub API не вернул статус PR")

    if isinstance(draft_flag, bool) and draft_flag:
        skip = True
        notices.append("PR находится в состоянии draft")

    if isinstance(locked_flag, bool) and locked_flag:
        skip = True
        notices.append("PR заблокирован для изменений")

    if not head_repo:
        skip = True
        notices.append("head-репозиторий недоступен (ветка могла быть удалена)")
    elif repository and head_repo.lower() != repository.lower():
        skip = True
        notices.append(f"PR создан из репозитория {head_repo}, ожидается {repository}")

    if not head_sha or not _SHA_RE.fullmatch(head_sha):
        skip = True
        if head_sha:
            notices.append("Получен некорректный SHA head-коммита PR")
        else:
            notices.append("Не удалось получить SHA head-коммита PR")

    return PRStatus(skip=skip, head_sha=head_sha if _SHA_RE.fullmatch(head_sha or "") else "", notices=notices)


def _build_api_url(repo: str, pr_number: str) -> str:
    if not repo or not pr_number:
        raise RuntimeError("Не указан номер PR или репозиторий")
    return f"https://api.github.com/repos/{repo}/pulls/{pr_number}"


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
    except ValueError as exc:
        print(f"::warning::{exc}", file=sys.stderr)
        _write_github_output(skip=True, head_sha="")
        return 0
    except SystemExit as exc:  # pragma: no cover - argparse --help
        code = getattr(exc, "code", 0)
        if code not in (0, None):
            print(
                f"::warning::Парсер аргументов завершился с кодом {code}",
                file=sys.stderr,
            )
        _write_github_output(skip=True, head_sha="")
        return 0

    repo = args.repo.strip()
    pr_number = str(args.pr_number).strip()
    if not repo or not pr_number:
        print("::notice::PR не найден – пропускаю запуск обзора.", file=sys.stderr)
        _write_github_output(skip=True, head_sha="")
        return 0

    try:
        url = _build_api_url(repo, pr_number)
        payload = _fetch_pull_request(url, args.token, args.timeout)
    except RuntimeError as exc:
        print(f"::warning::{exc}", file=sys.stderr)
        _write_github_output(skip=True, head_sha="")
        return 0
    except KeyboardInterrupt as exc:  # pragma: no cover - defensive guard
        print(
            f"::warning::Критическое исключение в check_pr_status: {exc}",
            file=sys.stderr,
        )
        _write_github_output(skip=True, head_sha="")
        return 0
    except Exception as exc:  # pragma: no cover - defensive guard
        print(
            f"::warning::Неожиданная ошибка при проверке PR: {exc}",
            file=sys.stderr,
        )
        _write_github_output(skip=True, head_sha="")
        return 0

    status = _evaluate_payload(payload, repo)

    if status.skip:
        if status.notices:
            print(
                "::notice::" + "; ".join(status.notices) + ". Пропускаю запуск обзора.",
                file=sys.stderr,
            )
        else:
            print("::notice::PR не подходит для обзора. Пропускаю запуск обзора.", file=sys.stderr)
    else:
        print("::debug::PR прошёл проверку статуса и источника.")

    _write_github_output(status.skip, status.head_sha)
    return 0


def cli(argv: list[str] | None = None) -> int:
    """CLI wrapper that shields the workflow from non-zero exits."""

    try:
        return main(argv)
    except SystemExit as exc:  # pragma: no cover - defensive guard
        code = exc.code
        if code not in (0, None):
            print(
                f"::warning::Скрипт завершился с кодом {code}. Возвращаю 0.",
                file=sys.stderr,
            )
            _write_github_output(skip=True, head_sha="")
        return 0
    except KeyboardInterrupt as exc:  # pragma: no cover - defensive guard
        print(
            f"::warning::Критическое исключение в check_pr_status: {exc}",
            file=sys.stderr,
        )
        _write_github_output(skip=True, head_sha="")
        return 0
    except Exception as exc:  # pragma: no cover - defensive guard
        print(
            f"::warning::Критическое исключение в check_pr_status: {exc}",
            file=sys.stderr,
        )
        _write_github_output(skip=True, head_sha="")
        return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(cli())

