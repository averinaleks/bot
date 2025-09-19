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
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from urllib import error, request
from urllib.parse import urlparse


@dataclass
class PullRequestInfo:
    """Subset of pull request metadata required to prepare a diff."""

    base_sha: str
    base_ref: str


@dataclass
class DiffComputation:
    """Result of running ``git diff`` for the requested changes."""

    content: str
    has_diff: bool


def _write_github_output(**outputs: str | bool) -> None:
    """Append key/value pairs to ``GITHUB_OUTPUT`` if available."""

    output_path = os.getenv("GITHUB_OUTPUT")
    if not output_path:
        return

    try:
        with open(output_path, "a", encoding="utf-8") as handle:
            for key, value in outputs.items():
                if isinstance(value, bool):
                    value = "true" if value else "false"
                handle.write(f"{key}={value}\n")
    except OSError as exc:  # pragma: no cover - extremely unlikely on CI
        print(f"::warning::Не удалось записать GITHUB_OUTPUT: {exc}", file=sys.stderr)

def _api_request(url: str, token: str | None, timeout: float = 10.0) -> dict:
    """Perform a GitHub API request and return the decoded JSON payload."""

    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise RuntimeError("Разрешены только HTTPS-запросы к GitHub API")

    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    req = request.Request(url, headers=headers)

    payload: bytes
    try:
        with request.urlopen(req, timeout=timeout) as response:
            payload = response.read()
    except error.HTTPError as exc:
        raise RuntimeError(
            f"HTTP запрос {url} завершился ошибкой: {exc.code} {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(f"HTTP запрос {url} завершился ошибкой: {exc}") from exc
    except error.URLError as exc:
        message = getattr(exc, "reason", exc)
        raise RuntimeError(f"HTTP запрос {url} завершился ошибкой: {message}") from exc

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("GitHub API вернул ответ в неизвестной кодировке") from exc

    try:
        return json.loads(text)
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
    if not base_sha or not base_ref:
        raise RuntimeError("GitHub API не вернул base.sha/base.ref")

    return PullRequestInfo(base_sha=base_sha, base_ref=base_ref)


def _run_git(args: Sequence[str], *, capture_output: bool = False) -> subprocess.CompletedProcess:
    """Execute a git command and return the completed process."""

    return subprocess.run(  # noqa: PLW1510 - we want ``check`` to raise
        args,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def _ensure_base_available(base_ref: str) -> None:
    """Ensure that the base reference is available locally."""

    if not base_ref:
        raise RuntimeError("Не указан base_ref для diff")

    try:
        _run_git(["git", "fetch", "--no-tags", "origin", base_ref])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"git fetch origin {base_ref} завершился с ошибкой") from exc


def _build_diff_args(base_sha: str, paths: Iterable[str]) -> list[str]:
    args = ["git", "diff", f"{base_sha}...HEAD"]
    path_list = list(paths)
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

    try:
        result = _run_git(
            _build_diff_args(base_sha, paths),
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("git diff завершился с ошибкой") from exc

    content = result.stdout
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

    _ensure_base_available(base_ref)
    monitored_paths: Sequence[str] = paths or [":(glob)**/*.py"]
    return _compute_diff(base_sha, monitored_paths, truncate=truncate)


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
