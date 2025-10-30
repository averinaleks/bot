"""Filesystem helpers used by command-line utilities."""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path
from typing import Final

_DEFAULT_PERMISSIONS: Final[int] = 0o600


def write_secure_text(
    path: Path,
    content: str,
    *,
    append: bool = False,
    permissions: int = _DEFAULT_PERMISSIONS,
    encoding: str = "utf-8",
    dir_permissions: int | None = 0o700,
    allow_special_files: bool = False,
) -> None:
    """Write ``content`` to ``path`` using restrictive file permissions.

    CodeQL flags file writes that rely on the process ``umask`` because they can
    inadvertently create world-readable artifacts.  The helper enforces an
    explicit permission mask and uses :func:`os.open` so that even when the
    target file is created by the helper its mode never exceeds ``permissions``.

    Parameters
    ----------
    path:
        Destination file to write.
    content:
        Text payload written to ``path``.
    append:
        When ``True`` content is appended to the existing file.  Otherwise the
        file is truncated prior to writing.
    permissions:
        POSIX file mode applied when creating the file.  Defaults to ``0o600``
        which keeps the file private to the current user.
    encoding:
        Text encoding used when converting ``content`` to bytes.
    dir_permissions:
        Optional mode applied when creating parent directories.  ``None`` skips
        directory creation and leaves existing permissions untouched.
    allow_special_files:
        When ``True`` the helper tolerates writing to special files such as
        FIFOs.  This is primarily intended for GitHub Actions command files
        (e.g. ``GITHUB_OUTPUT``) which are provided as named pipes on hosted
        runners.  The safeguard against symlinks remains in place to avoid
        TOCTOU attacks.
    """

    if dir_permissions is not None:
        parent = path.parent
        if parent and parent != Path(".") and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True, mode=dir_permissions)

    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_APPEND if append else os.O_TRUNC

    # ``os.open`` applies ``permissions`` when creating the file.  If the file
    # already exists we still ensure its mode does not exceed the desired
    # permissions to guard against historic artifacts created with permissive
    # settings.
    fd = os.open(path, flags, permissions)
    try:
        info = os.fstat(fd)

        # GitHub Actions exposes ``GITHUB_OUTPUT`` as either a FIFO, UNIX socket,
        # or character device depending on the runner version.  The caller can
        # opt-in to writing to these special files by setting
        # ``allow_special_files`` which we use for workflow command files.  Keep
        # treating every other special file as an error to avoid accidentally
        # writing sensitive information to unexpected locations.
        is_regular_file = stat.S_ISREG(info.st_mode)
        is_allowed_special = False
        if allow_special_files and not is_regular_file:
            is_allowed_special = any(
                checker(info.st_mode)
                for checker in (
                    stat.S_ISFIFO,
                    getattr(stat, "S_ISSOCK", lambda mode: False),
                    getattr(stat, "S_ISCHR", lambda mode: False),
                )
            )

        if not is_regular_file and not is_allowed_special:
            raise OSError(errno.EPERM, "target file must be a regular file")

        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, permissions)
            else:  # pragma: no cover - Windows compatibility
                os.chmod(path, permissions)
        except OSError:
            if not is_allowed_special:
                raise

        try:
            link_info = os.lstat(path)
        except OSError:
            link_info = None
        if link_info is not None and stat.S_ISLNK(link_info.st_mode):
            raise OSError(errno.EPERM, "refusing to write through symlink")

        with os.fdopen(fd, "a" if append else "w", encoding=encoding, closefd=False) as handle:
            handle.write(content)
    finally:
        os.close(fd)
