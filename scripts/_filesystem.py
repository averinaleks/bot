"""Filesystem helpers used by command-line utilities."""

from __future__ import annotations

import os
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
    """

    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_APPEND if append else os.O_TRUNC

    # ``os.open`` applies ``permissions`` when creating the file.  If the file
    # already exists we still ensure its mode does not exceed the desired
    # permissions to guard against historic artifacts created with permissive
    # settings.
    fd = os.open(path, flags, permissions)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, permissions)
        else:  # pragma: no cover - Windows compatibility
            os.chmod(path, permissions)
        with os.fdopen(fd, "a" if append else "w", encoding=encoding, closefd=False) as handle:
            handle.write(content)
    finally:
        os.close(fd)
