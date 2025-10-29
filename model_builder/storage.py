"""Helpers for persisting and validating ModelBuilder artifacts."""

from __future__ import annotations

import errno
import importlib.metadata
import json
import os
import platform
import re
import stat
import sys
import time
from pathlib import Path

from bot.utils_loader import require_utils
from security import create_joblib_stub, set_model_dir
from services.logging_utils import sanitize_log_value

_utils = require_utils(
    "ensure_writable_directory",
    "logger",
)

ensure_writable_directory = _utils.ensure_writable_directory
logger = _utils.logger

MODEL_DIR = ensure_writable_directory(
    Path(os.getenv("MODEL_DIR", ".")),
    description="моделей",
    fallback_subdir="trading_bot_models",
).resolve()
set_model_dir(MODEL_DIR)

MODEL_FILE: str | Path | None = os.environ.get("MODEL_FILE", "model.pkl")

JOBLIB_AVAILABLE = True
try:  # pragma: no cover - optional dependency
    import joblib  # type: ignore
except Exception as exc:  # pragma: no cover - stub for tests
    JOBLIB_AVAILABLE = False
    logger.warning(
        "Не удалось импортировать joblib: %s. Сериализация моделей будет отключена.",
        exc,
    )
    joblib = create_joblib_stub(
        "joblib недоступен: установите зависимость для сохранения/загрузки моделей"
    )
    sys.modules.setdefault("joblib", joblib)


def _is_within_directory(path: Path, directory: Path) -> bool:
    """Return ``True`` if ``path`` is located within ``directory``."""

    try:
        path.resolve(strict=False).relative_to(directory.resolve(strict=False))
    except ValueError:
        return False
    return True


def _resolve_model_artifact(path_value: str | Path | None) -> Path:
    """Return a sanitised model path confined to :data:`MODEL_DIR`."""

    if path_value is None:
        raise ValueError("model path is not set")
    candidate = Path(path_value)
    if not candidate.parts or candidate == Path("."):
        raise ValueError("model path is empty")
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
    else:
        resolved = (MODEL_DIR / candidate).resolve(strict=False)
    if not _is_within_directory(resolved, MODEL_DIR):
        raise ValueError("model path escapes MODEL_DIR")
    if resolved.exists():
        if resolved.is_symlink():
            raise ValueError("model path must not be a symlink")
        if not resolved.is_file():
            raise ValueError("model path must reference a regular file")
    return resolved


_SYMBOL_STRIP_RE = re.compile(r"[^A-Za-z0-9_.-]")
_SYMBOL_HYPHEN_RE = re.compile(r"[-\s]+")


def _symbol_directory(symbol: str) -> Path:
    """Return a sanitised per-symbol directory within :data:`MODEL_DIR`."""

    if not isinstance(symbol, str):
        raise ValueError("symbol must be a string")

    safe = Path(symbol).name
    safe = _SYMBOL_HYPHEN_RE.sub("-", safe)
    safe = _SYMBOL_STRIP_RE.sub("", safe)
    safe = safe.strip("._-")[:64]
    if not safe:
        raise ValueError("symbol resolves to an empty directory name")

    candidate = MODEL_DIR / safe
    resolved = candidate.resolve(strict=False)
    if not _is_within_directory(resolved, MODEL_DIR):
        raise ValueError("symbol directory escapes MODEL_DIR")
    if resolved.exists():
        if resolved.is_symlink():
            raise ValueError("symbol directory must not be a symlink")
        if not resolved.is_dir():
            raise ValueError("symbol directory must reference a directory")
    return resolved


def _safe_model_file_path() -> Path | None:
    """Return a validated path for ``MODEL_FILE`` or ``None`` if invalid."""

    try:
        return _resolve_model_artifact(MODEL_FILE)
    except ValueError as exc:
        logger.warning(
            "Refusing to use MODEL_FILE %s: %s",
            sanitize_log_value("<unset>" if MODEL_FILE is None else str(MODEL_FILE)),
            exc,
        )
        return None


def save_artifacts(model: object, symbol: str, meta: dict) -> Path:
    """Сохранить модель и метаданные в каталог артефактов."""

    base_dir = _symbol_directory(symbol)
    timestamp = str(int(time.time()))
    target_dir = base_dir / timestamp
    target_dir.mkdir(parents=True, exist_ok=True)
    if target_dir.is_symlink() or not target_dir.is_dir():
        raise ValueError("artifact directory could not be created safely")
    if not _is_within_directory(target_dir.resolve(strict=False), MODEL_DIR):
        raise ValueError("artifact directory escapes MODEL_DIR")
    model_file = target_dir / "model.pkl"
    if JOBLIB_AVAILABLE:
        with _open_secure_artifact(model_file, "wb") as handle:
            joblib.dump(model, handle)
    else:
        logger.warning(
            "joblib недоступен, модель %s не будет сохранена на диск", symbol
        )

    try:
        head_file = Path(".git/HEAD")
        if head_file.is_file():
            ref = head_file.read_text().strip()
            if ref.startswith("ref:"):
                ref_path = Path(".git") / ref.split()[1]
                code_version = ref_path.read_text().strip()
            else:
                code_version = ref
        else:
            code_version = "unknown"
    except Exception:
        code_version = "unknown"

    try:
        pip_freeze = sorted(
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in importlib.metadata.distributions()
        )
    except Exception:
        pip_freeze = []

    meta_env = {
        "code_version": code_version,
        "python_version": platform.python_version(),
        "pip_freeze": pip_freeze,
        "platform": platform.platform(),
    }
    meta_all = {**meta_env, **(meta or {})}
    meta_path = target_dir / "meta.json"
    with _open_secure_artifact(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta_all, handle, ensure_ascii=False, indent=2)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except (AttributeError, OSError):  # pragma: no cover - fsync may be unsupported
            pass

    return target_dir


__all__ = [
    "MODEL_DIR",
    "MODEL_FILE",
    "JOBLIB_AVAILABLE",
    "joblib",
    "_is_within_directory",
    "_resolve_model_artifact",
    "_safe_model_file_path",
    "save_artifacts",
]


def _open_secure_artifact(
    path: Path, mode: str, *, encoding: str | None = None
):
    """Return a file object that refuses to follow symlinks when writing artifacts."""

    if "w" not in mode or any(flag in mode for flag in "ax+"):
        raise ValueError("_open_secure_artifact supports only overwrite write modes")
    if "b" in mode and encoding is not None:
        raise ValueError("binary mode cannot be combined with text encoding")

    nofollow = getattr(os, "O_NOFOLLOW", 0)

    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_TRUNC
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
    )
    if nofollow:
        flags |= nofollow

    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:  # pragma: no cover - platform specific errno
        if exc.errno in (errno.ELOOP, errno.EPERM):
            raise RuntimeError(
                f"Отказ сохранения артефакта модели через символьную ссылку: {path}"
            ) from exc
        raise

    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(
                "Артефакт модели должен сохраняться в обычный файл"
            )

        try:
            link_info = os.lstat(path)
        except OSError:
            link_info = None
        if link_info is not None and stat.S_ISLNK(link_info.st_mode):
            raise RuntimeError(
                f"Отказ сохранения артефакта модели через символьную ссылку: {path}"
            )

        return os.fdopen(fd, mode, encoding=encoding)
    except Exception:
        os.close(fd)
        raise
