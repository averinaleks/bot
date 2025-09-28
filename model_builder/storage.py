"""Helpers for persisting and validating ModelBuilder artifacts."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
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
    """Return ``True`` when ``path`` is confined to ``directory``."""

    base = directory.resolve(strict=False)
    target = path.resolve(strict=False)
    try:
        common = os.path.commonpath([str(base), str(target)])
    except ValueError:
        return False
    return common == str(base)


def _safe_join(directory: Path, *parts: str) -> Path:
    """Join ``parts`` to ``directory`` ensuring the result stays within ``directory``."""

    candidate = directory.joinpath(*parts)
    resolved = candidate.resolve(strict=False)
    if not _is_within_directory(resolved, directory):
        raise ValueError(f"path {resolved} escapes base directory {directory}")
    return resolved


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

    timestamp = str(int(time.time()))
    target_dir = MODEL_DIR / symbol / timestamp
    target_dir.mkdir(parents=True, exist_ok=True)
    model_file = target_dir / "model.pkl"
    if JOBLIB_AVAILABLE:
        joblib.dump(model, model_file)
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
    with open(target_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_all, f, ensure_ascii=False, indent=2)

    return target_dir


__all__ = [
    "MODEL_DIR",
    "MODEL_FILE",
    "JOBLIB_AVAILABLE",
    "joblib",
    "_is_within_directory",
    "_safe_join",
    "_resolve_model_artifact",
    "_safe_model_file_path",
    "save_artifacts",
]
