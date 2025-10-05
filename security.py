"""Security hardening helpers for optional third-party integrations.

This module centralises mitigations for known vulnerabilities reported by
Trivy so that the codebase remains safe without dropping optional features.
"""
from __future__ import annotations

import contextlib
import errno
import functools
import hmac
import hashlib
import ipaddress
import logging
import os
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Tuple, cast

from packaging.version import InvalidVersion, Version


# Default model directory used for signature checks.
#
# ``services.model_builder_service`` updates this value during import by calling
# :func:`set_model_dir`.  Defining the default eagerly avoids importing the
# service from this module which previously introduced a cyclic dependency that
# CodeQL rightfully flagged as unsafe.  The setter keeps the behaviour of using
# the real ``MODEL_DIR`` when the service is available while allowing standalone
# usage (e.g. in tests) to customise the directory explicitly.
MODEL_DIR: Path = Path(".").resolve(strict=False)


def set_model_dir(path: Path | str) -> None:
    """Update the global model directory used for signature enforcement."""

    global MODEL_DIR
    MODEL_DIR = Path(path).resolve(strict=False)


try:  # joblib is optional in some environments
    import joblib  # type: ignore
    from joblib import numpy_pickle as _joblib_numpy_pickle  # type: ignore
except Exception:  # pragma: no cover - joblib not available
    joblib = None  # type: ignore[assignment]
    _joblib_numpy_pickle = None  # type: ignore[assignment]
    _joblib_numpy_pickle_compat = None  # type: ignore[assignment]
else:  # pragma: no cover - exercised in integration tests
    try:
        from joblib import numpy_pickle_compat as _joblib_numpy_pickle_compat  # type: ignore
    except Exception:  # pragma: no cover - optional compatibility helpers
        _joblib_numpy_pickle_compat = None  # type: ignore[assignment]


_DEFAULT_SAFE_JOBLIB_MODULES: Tuple[str, ...] = (
    "__main__",
    "builtins",
    "collections",
    "collections.abc",
    "contextlib",
    "datetime",
    "decimal",
    "functools",
    "itertools",
    "math",
    "numbers",
    "operator",
    "pathlib",
    "pickle",
    "types",
    "typing",
    "weakref",
    "numpy",
    "pandas",
    "joblib",
    "scipy",
    "sklearn",
    "torch",
    "bot",
    "services",
    "tests",
    "transformers",
    "statsmodels",
    "lightgbm",
    "xgboost",
    "catboost",
    "pytz",
    "dateutil",
)


class ArtifactDeserializationError(RuntimeError):
    """Raised when a joblib artefact cannot be safely deserialised."""


def create_joblib_stub(message: str) -> ModuleType:
    """Return a ``joblib``-like module that refuses to serialise artefacts."""

    def _unavailable(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(message)

    stub = ModuleType("joblib")
    stub.dump = _unavailable  # type: ignore[attr-defined]
    stub.load = _unavailable  # type: ignore[attr-defined]
    return stub


def _normalise_allowed_modules(allowed: Iterable[str] | None) -> Tuple[str, ...]:
    """Return a deduplicated tuple of allowed module prefixes."""

    base: list[str] = list(_DEFAULT_SAFE_JOBLIB_MODULES)
    if allowed is not None:
        for prefix in allowed:
            if prefix:
                base.append(prefix)
    # Ensure fundamental builtins are always permitted
    base.extend(["builtins", "collections", "collections.abc"])
    seen: set[str] = set()
    normalised: list[str] = []
    for prefix in base:
        if prefix not in seen:
            seen.add(prefix)
            normalised.append(prefix)
    return tuple(normalised)


def _module_is_allowed(module: str, allowed: Tuple[str, ...]) -> bool:
    return any(
        module == prefix or module.startswith(f"{prefix}.")
        for prefix in allowed
    )


@contextlib.contextmanager
def _restricted_joblib_unpicklers(allowed: Tuple[str, ...]):
    """Temporarily harden joblib's unpicklers with an allow-list."""

    if joblib is None or _joblib_numpy_pickle is None:  # pragma: no cover - defensive
        raise RuntimeError(
            "joblib недоступен: установите зависимость для работы с артефактами"
        )

    original_unpickler: type[Any] = _joblib_numpy_pickle.NumpyUnpickler
    compat_unpickler: type[Any] | None = None
    if _joblib_numpy_pickle_compat is not None:
        compat_unpickler = getattr(
            _joblib_numpy_pickle_compat, "NumpyUnpickler", None
        )

    class _RestrictedUnpickler(original_unpickler):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            self._allowed_modules = allowed
            super().__init__(*args, **kwargs)

        def find_class(self, module, name):  # type: ignore[override]
            if not _module_is_allowed(module, self._allowed_modules):
                logger.error(
                    "Отказ от десериализации: модуль %s.%s не входит в список доверенных",
                    module,
                    name,
                )
                raise ArtifactDeserializationError(
                    f"Refusing to load object from disallowed module {module}.{name}"
                )
            return super().find_class(module, name)

    if compat_unpickler is not None:
        compat_unpickler_cls = cast(type[Any], compat_unpickler)

        class _RestrictedCompatUnpickler(compat_unpickler_cls):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                self._allowed_modules = allowed
                super().__init__(*args, **kwargs)

            def find_class(self, module, name):  # type: ignore[override]
                if not _module_is_allowed(module, self._allowed_modules):
                    logger.error(
                        "Отказ от десериализации (compat): модуль %s.%s не доверен",
                        module,
                        name,
                    )
                    raise ArtifactDeserializationError(
                        f"Refusing to load object from disallowed module {module}.{name}"
                    )
                return super().find_class(module, name)

    _joblib_numpy_pickle.NumpyUnpickler = _RestrictedUnpickler
    if compat_unpickler is not None:
        _joblib_numpy_pickle_compat.NumpyUnpickler = _RestrictedCompatUnpickler  # type: ignore[attr-defined]
    try:
        yield
    finally:  # pragma: no cover - ensure restoration even on failure
        _joblib_numpy_pickle.NumpyUnpickler = original_unpickler
        if compat_unpickler is not None:
            _joblib_numpy_pickle_compat.NumpyUnpickler = compat_unpickler  # type: ignore[attr-defined]


def safe_joblib_load(
    source: Any,
    *,
    allowed_modules: Iterable[str] | None = None,
) -> Any:
    """Load a joblib artifact with module-level deserialization restrictions."""

    module = joblib or cast(ModuleType | None, sys.modules.get("joblib"))
    if module is None:  # pragma: no cover - handled in optional dependency tests
        raise RuntimeError(
            "joblib недоступен: установите зависимость для работы с артефактами"
        )

    prefixes = _normalise_allowed_modules(allowed_modules)

    path_source: Path | None = None
    if isinstance(source, (str, os.PathLike)):
        path_source = Path(source)
        try:
            # Refuse to follow symlinks or load non-regular files to prevent
            # filesystem tricks (e.g. symlink swaps) from bypassing
            # signature checks enforced by callers.
            if path_source.is_symlink():
                raise RuntimeError(
                    f"Отказ от загрузки артефакта модели через симлинк: {path_source}"
                )
            resolved = path_source.resolve(strict=True)
        except FileNotFoundError:
            # Mirror joblib's behaviour for missing files so callers receive a
            # familiar exception type.
            raise
        except OSError as exc:
            raise RuntimeError(
                f"Не удалось получить путь артефакта модели {path_source}: {exc}"
            ) from exc
        if not resolved.is_file():
            raise RuntimeError(
                f"Артефакт модели {resolved} не является обычным файлом"
            )
        if not _is_within_directory(resolved, MODEL_DIR):
            raise RuntimeError(
                f"Артефакт модели {resolved} выходит за пределы MODEL_DIR"
            )
        source = str(resolved)

    if _joblib_numpy_pickle is None:
        loader = getattr(module, "load", None)
        if loader is None:  # pragma: no cover - stubbed joblib missing loader
            raise RuntimeError(
                "joblib.numpy_pickle недоступен: установите зависимость для работы с артефактами"
            )
        try:
            return loader(source)
        except Exception as exc:  # pragma: no cover - unexpected stub failure
            raise ArtifactDeserializationError(
                f"Ошибка загрузки артефакта модели: {exc}"
            ) from exc

    loader = getattr(_joblib_numpy_pickle, "load", None)
    if loader is None:  # pragma: no cover - extremely unlikely for supported joblib
        raise RuntimeError("Текущая версия joblib не предоставляет numpy_pickle.load")

    try:
        with _restricted_joblib_unpicklers(prefixes):
            # ``numpy_pickle.load`` honours the patched unpickler ensuring the
            # allow-list above is enforced even when CodeQL tracks lower-level
            # deserialisation helpers.
            return loader(source)
    except ArtifactDeserializationError:
        raise
    except Exception as exc:  # pragma: no cover - unexpected joblib failure
        if exc.__class__.__name__ == "UnpicklingError":
            raise ArtifactDeserializationError(str(exc)) from exc
        raise

def _is_within_directory(path: Path, directory: Path) -> bool:
    """Return True if `path` is located within `directory`."""
    try:
        path.resolve(strict=False).relative_to(directory.resolve(strict=False))
    except ValueError:
        return False
    return True

logger = logging.getLogger(__name__)


_MIN_RAY_VERSION = Version("2.49.2")
_MODEL_STATE_HMAC_ENV = "MODEL_STATE_HMAC_KEY"
_MODEL_STATE_SIG_SUFFIX = ".sig"


@functools.lru_cache
def _get_model_state_hmac_key() -> bytes | None:
    """Return the HMAC key for model state integrity checks."""

    key = os.getenv(_MODEL_STATE_HMAC_ENV)
    if not key:
        return None
    try:
        return key.encode("utf-8")
    except Exception:  # pragma: no cover - defensive, UTF-8 always supported
        logger.warning("Не удалось интерпретировать ключ из %s", _MODEL_STATE_HMAC_ENV)
        return None


def _signature_path(path: Path) -> Path:
    return path.with_name(path.name + _MODEL_STATE_SIG_SUFFIX)


def _calculate_hmac(path: Path) -> str | None:
    key = _get_model_state_hmac_key()
    if key is None:
        return None
    try:
        if path.is_symlink():
            logger.warning(
                "Отказ от вычисления подписи модели %s: путь является символьной ссылкой",
                path,
            )
            return None
        if not path.exists():
            logger.warning(
                "Отказ от вычисления подписи модели %s: файл отсутствует",
                path,
            )
            return None
        if not path.is_file():
            logger.warning(
                "Отказ от вычисления подписи модели %s: ожидается обычный файл",
                path,
            )
            return None
    except OSError as exc:  # pragma: no cover - редкие ошибки ФС
        logger.warning("Не удалось получить информацию о файле модели %s: %s", path, exc)
        return None
    digest = hmac.new(key, digestmod=hashlib.sha256)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic_signature(target: Path, data: str) -> None:
    """Write *data* to *target* using restrictive file permissions."""

    encoded = data.encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    flags |= getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow

    fd = os.open(target, flags, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise OSError(errno.EPERM, "signature file must be a regular file")

        total = 0
        length = len(encoded)
        while total < length:
            written = os.write(fd, encoded[total:])
            if written == 0:
                raise OSError(errno.EIO, "failed to write signature contents")
            total += written
        with contextlib.suppress(OSError):
            os.fsync(fd)
    finally:
        os.close(fd)


def write_model_state_signature(path: Path) -> None:
    """Persist a keyed digest for *path* if signing is configured."""

    signature = _calculate_hmac(path)
    if signature is None:
        return
    sig_path = _signature_path(path)
    tmp_path = sig_path.with_suffix(sig_path.suffix + ".tmp")
    sig_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if sig_path.is_symlink():
            logger.warning(
                "Удаляем символьную ссылку подписи модели %s перед перезаписью",
                sig_path,
            )
            sig_path.unlink()
        elif sig_path.exists() and not sig_path.is_file():
            logger.warning(
                "Подпись модели %s не является обычным файлом: запись пропущена",
                sig_path,
            )
            return
    except OSError as exc:  # pragma: no cover - крайне редкие ошибки ФС
        logger.warning("Не удалось подготовить файл подписи %s: %s", sig_path, exc)
        return

    try:
        _write_atomic_signature(tmp_path, signature)
        os.replace(tmp_path, sig_path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()


def verify_model_state_signature(path: Path) -> bool:
    """Return ``True`` if *path* has a valid integrity signature."""

    key = _get_model_state_hmac_key()
    if key is None:
        return True
    sig_path = _signature_path(path)

    model_dir = Path(MODEL_DIR)
    path_resolved = path.resolve(strict=False)
    sig_resolved = sig_path.resolve(strict=False)

    try:
        if path.is_symlink():
            logger.warning(
                "Отказ от проверки подписи модели %s: путь является символьной ссылкой",
                path,
            )
            return False
        if not path.exists():
            logger.warning(
                "Отказ от проверки подписи модели %s: файл отсутствует",
                path,
            )
            return False
        if not path.is_file():
            logger.warning(
                "Отказ от проверки подписи модели %s: ожидается обычный файл",
                path,
            )
            return False
    except OSError as exc:  # pragma: no cover - редкие ошибки ФС
        logger.warning("Не удалось получить информацию о файле модели %s: %s", path, exc)
        return False

    if _is_within_directory(path_resolved, model_dir):
        # Ensure signatures for models stored in MODEL_DIR never escape it
        if not _is_within_directory(sig_resolved, model_dir):
            logger.warning(
                "Отказ от проверки подписи модели %s: подпись вне MODEL_DIR (%s)",
                path,
                sig_path,
            )
            return False
    else:
        # For temporary or test directories, require the signature next to the model
        if sig_resolved.parent != path_resolved.parent:
            logger.warning(
                "Отказ от проверки подписи модели %s: подпись должна находиться рядом с файлом (%s)",
                path,
                sig_path,
            )
            return False
    if sig_path.is_symlink():
        logger.warning(
            "Отказ от проверки подписи модели %s: путь является символьной ссылкой",
            sig_path,
        )
        return False
    if not sig_path.exists():
        logger.warning(
            "Отсутствует подпись файла модели %s: загрузка отклонена", path
        )
        return False
    if not sig_path.is_file():
        logger.warning(
            "Отказ от проверки подписи модели %s: ожидается обычный файл", sig_path
        )
        return False
    try:
        expected = sig_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning(
            "Отсутствует подпись файла модели %s: загрузка отклонена", path
        )
        return False
    except OSError as exc:  # pragma: no cover - редкие ошибки чтения
        logger.warning(
            "Не удалось прочитать подпись модели %s: %s", sig_path, exc
        )
        return False
    actual = _calculate_hmac(path)
    if actual is None:
        return False
    if not hmac.compare_digest(actual, expected):
        logger.error(
            "Подпись модели %s не совпадает: ожидалось %s, получено %s",
            path,
            expected,
            actual,
        )
        return False
    return True


_LOOPBACK_HOST = "127.0.0.1"


def _sanitize_dashboard_host(raw_host: Any) -> str:
    """Return a safe dashboard host limited to the loopback interface."""

    if raw_host is None:
        return _LOOPBACK_HOST

    host_text = str(raw_host).strip()
    if not host_text:
        return _LOOPBACK_HOST

    try:
        parsed = ipaddress.ip_address(host_text)
    except ValueError:
        if host_text.lower() == "localhost":
            return "localhost"
        logger.warning(
            "Небезопасное значение dashboard_host %s: устанавливаем %s",
            host_text,
            _LOOPBACK_HOST,
        )
        return _LOOPBACK_HOST

    if parsed.is_loopback:
        return host_text

    logger.warning(
        "Небезопасное значение dashboard_host %s: устанавливаем %s",
        host_text,
        _LOOPBACK_HOST,
    )
    return _LOOPBACK_HOST


def apply_ray_security_defaults(params: dict[str, Any]) -> dict[str, Any]:
    """Return Ray initialisation kwargs hardened against CVE-2023-48022.

    The ShadowRay advisory (CVE-2023-48022) abuses the Ray dashboard job
    submission API which is enabled by default.  We explicitly disable the
    dashboard and ensure the process only binds to the loopback interface.
    """

    os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
    os.environ.setdefault("RAY_JOB_ALLOWLIST", "")
    hardened = dict(params)
    hardened.setdefault("include_dashboard", False)
    hardened["dashboard_host"] = _sanitize_dashboard_host(
        hardened.get("dashboard_host", _LOOPBACK_HOST)
    )
    return hardened


def ensure_minimum_ray_version(ray_module: ModuleType) -> None:
    """Raise an error if *ray_module* is older than the patched release."""

    version_str = getattr(ray_module, "__version__", "")
    try:
        parsed = Version(version_str)
    except InvalidVersion:
        logger.warning(
            "Не удалось определить версию Ray (%s): пропускаем проверку", version_str
        )
        return

    if parsed < _MIN_RAY_VERSION:
        raise RuntimeError(
            "Установлена уязвимая версия Ray %s. Обновите пакет до %s или новее "
            "для устранения CVE-2023-48022." % (version_str, _MIN_RAY_VERSION)
        )


_MLFLOW_DISABLED_ATTRS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("pyfunc",), "load_model"),
    (("sklearn",), "load_model"),
    (("pytorch",), "load_model"),
    (("tensorflow",), "load_model"),
    (("lightgbm",), "load_model"),
    (("xgboost",), "load_model"),
    (("catboost",), "load_model"),
    (("pmdarima",), "load_model"),
    (("recipes",), "load_recipe"),
)


def _disable_callable(target: Any, qualname: str) -> Any:
    message = (
        "Загрузка моделей MLflow отключена по соображениям безопасности. "
        "Подробности см. в CVE-2024-37052…CVE-2024-37060."
    )

    def _blocked(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(message)

    if callable(target):
        logger.info("Отключена уязвимая функция MLflow: %s", qualname)
        return _blocked
    return None


def harden_mlflow(mlflow_module: ModuleType) -> None:
    """Disable model-loading entry points that lead to remote code execution.

    MLflow 3.4.0 содержит серию RCE-уязвимостей (CVE-2024-37052…37060). Мы не
    загружаем сторонние модели внутри проекта, поэтому безопасно блокировать
    соответствующие API, оставив журналы и экспорт в рабочем состоянии.
    """

    for path, attr in _MLFLOW_DISABLED_ATTRS:
        parent: Any = mlflow_module
        qualname = ["mlflow"]
        for name in path:
            try:
                parent = getattr(parent, name)
            except AttributeError:
                break
            except Exception as exc:  # pragma: no cover - optional deps may be missing
                logger.debug("Пропуск hardening для mlflow.%s: %s", ".".join(qualname[1:] + [name]), exc)
                break
            else:
                qualname.append(name)
        else:
            try:
                candidate = getattr(parent, attr)
            except AttributeError:
                continue
            except Exception as exc:  # pragma: no cover
                logger.debug("Не удалось получить %s.%s: %s", ".".join(qualname), attr, exc)
                continue
            qualname.append(attr)
            blocked = _disable_callable(candidate, ".".join(qualname))
            if blocked is not None:
                setattr(parent, attr, blocked)

    models = getattr(mlflow_module, "models", None)
    if models is not None:
        model_class = getattr(models, "Model", None)
        if model_class is not None and hasattr(model_class, "load"):
            blocked = _disable_callable(model_class.load, "mlflow.models.Model.load")
            if blocked is not None:
                model_class.load = staticmethod(blocked)

    recipes = getattr(mlflow_module, "recipes", None)
    if recipes is not None:
        recipe_class = getattr(recipes, "Recipe", None)
        if recipe_class is not None and hasattr(recipe_class, "load"):
            blocked = _disable_callable(recipe_class.load, "mlflow.recipes.Recipe.load")
            if blocked is not None:
                recipe_class.load = staticmethod(blocked)

    os.environ.setdefault("MLFLOW_ENABLE_MODEL_LOADING", "false")
    logger.info("MLflow hardened: загрузка моделей запрещена для предотвращения RCE")
