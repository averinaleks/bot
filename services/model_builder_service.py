"""Model builder microservice.

This service exposes ``/train`` and ``/predict`` endpoints used by tests and
examples.  Depending on the ``nn_framework`` specified in ``config.json`` it
either trains a simple scikit-learn model or delegates work to the more
feature-rich :class:`ModelBuilder` class.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
import re
import threading
import unicodedata
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
from flask import Flask, jsonify, request
from bot.dotenv_utils import load_dotenv
from bot.utils_loader import require_utils
from config import open_config_file
from model_builder.validation import (
    FeatureValidationError,
    MAX_FEATURES_PER_SAMPLE,
    MAX_SAMPLES,
    coerce_feature_matrix,
    coerce_feature_vector,
    coerce_label_vector,
    coerce_float,
)
from services.logging_utils import configure_service_logging, sanitize_log_value
from security import (
    ArtifactDeserializationError,
    create_joblib_stub,
    safe_joblib_load,
    set_model_dir,
    verify_model_state_signature,
    write_model_state_signature,
)

configure_service_logging()
_utils = require_utils(
    "ensure_writable_directory",
    "safe_int",
    "sanitize_symbol",
    "validate_host",
)
ensure_writable_directory = _utils.ensure_writable_directory
safe_int = _utils.safe_int
sanitize_symbol = _utils.sanitize_symbol
validate_host = _utils.validate_host

_FILENAME_STRIP_RE = re.compile(r"[^A-Za-z0-9_.-]")
_FILENAME_HYPHEN_RE = re.compile(r"[-_\s]+")


def _fallback_secure_filename(filename: str) -> str:
    """Return a Werkzeug-compatible safe filename implementation."""

    if not isinstance(filename, str):
        raise TypeError("filename must be a string")
    value = unicodedata.normalize("NFKD", filename)
    # Drop path separators to avoid directory traversal when Werkzeug is absent
    for sep in (os.path.sep, os.path.altsep):
        if sep:
            value = value.replace(sep, " ")
    # Coerce to ASCII to align with Werkzeug behaviour
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.strip()
    value = _FILENAME_STRIP_RE.sub("", value)
    value = _FILENAME_HYPHEN_RE.sub("-", value)
    value = value.strip("._")
    return value or "file"


try:
    from werkzeug.utils import secure_filename as _werkzeug_secure_filename
except ImportError:
    secure_filename = _fallback_secure_filename
else:
    secure_filename = _werkzeug_secure_filename

try:  # optional dependency
    from flask.typing import ResponseReturnValue
except Exception:  # pragma: no cover - fallback when flask.typing missing
    ResponseReturnValue = Any  # type: ignore

LOGGER = logging.getLogger(__name__)

try:
    import joblib  # type: ignore
    JOBLIB_AVAILABLE = True
except Exception as exc:  # pragma: no cover - joblib is optional in tests
    JOBLIB_AVAILABLE = False
    LOGGER.warning(
        "Не удалось импортировать joblib: %s. Сохранение моделей отключено.", exc
    )
    joblib = create_joblib_stub(
        "joblib недоступен: установите зависимость для работы с артефактами"
    )
    sys.modules.setdefault("joblib", joblib)

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - critical dependency missing
    LOGGER.critical(
        "Не удалось импортировать `pandas`. Установите пакет "
        "(`pip install pandas`) или переключитесь на обработку данных через "
        "стандартные структуры/CSV до подключения `pandas`."
    )
    raise ImportError(
        "Сервис требует установленный `pandas` для подготовки данных."
    ) from exc

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from model_builder import ModelBuilder

load_dotenv()
app = Flask(__name__)
if hasattr(app, "config"):
    app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB limit

# Determine the project root based on this file's location rather than the
# current working directory so the service can be launched from anywhere.
BASE_DIR = Path(__file__).resolve().parent.parent
_CONFIG_DIR = BASE_DIR
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "config.json"


def _is_within_directory(path: Path, directory: Path) -> bool:
    """Return ``True`` if ``path`` is located within ``directory``."""

    try:
        path.resolve(strict=False).relative_to(directory.resolve(strict=False))
    except ValueError:
        return False
    return True


def _resolve_config_path(raw: str | os.PathLike[str] | None) -> Path:
    if raw is None:
        return _DEFAULT_CONFIG_PATH

    try:
        raw_path = os.fspath(raw)
    except TypeError:
        LOGGER.warning("Invalid CONFIG_PATH %r; using default", raw)
        return _DEFAULT_CONFIG_PATH

    if raw_path == "":
        return _DEFAULT_CONFIG_PATH

    candidate = Path(raw_path)

    if not candidate.is_absolute():
        candidate = (_CONFIG_DIR / candidate).resolve(strict=False)
    else:
        try:
            candidate = candidate.resolve(strict=False)
        except OSError as exc:
            LOGGER.warning(
                "Failed to resolve CONFIG_PATH %s: %s; using default",
                sanitize_log_value(str(candidate)),
                exc,
            )
            return _DEFAULT_CONFIG_PATH

    if candidate.is_symlink():
        LOGGER.warning(
            "CONFIG_PATH %s is a symlink; using default",
            sanitize_log_value(str(candidate)),
        )
        return _DEFAULT_CONFIG_PATH

    if not _is_within_directory(candidate, _CONFIG_DIR):
        LOGGER.warning(
            "CONFIG_PATH %s escapes service directory %s; using default",
            sanitize_log_value(str(candidate)),
            sanitize_log_value(str(_CONFIG_DIR)),
        )
        return _DEFAULT_CONFIG_PATH

    return candidate


CONFIG_PATH = _resolve_config_path(os.getenv("CONFIG_PATH"))
try:
    with open_config_file(CONFIG_PATH) as handle:
        _CFG: Dict[str, Any] = json.load(handle)
except FileNotFoundError:
    _CFG = {}
except (OSError, json.JSONDecodeError) as exc:
    LOGGER.warning("Failed to load model builder config %s: %s", CONFIG_PATH, exc)
    _CFG = {}

NN_FRAMEWORK = os.getenv("NN_FRAMEWORK", _CFG.get("nn_framework", "sklearn")).lower()
if os.getenv("TEST_MODE") == "1" and "NN_FRAMEWORK" not in os.environ:
    NN_FRAMEWORK = "sklearn"
MODEL_TYPE = _CFG.get("model_type", "transformer")
MODEL_DIR = ensure_writable_directory(
    Path(os.getenv("MODEL_DIR", ".")),
    description="моделей",
    fallback_subdir="trading_bot_models",
).resolve()
set_model_dir(MODEL_DIR)
def _resolve_model_file(path_value: str | Path | None) -> Path:
    """Return a sanitised path for pre-trained model artefacts."""

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


MODEL_FILE: str | Path | None = os.environ.get("MODEL_FILE")


def _get_model_file_path() -> Path | None:
    if MODEL_FILE in (None, "", Path("")):
        return None
    try:
        return _resolve_model_file(MODEL_FILE)
    except ValueError as exc:
        app.logger.warning(
            "Refusing to use MODEL_FILE %s: %s",
            sanitize_log_value("<unset>" if MODEL_FILE is None else str(MODEL_FILE)),
            exc,
        )
        return None

_state_lock = threading.RLock()
_models: Dict[str, Any] = {}
_scalers: Dict[str, Any] = {}
_scaler: Any = None  # backwards compatibility for tests
_model_builder: ModelBuilder | None = None


def _load_model() -> None:
    """Best-effort loading of a pre-trained model for compatibility tests."""

    model_file = _get_model_file_path()
    if model_file is None or not model_file.exists():  # nothing to load
        return
    if model_file.is_symlink():
        app.logger.warning(
            "Refusing to load model from symlink %s",
            sanitize_log_value(model_file),
        )
        return
    if not JOBLIB_AVAILABLE:
        app.logger.warning(
            "joblib недоступен, предварительно обученная модель не загружена"
        )
        return
    try:  # pragma: no cover - exercised in integration tests
        if not verify_model_state_signature(model_file):
            app.logger.warning(
                "Отказ от загрузки неподписанного артефакта модели %s",
                sanitize_log_value(model_file),
            )
            return
        with _state_lock:
            _models["default"] = safe_joblib_load(model_file)
    except Exception:
        app.logger.exception("Failed to load model from %s", model_file)


if NN_FRAMEWORK != "sklearn":
    from bot.config import BotConfig
    from model_builder import ModelBuilder, KERAS_FRAMEWORKS, _get_torch_modules

    class _DummyDH:
        usdt_pairs: list[str] = []
        indicators: Dict[str, Any] = {}
        ohlcv = pd.DataFrame()
        funding_rates: Dict[str, float] = {}
        open_interest: Dict[str, float] = {}
        telegram_logger = type(
            "TL", (), {"send_telegram_message": staticmethod(lambda *a, **k: None)}
        )()

    class _DummyTM:
        last_volatility: Dict[str, float] = {}

        async def get_loss_streak(self, symbol: str) -> int:  # pragma: no cover - stub
            return 0

        async def get_win_streak(self, symbol: str) -> int:  # pragma: no cover - stub
            return 0

        async def get_sharpe_ratio(self, symbol: str) -> float:  # pragma: no cover - stub
            return 0.0

    bot_cfg = BotConfig()
    bot_cfg.nn_framework = NN_FRAMEWORK
    bot_cfg.model_type = MODEL_TYPE
    bot_cfg.cache_dir = str(MODEL_DIR)
    _model_builder = ModelBuilder(bot_cfg, _DummyDH(), _DummyTM())
    try:  # pragma: no cover - file may not exist
        _model_builder.load_state()
    except Exception:
        app.logger.exception("Failed to load ModelBuilder state")

else:  # scikit-learn fallback used by tests
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # pragma: no cover - missing sklearn in tests
        LOGGER.warning("Не удалось импортировать sklearn.linear_model: %s", exc)

        class LogisticRegression:  # type: ignore
            """Упрощённая замена LogisticRegression для тестового режима."""

            def __init__(self) -> None:
                self.coef_: np.ndarray | None = None
                self.intercept_: float = 0.0

            def fit(self, X, y):  # pragma: no cover - simple heuristic
                features = np.asarray(X, dtype=float)
                labels = np.asarray(y, dtype=float)
                if features.ndim == 1:
                    features = features.reshape(-1, 1)
                if features.size == 0:
                    raise ValueError("training data is empty")
                if labels.shape[0] != features.shape[0]:
                    raise ValueError("labels size mismatch")
                positive = features[labels >= 0.5]
                negative = features[labels < 0.5]
                if positive.size == 0 or negative.size == 0:
                    self.coef_ = np.zeros((1, features.shape[1]))
                    self.intercept_ = 0.0
                else:
                    pos_mean = positive.mean(axis=0)
                    neg_mean = negative.mean(axis=0)
                    weights = pos_mean - neg_mean
                    self.coef_ = weights.reshape(1, -1)
                    midpoint = (pos_mean + neg_mean) / 2
                    self.intercept_ = -float(np.dot(weights, midpoint))
                return self

            def predict_proba(self, X):  # pragma: no cover - simple heuristic
                features = np.asarray(X, dtype=float)
                if features.ndim == 1:
                    features = features.reshape(-1, 1)
                if self.coef_ is None:
                    probs = np.full((features.shape[0], 1), 0.5)
                else:
                    logits = features @ self.coef_.T + self.intercept_
                    logits = np.clip(logits, -50.0, 50.0)
                    probs = 1.0 / (1.0 + np.exp(-logits))
                probs = probs.reshape(-1)
                return np.column_stack((1.0 - probs, probs))

    try:  # optional dependency
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - fallback when sklearn missing
        LOGGER.warning("Не удалось импортировать sklearn.preprocessing: %s", exc)

        class StandardScaler:  # type: ignore
            def fit(self, X):
                values = np.asarray(X, dtype=float)
                if values.ndim == 1:
                    values = values.reshape(-1, 1)
                self.mean_ = np.mean(values, axis=0)
                self.scale_ = np.std(values, axis=0)
                return self

            def transform(self, X):
                values = np.asarray(X, dtype=float)
                if values.ndim == 1:
                    values = values.reshape(-1, 1)
                scale = np.where(self.scale_ == 0, 1, self.scale_)
                return (values - self.mean_) / scale

            def fit_transform(self, X):
                self.fit(X)
                return self.transform(X)

    def _normalise_symbol(symbol: str) -> str:
        """Return a safe filesystem token derived from ``symbol``."""

        cleaned = sanitize_symbol(symbol)
        cleaned = _FILENAME_HYPHEN_RE.sub("-", cleaned)[:64].strip(".-_")
        if not cleaned:
            raise ValueError("symbol resolves to an empty filename")
        if cleaned.startswith(".") or ".." in cleaned:
            raise ValueError("symbol resolves to a hidden or unsafe filename")
        if "/" in cleaned or "\\" in cleaned:
            raise ValueError("symbol must not contain path separators")
        return cleaned

    def _model_path(symbol: str) -> Path:
        """Return a strictly sanitized and contained path for per-symbol joblib artefacts."""

        safe = _normalise_symbol(symbol)
        filename = f"{safe}.pkl"
        # Create candidate path and normalize it
        candidate_path = MODEL_DIR / filename
        resolved = candidate_path.resolve(strict=False)
        if not _is_within_directory(resolved, MODEL_DIR):
            raise ValueError("Invalid model path - outside of MODEL_DIR")
        if resolved.exists() and resolved.is_symlink():
            raise ValueError("Invalid model path - symlink not allowed")
        return resolved

    def _load_state(symbol: str) -> None:
        with _state_lock:
            # Only allow models that actually exist in the MODEL_DIR.
            allowed_symbols = {p.stem for p in MODEL_DIR.glob("*.pkl")}
            if symbol not in allowed_symbols:
                app.logger.warning(
                    "Refused to load model: %s is not whitelisted",
                    sanitize_log_value(symbol),
                )
                return
            if not JOBLIB_AVAILABLE:
                app.logger.warning(
                    "joblib недоступен, загрузка состояния модели %s пропущена",
                    sanitize_log_value(symbol),
                )
                return
            try:
                path = _model_path(symbol)
            except ValueError as exc:
                app.logger.warning(
                    "Refused to load model %s: %s",
                    sanitize_log_value(symbol),
                    exc,
                )
                return
            if not path.exists():
                return
            if path.is_symlink():
                app.logger.warning(
                    "Refused to load model %s: path is a symlink",
                    sanitize_log_value(path),
                )
                return
            if not path.is_file():
                app.logger.warning(
                    "Refused to load model %s: not a regular file",
                    sanitize_log_value(path),
                )
                return
            if not _is_within_directory(path, MODEL_DIR):
                app.logger.warning(
                    "Refused to load model %s: path escapes MODEL_DIR",
                    sanitize_log_value(path),
                )
                return
            if not verify_model_state_signature(path):
                app.logger.warning(
                    "Refused to load model %s: signature mismatch",
                    sanitize_log_value(path),
                )
                return
            try:
                data = safe_joblib_load(path)
            except ArtifactDeserializationError:
                app.logger.warning(
                    "Refused to load model %s: содержит недоверенные объекты", path
                )
                return
            except Exception:
                app.logger.exception("Failed to load model artefact from %s", path)
                return
            _models[symbol] = data.get("model")
            _scalers[symbol] = data.get("scaler")

    def _save_state(symbol: str) -> None:
        with _state_lock:
            model = _models.get(symbol)
            if model is None:
                return
            if not JOBLIB_AVAILABLE:
                app.logger.warning(
                    "joblib недоступен, состояние модели %s не сохранено",
                    sanitize_log_value(symbol),
                )
                return
            data = {"model": model, "scaler": _scalers.get(symbol)}
            try:
                path = _model_path(symbol)
            except ValueError as exc:
                app.logger.warning(
                    "Refused to save model %s: %s",
                    sanitize_log_value(symbol),
                    exc,
                )
                return
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                app.logger.exception(
                    "Failed to prepare directory for %s: %s",
                    sanitize_log_value(path),
                    exc,
                )
                return
            tmp_path = path.with_name(f"{path.name}.tmp")
            try:
                joblib.dump(data, tmp_path)
                os.replace(tmp_path, path)
                try:
                    write_model_state_signature(path)
                except OSError as exc:
                    app.logger.warning(
                        "Failed to persist signature for %s: %s",
                        sanitize_log_value(path),
                        exc,
                    )
            except Exception as exc:  # pragma: no cover - dump failures are rare
                app.logger.exception(
                    "Failed to persist model %s: %s",
                    sanitize_log_value(symbol),
                    exc,
                )
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                # Last-resort cleanup; log at debug level to avoid log spam
                app.logger.debug(
                    "Unable to clean up temporary file %s", sanitize_log_value(tmp_path)
                )


def _compute_ema(prices: list[float], span: int = 3) -> np.ndarray:
    """Calculate EMA and shift to avoid look-ahead bias."""

    series = pd.Series(prices, dtype=float)
    ema = series.ewm(span=span, adjust=False).mean().shift(1)
    return ema.to_numpy(dtype=np.float32)


@app.route("/train", methods=["POST"])
def train() -> ResponseReturnValue:
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        app.logger.warning("/train: payload is not a JSON object")
        return jsonify({"error": "invalid training data"}), 400

    symbol = data.get("symbol", "default")
    if NN_FRAMEWORK != "sklearn":
        import asyncio
        if _model_builder is None:  # pragma: no cover - should be set in non-sklearn mode
            return jsonify({"error": "model builder not initialized"}), 500
        try:
            asyncio.run(_model_builder.retrain_symbol(symbol))
            _model_builder.save_state()
            return jsonify({"status": "trained"})
        except Exception as exc:  # pragma: no cover - training may fail
            app.logger.exception("ModelBuilder training failed: %s", exc)
            return jsonify({"error": "training failed"}), 500

    prices = data.get("prices")
    try:
        if prices is not None:
            price_series = coerce_label_vector(
                prices,
                max_rows=MAX_SAMPLES,
            )
            features = _compute_ema(price_series.tolist()).reshape(-1, 1)
        else:
            features = coerce_feature_matrix(
                data.get("features"),
                max_rows=MAX_SAMPLES,
                max_features=MAX_FEATURES_PER_SAMPLE,
            )

        labels = coerce_label_vector(
            data.get("labels"),
            max_rows=len(features) if len(features) else MAX_SAMPLES,
        )
    except FeatureValidationError as exc:
        app.logger.warning(
            "Rejected training payload: %s",
            sanitize_log_value(str(exc)),
        )
        return jsonify({"error": "invalid training data"}), 400

    if features.size == 0 or len(features) != len(labels):
        return jsonify({"error": "invalid training data"}), 400

    mask = ~np.isnan(features).any(axis=1)
    features = features[mask]
    labels = labels[mask]
    if features.size == 0:
        return jsonify({"error": "invalid training data"}), 400
    df = pd.DataFrame(features)
    mask = pd.isna(df) | ~np.isfinite(df)
    if mask.any().any():
        bad_rows = mask.any(axis=1)
        app.logger.warning(
            "Обнаружены некорректные значения в данных: %s строк", int(bad_rows.sum())
        )
        df = df[~bad_rows]
        labels = labels[~bad_rows.to_numpy()]
    features = df.to_numpy(dtype=np.float32)
    if pd.isna(df).any().any():
        return jsonify({"error": "training data contains NaN values"}), 400
    if not np.isfinite(features).all():
        return jsonify({"error": "training data contains infinite values"}), 400
    if len(np.unique(labels)) < 2:
        return jsonify({"error": "labels must contain at least two classes"}), 400
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)
    # Параметр ``multi_class="auto"`` больше не нужен в новых версиях
    # scikit-learn: логистическая регрессия всегда работает в режиме
    # ``multinomial``. Убираем явное указание, чтобы не появлялись
    # предупреждения об устаревании и чтобы код оставался совместимым
    # с будущими релизами библиотеки.
    model = LogisticRegression()
    model.fit(features, labels)
    try:
        with _state_lock:
            _models[symbol] = model
            _scalers[symbol] = scaler
            if symbol == "default":  # maintain legacy globals
                global _scaler
                _scaler = scaler
            _save_state(symbol)
    except Exception:
        app.logger.exception(
            "Failed to save model state for %s",
            sanitize_log_value(symbol),
        )
        return jsonify({"error": "invalid model path"}), 400
    return jsonify({"status": "trained"})


@app.route("/predict", methods=["POST"])
def predict() -> ResponseReturnValue:
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        app.logger.warning("/predict: payload is not a JSON object")
        return jsonify({"error": "invalid payload"}), 400

    symbol = data.get("symbol", "default")
    if NN_FRAMEWORK != "sklearn":
        if _model_builder is None:
            return jsonify({"error": "ModelBuilder not initialized"}), 500
        raw_features = data.get("features")
        try:
            if raw_features is None:
                price_raw = data.get("price")
                price_val = coerce_float(price_raw if price_raw is not None else 0.0)
                features = coerce_feature_vector([price_val])
            else:
                features = coerce_feature_vector(raw_features)
        except FeatureValidationError as exc:
            app.logger.warning(
                "Rejected prediction payload: %s",
                sanitize_log_value(str(exc)),
            )
            return jsonify({"error": "invalid features"}), 400

        model = _model_builder.predictive_models.get(symbol)
        if model is None:
            price = float(features[0, 0]) if features.size else 0.0
            prob = 1.0 if price > 0 else 0.0
        else:
            try:
                scaler = _model_builder.scalers.get(symbol)
                if scaler is not None:
                    features = scaler.transform(features)
                if _model_builder.nn_framework in KERAS_FRAMEWORKS:
                    raw_prob = float(model.predict(features)[0, 0])
                else:
                    torch_mods = _get_torch_modules()
                    torch = torch_mods["torch"]
                    model.eval()
                    with torch.no_grad():
                        tens = torch.tensor(
                            features, dtype=torch.float32, device=_model_builder.device
                        )
                        if _model_builder.model_type == "mlp":
                            tens = tens.view(tens.size(0), -1)
                        raw_prob = float(torch.sigmoid(model(tens)).cpu().numpy().item())
                prob = raw_prob
                calibrator = _model_builder.calibrators.get(symbol)
                if calibrator is not None:
                    prob = float(calibrator.predict_proba([[raw_prob]])[0, 1])
            except Exception as exc:  # pragma: no cover - prediction may fail
                app.logger.exception("Prediction failed: %s", exc)
                prob = 0.0
        threshold = _model_builder.base_thresholds.get(symbol, 0.5) + _model_builder.threshold_offset.get(symbol, 0.0)
        signal = "buy" if prob >= threshold else "sell"
        return jsonify({"signal": signal, "prob": prob, "threshold": threshold})

    raw_features = data.get("features")
    try:
        if raw_features is None:
            price_raw = data.get("price")
            price_val = coerce_float(price_raw if price_raw is not None else 0.0)
            features = coerce_feature_vector([price_val])
        else:
            features = coerce_feature_vector(raw_features)
    except FeatureValidationError as exc:
        app.logger.warning(
            "Rejected prediction payload: %s",
            sanitize_log_value(str(exc)),
        )
        return jsonify({"error": "invalid features"}), 400

    with _state_lock:
        model = _models.get(symbol)
        scaler = _scalers.get(symbol)
    if model is None:
        _load_state(symbol)
        with _state_lock:
            model = _models.get(symbol)
            scaler = _scalers.get(symbol)
    if scaler is not None:
        features = scaler.transform(features)
    if model is None:
        price = float(features[0, 0]) if features.size else 0.0
        prob = 1.0 if price > 0 else 0.0
    else:
        prob = float(model.predict_proba(features)[0, 1])
    threshold = 0.5
    signal = "buy" if prob >= threshold else "sell"
    return jsonify({"signal": signal, "prob": prob, "threshold": threshold})


@app.route("/ping")
def ping() -> ResponseReturnValue:
    return jsonify({"status": "ok"})


if hasattr(app, "errorhandler"):
    @app.errorhandler(413)
    def too_large(_) -> ResponseReturnValue:
        return jsonify({"error": "payload too large"}), 413
else:  # pragma: no cover - simplified Flask used in tests
    def too_large(_) -> ResponseReturnValue:
        return jsonify({"error": "payload too large"}), 413


if __name__ == "__main__":  # pragma: no cover - manual launch
    from bot.utils import configure_logging

    configure_logging()
    host = validate_host()
    port = safe_int(os.getenv("PORT", "8000"))
    app.logger.info("Запуск сервиса ModelBuilder на %s:%s", host, port)
    app.run(host=host, port=port)  # host validated above

