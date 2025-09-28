"""REST API surface for the lightweight model builder service."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from flask import Flask, jsonify, request

from bot.dotenv_utils import load_dotenv
from services.logging_utils import sanitize_log_value
from security import (
    ArtifactDeserializationError,
    safe_joblib_load,
    verify_model_state_signature,
    write_model_state_signature,
)

from .core import fit_scaler, logger, prepare_features, validate_host
from .storage import JOBLIB_AVAILABLE, _safe_model_file_path, joblib

api_app = Flask(__name__)
_model: Any | None = None


def _load_model() -> None:
    """Load the persisted scikit-learn model if it exists."""

    global _model
    if not JOBLIB_AVAILABLE:
        logger.warning("joblib недоступен, REST API пропускает загрузку модели")
        _model = None
        return
    model_path = _safe_model_file_path()
    if model_path is None:
        _model = None
        return
    if not model_path.exists():
        return
    if model_path.is_symlink():
        logger.warning(
            "Отказ от загрузки модели из символьной ссылки %s",
            sanitize_log_value(model_path),
        )
        _model = None
        return
    if not model_path.is_file():
        logger.warning(
            "Отказ от загрузки модели: %s не является обычным файлом",
            sanitize_log_value(model_path),
        )
        _model = None
        return
    if not verify_model_state_signature(model_path):
        logger.warning(
            "Отказ от загрузки модели %s: подпись не прошла проверку",
            sanitize_log_value(model_path),
        )
        _model = None
        return
    try:
        _model = safe_joblib_load(model_path)
    except ArtifactDeserializationError:
        logger.error(
            "Отказ от загрузки модели %s: обнаружены недоверенные объекты",
            sanitize_log_value(model_path),
        )
        _model = None
    except (OSError, ValueError) as exc:  # pragma: no cover - model may be corrupted
        logger.exception("Не удалось загрузить модель: %s", exc)
        _model = None
    except Exception as exc:  # pragma: no cover - unexpected joblib failure
        logger.exception(
            "Неожиданная ошибка десериализации модели %s: %s",
            sanitize_log_value(model_path),
            exc,
        )
        _model = None


@api_app.route("/train", methods=["POST"])
def train_route():
    data = request.get_json(force=True)
    features, labels = prepare_features(
        data.get("features", []), data.get("labels", [])
    )
    if features.size == 0 or len(features) != len(labels):
        return jsonify({"error": "invalid training data"}), 400
    if len(np.unique(labels)) < 2:
        return jsonify({"error": "labels must contain at least two classes"}), 400
    model = fit_scaler(features, labels)
    if not JOBLIB_AVAILABLE:
        return jsonify({"error": "joblib unavailable"}), 503
    model_path = _safe_model_file_path()
    if model_path is None:
        return jsonify({"error": "invalid model path"}), 500
    if model_path.exists():
        if model_path.is_symlink():
            logger.warning(
                "Отказ от сохранения модели: путь %s является символьной ссылкой",
                sanitize_log_value(model_path),
            )
            return jsonify({"error": "model path unavailable"}), 500
        if not model_path.is_file():
            logger.warning(
                "Отказ от сохранения модели: %s не является обычным файлом",
                sanitize_log_value(model_path),
            )
            return jsonify({"error": "model path unavailable"}), 500
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - filesystem errors are rare
        logger.exception(
            "Не удалось подготовить каталог модели %s: %s",
            sanitize_log_value(model_path),
            exc,
        )
        return jsonify({"error": "model path unavailable"}), 500
    tmp_path = model_path.with_name(f"{model_path.name}.tmp")
    try:
        joblib.dump(model, tmp_path)
        os.replace(tmp_path, model_path)
        try:
            write_model_state_signature(model_path)
        except OSError as exc:  # pragma: no cover - проблемы с ФС крайне редки
            logger.warning(
                "Не удалось сохранить подпись модели %s: %s",
                sanitize_log_value(model_path),
                exc,
            )
    except Exception as exc:  # pragma: no cover - dump failures are rare
        logger.exception(
            "Не удалось сохранить модель в %s: %s",
            sanitize_log_value(model_path),
            exc,
        )
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            logger.debug(
                "Не удалось удалить временный файл %s",
                sanitize_log_value(tmp_path),
            )
        return jsonify({"error": "model save failed"}), 500
    global _model
    _model = model
    return jsonify({"status": "trained"})


@api_app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(force=True)
    features = data.get("features")
    if features is None:
        price_val = float(data.get("price", 0.0))
        features = [price_val]
    features = np.array(features, dtype=np.float32)
    if features.ndim == 0:
        features = np.array([[features]], dtype=np.float32)
    elif features.ndim == 1:
        features = features.reshape(1, -1)
    else:
        features = features.reshape(1, -1)
    price = float(features[0, 0]) if features.size else 0.0
    if _model is None:
        signal = "buy" if price > 0 else None
        prob = 1.0 if signal else 0.0
    else:
        prob = float(_model.predict_proba(features)[0, 1])
        signal = "buy" if prob >= 0.5 else "sell"
    return jsonify({"signal": signal, "prob": prob})


@api_app.route("/ping")
def ping():
    return jsonify({"status": "ok"})


try:  # pragma: no cover - optional in tests
    from bot.utils import configure_logging
except ImportError:  # pragma: no cover - stub for test environment

    def configure_logging() -> None:  # type: ignore
        """Stubbed logging configurator."""

        pass


def main() -> None:  # pragma: no cover - convenience CLI hook
    """Run the API using the same behaviour as ``python -m model_builder.api``."""

    configure_logging()
    load_dotenv()
    host = validate_host()
    port = int(os.getenv("MODEL_BUILDER_PORT", "8001"))
    _load_model()
    logger.info("Запуск сервиса ModelBuilder на %s:%s", host, port)
    api_app.run(host=host, port=port)  # хост проверен выше


if __name__ == "__main__":  # pragma: no cover - CLI execution
    main()
