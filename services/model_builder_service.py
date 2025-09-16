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
try:
    from werkzeug.utils import secure_filename
except ImportError:
    def secure_filename(filename: str) -> str:
        return filename
from typing import Any, Dict, TYPE_CHECKING

LOGGER = logging.getLogger(__name__)

try:
    import joblib  # type: ignore
    JOBLIB_AVAILABLE = True
except Exception as exc:  # pragma: no cover - joblib is optional in tests
    JOBLIB_AVAILABLE = False
    LOGGER.warning(
        "Не удалось импортировать joblib: %s. Сохранение моделей отключено.", exc
    )

    import types

    def _joblib_unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "joblib недоступен: установите зависимость для работы с артефактами"
        )

    joblib = types.ModuleType("joblib")
    joblib.dump = _joblib_unavailable  # type: ignore[attr-defined]
    joblib.load = _joblib_unavailable  # type: ignore[attr-defined]
    sys.modules.setdefault("joblib", joblib)
import numpy as np
from numpy.typing import NDArray
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
from bot.dotenv_utils import load_dotenv
from flask import Flask, jsonify, request

try:  # optional dependency
    from flask.typing import ResponseReturnValue
except Exception:  # pragma: no cover - fallback when flask.typing missing
    ResponseReturnValue = Any  # type: ignore

from utils import validate_host, safe_int

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from model_builder import ModelBuilder

load_dotenv()
app = Flask(__name__)
if hasattr(app, "config"):
    app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB limit

# Determine the project root based on this file's location rather than the
# current working directory so the service can be launched from anywhere.
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.getenv("CONFIG_PATH", BASE_DIR / "config.json"))
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _CFG: Dict[str, Any] = json.load(f)
except Exception:
    _CFG = {}

NN_FRAMEWORK = os.getenv("NN_FRAMEWORK", _CFG.get("nn_framework", "sklearn")).lower()
if os.getenv("TEST_MODE") == "1" and "NN_FRAMEWORK" not in os.environ:
    NN_FRAMEWORK = "sklearn"
MODEL_TYPE = _CFG.get("model_type", "transformer")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "."))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

_models: Dict[str, Any] = {}
_scalers: Dict[str, Any] = {}
_scaler: Any = None  # backwards compatibility for tests
_model_builder: ModelBuilder | None = None


def _load_model() -> None:
    """Best-effort loading of a pre-trained model for compatibility tests."""
    model_file = Path(os.getenv("MODEL_FILE", ""))
    if not model_file.is_file():  # nothing to load
        return
    if not JOBLIB_AVAILABLE:
        app.logger.warning(
            "joblib недоступен, предварительно обученная модель не загружена"
        )
        return
    try:  # pragma: no cover - exercised in integration tests
        _models["default"] = joblib.load(model_file)
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
    from sklearn.linear_model import LogisticRegression

    try:  # optional dependency
        from sklearn.preprocessing import StandardScaler
    except Exception:  # pragma: no cover - fallback when sklearn missing
        class StandardScaler:  # type: ignore
            def fit(self, X):
                self.mean_ = np.mean(X, axis=0)
                self.scale_ = np.std(X, axis=0)
                return self

            def transform(self, X):
                scale = np.where(self.scale_ == 0, 1, self.scale_)
                return (X - self.mean_) / scale

            def fit_transform(self, X):
                self.fit(X)
                return self.transform(X)

    def _model_path(symbol: str) -> Path:
        # Use werkzeug.utils.secure_filename if available, otherwise fallback to basic sanitation
        if secure_filename is not None:
            safe = secure_filename(symbol)
        else:
            safe = Path(symbol).name
        candidate = (MODEL_DIR / f"{safe}.pkl")
        fullpath = candidate.resolve()
        # Ensure the full path is within the MODEL_DIR
        if MODEL_DIR.resolve() not in fullpath.parents:
            raise ValueError("Invalid model path - outside of MODEL_DIR")
        return fullpath

    def _load_state(symbol: str) -> None:
        # Only allow models that actually exist in the MODEL_DIR.
        allowed_symbols = {p.stem for p in MODEL_DIR.glob("*.pkl")}
        if symbol not in allowed_symbols:
            app.logger.warning("Refused to load model: %r is not whitelisted", symbol)
            return
        if not JOBLIB_AVAILABLE:
            app.logger.warning(
                "joblib недоступен, загрузка состояния модели %s пропущена", symbol
            )
            return
        path = _model_path(symbol)
        if path.exists():
            data = joblib.load(path)
            _models[symbol] = data.get("model")
            _scalers[symbol] = data.get("scaler")

    def _save_state(symbol: str) -> None:
        model = _models.get(symbol)
        if model is None:
            return
        if not JOBLIB_AVAILABLE:
            app.logger.warning(
                "joblib недоступен, состояние модели %s не сохранено", symbol
            )
            return
        data = {"model": model, "scaler": _scalers.get(symbol)}
        joblib.dump(data, _model_path(symbol))


def _compute_ema(prices: list[float], span: int = 3) -> np.ndarray:
    """Calculate EMA and shift to avoid look-ahead bias."""

    series = pd.Series(prices, dtype=float)
    ema = series.ewm(span=span, adjust=False).mean().shift(1)
    return ema.to_numpy(dtype=np.float32)


@app.route("/train", methods=["POST"])
def train() -> ResponseReturnValue:
    data = request.get_json(force=True)
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
    features: NDArray[np.float32]
    if prices is not None:
        features = _compute_ema(prices).reshape(-1, 1)
    else:
        features = np.array(data.get("features", []), dtype=np.float32)
    labels = np.array(data.get("labels", []), dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    else:
        features = features.reshape(len(features), -1)
    mask = ~np.isnan(features).any(axis=1)
    features = features[mask]
    labels = labels[mask]
    if features.size == 0 or len(features) != len(labels):
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
    _models[symbol] = model
    _scalers[symbol] = scaler
    if symbol == "default":  # maintain legacy globals
        global _scaler
        _scaler = scaler
    try:
        _save_state(symbol)
    except Exception:
        app.logger.exception("Failed to save model state for %s", symbol)
        return jsonify({"error": "invalid model path"}), 400
    return jsonify({"status": "trained"})


@app.route("/predict", methods=["POST"])
def predict() -> ResponseReturnValue:
    data = request.get_json(force=True)
    symbol = data.get("symbol", "default")
    if NN_FRAMEWORK != "sklearn":
        if _model_builder is None:
            return jsonify({"error": "ModelBuilder not initialized"}), 500
        features = np.array(data.get("features", []), dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        model = _model_builder.predictive_models.get(symbol)
        if model is None:
            price = float(features[0, 0]) if features.size else 0.0
            signal = "buy" if price > 0 else None
            prob = 1.0 if signal else 0.0
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

    features = data.get("features")
    if features is None:
        price_val = float(data.get("price", 0.0))
        features = np.array([price_val], dtype=np.float32)
    else:
        features = np.array(features, dtype=np.float32)
    if features.ndim == 0:
        features = np.array([[features]], dtype=np.float32)
    elif features.ndim == 1:
        features = features.reshape(1, -1)
    else:
        features = features.reshape(1, -1)
    model = _models.get(symbol)
    scaler = _scalers.get(symbol)
    if model is None:
        _load_state(symbol)
        model = _models.get(symbol)
        scaler = _scalers.get(symbol)
    if scaler is not None:
        features = scaler.transform(features)
    if model is None:
        price = float(features[0, 0]) if features.size else 0.0
        signal = "buy" if price > 0 else None
        prob = 1.0 if signal else 0.0
    else:
        prob = float(model.predict_proba(features)[0, 1])
    threshold = 0.5
    if model is not None:
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

