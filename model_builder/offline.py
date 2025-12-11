"""Offline fallback implementations for :mod:`bot.model_builder`."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger("TradingBot")


class OfflineModelBuilder:
    """Легковесная заглушка для :class:`model_builder.core.ModelBuilder`."""

    __offline_stub__ = True

    def __init__(
        self,
        config: Any,
        data_handler: Any,
        trade_manager: Any,
        gpt_client_factory: Any | None = None,
    ) -> None:
        self.config = config
        self.data_handler = data_handler
        self.trade_manager = trade_manager
        self.gpt_client_factory = gpt_client_factory
        self.predictive_models: dict[str, Any] = {}
        self.feature_cache: dict[str, Any] = {}
        self.calibrators: dict[str, Any | None] = {}
        self.scalers: dict[str, Any] = {}
        self.prediction_history: dict[str, list[float]] = {}
        self.base_thresholds: dict[str, float] = {}
        self.threshold_offset: float = 0.0
        self.device = "cpu"
        self.last_update_at: float | None = None
        self.save_interval = 900
        self.last_save_time = time.time()
        cache_dir = getattr(
            config, "cache_dir", getattr(config, "get", lambda k, d=None: d)("cache_dir", None)
        )
        if not cache_dir:
            cache_dir = "."
        self.state_file_path = os.path.join(cache_dir, "model_builder_state.pkl")
        logger.info(
            "OFFLINE_MODE=1 или отсутствуют зависимости: используется заглушка ModelBuilder"
        )

    async def backtest_loop(self) -> None:
        """Простейший цикл бэктеста для офлайн-режима."""

        interval = getattr(self.config, "backtest_interval", 0) or 0
        min_sharpe = getattr(self.config, "min_sharpe_ratio", 0.0) or 0.0
        while True:
            results = await self.backtest_all()
            for symbol, sharpe in results.items():
                logger.warning(
                    "Sharpe ratio for %s: %.3f (offline stub)", symbol, sharpe
                )
                if sharpe < min_sharpe:
                    try:
                        telegram = getattr(self.data_handler, "telegram_logger", None)
                        if telegram is not None:
                            await telegram.send_telegram_message(
                                f"Sharpe ratio warning for {symbol}: {sharpe}"
                            )
                    except Exception:  # pragma: no cover - best-effort notification
                        logger.debug("Telegram stub failed", exc_info=True)
            if interval <= 0:
                await asyncio.sleep(0)
                break
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # Публичные методы, которые вызываются офлайн-сервисами.
    # ------------------------------------------------------------------
    def update_models(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Имитировать обновление моделей и вернуть фиктивный результат."""

        self.last_update_at = time.time()
        return {"updated": True, "timestamp": self.last_update_at}

    async def backtest_all(self) -> dict[str, float]:
        """Вернуть фиктивные значения Sharpe ratio по символам."""

        pairs = getattr(self.data_handler, "usdt_pairs", []) or []
        return {symbol: 0.0 for symbol in pairs}

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Вернуть детерминированный ответ с нейтральным сигналом."""

        symbol = kwargs.get("symbol") or (args[0] if args else None)
        return {
            "symbol": symbol,
            "signal": "hold",
            "probability": 0.5,
            "meta": {"source": "offline"},
        }

    def get_cached_features(self, symbol: str):  # noqa: ANN201 - совместимость с основным классом
        return self.feature_cache.get(symbol)

    def save_state(self, *args: Any, **kwargs: Any) -> None:
        """Persist thresholds to a lightweight JSON file."""

        if time.time() - self.last_save_time < self.save_interval:
            return
        tmp_path = f"{self.state_file_path}.tmp"
        os.makedirs(os.path.dirname(self.state_file_path), exist_ok=True)
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(self.base_thresholds, handle)
            os.replace(tmp_path, self.state_file_path)
            self.last_save_time = time.time()
            logger.debug("OfflineModelBuilder.save_state завершен")
        except (OSError, ValueError) as exc:
            logger.error("Ошибка сохранения состояния ModelBuilder: %s", exc)
            with contextlib.suppress(OSError):
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            raise

    def load_state(self, *args: Any, **kwargs: Any) -> None:
        """Load thresholds persisted by :meth:`save_state`."""

        try:
            if os.path.exists(self.state_file_path):
                with open(self.state_file_path, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    self.base_thresholds = {
                        str(k): float(v) for k, v in loaded.items()
                    }
            logger.debug(
                "OfflineModelBuilder.load_state восстановил %d символов", len(self.base_thresholds)
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.error("Ошибка загрузки состояния ModelBuilder: %s", exc)
            self.base_thresholds = {}

    def compute_prediction_metrics(self, symbol: str) -> None:  # noqa: D401
        """Вернуть ``None``, показывая отсутствие метрик в офлайн-режиме."""

        logger.debug("OfflineModelBuilder.compute_prediction_metrics(%s) -> None", symbol)
        return None

    # Совместимость со старыми вызовами -------------------------------------------------
    def train_model(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Совместимость с устаревшими вызовами обучения."""

        return self.update_models(*args, **kwargs)

    def get_threshold(self, symbol: str) -> float:
        """Вернуть базовое значение порога для детерминированных ответов."""

        return self.base_thresholds.get(symbol, 0.5)

    async def prepare_lstm_features(self, symbol: str, indicators: Any) -> Any:
        """Подготовить минимальный набор признаков для LSTM в офлайне."""

        _ = (symbol, indicators)
        try:  # pragma: no cover - numpy может отсутствовать
            import numpy as np

            return np.zeros((60, 5), dtype=float)
        except Exception:
            return [[0.0] * 5 for _ in range(60)]

    async def precompute_features(self, symbol: str) -> None:
        """Сохранить рассчитанные признаки в кэш."""

        indicators = getattr(self.data_handler, "indicators", {}).get(symbol)
        features = await self.prepare_lstm_features(symbol, indicators)
        self.feature_cache[symbol] = features

    async def adjust_thresholds(self, symbol: str, prediction: float) -> tuple[float, float]:
        """Вернуть фиксированные пороги для офлайн-режима."""

        _ = (symbol, prediction)
        return 0.6, 0.4

    async def retrain_symbol(self, symbol: str) -> None:
        """Имитировать переобучение модели для указанного символа."""

        self.last_update_at = time.time()
        self.predictive_models[symbol] = {"retrained": True, "timestamp": self.last_update_at}

    async def compute_shap_values(self, symbol: str, model: Any, data: Any) -> None:
        """Сгенерировать фиктивные SHAP значения и записать их в файл."""

        shap_dir = os.path.join(getattr(self.config, "cache_dir", "."), "shap")
        os.makedirs(shap_dir, exist_ok=True)
        filename = hashlib.sha256(symbol.encode("utf-8", "replace")).hexdigest()
        path = os.path.join(shap_dir, f"shap_{filename}.pkl")
        try:
            import joblib

            values = data
            joblib.dump(values, path)
        except Exception:  # pragma: no cover - best effort persistence
            logger.debug("Failed to write SHAP cache", exc_info=True)

    async def prepare_dataset(self, *args: Any, **kwargs: Any) -> None:
        """Заглушка для совместимости с основным интерфейсом."""

        return None

    def __repr__(self) -> str:  # pragma: no cover - диагностический хелпер
        return "<OfflineModelBuilder stub>"
