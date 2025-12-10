"""Offline fallback implementations for :mod:`bot.model_builder`."""

from __future__ import annotations

import contextlib
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
            config,
            "cache_dir",
            getattr(config, "get", lambda k, d=None: d)("cache_dir", None),
        )
        if cache_dir is None:
            cache_dir = "."
        # Use an offline-only filename to avoid overwriting the real joblib state
        # used by :class:`model_builder.core.ModelBuilder`.
        self.state_file_path = os.path.join(
            cache_dir, "model_builder_state.offline.json"
        )
        # Backwards-compatible path for previously written offline JSON files.
        self._legacy_state_file_path = os.path.join(cache_dir, "model_builder_state.pkl")
        logger.info(
            "OFFLINE_MODE=1 или отсутствуют зависимости: используется заглушка ModelBuilder"
        )

    # ------------------------------------------------------------------
    # Публичные методы, которые вызываются офлайн-сервисами.
    # ------------------------------------------------------------------
    def update_models(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Имитировать обновление моделей и вернуть фиктивный результат."""

        self.last_update_at = time.time()
        return {"updated": True, "timestamp": self.last_update_at}

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

        candidates = [self.state_file_path]
        if self._legacy_state_file_path not in candidates:
            candidates.append(self._legacy_state_file_path)

        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
            except json.JSONDecodeError:
                logger.debug(
                    "Пропущен неподдерживаемый файл состояния ModelBuilder: %s", path
                )
                continue
            except (OSError, ValueError) as exc:
                logger.error("Ошибка загрузки состояния ModelBuilder: %s", exc)
                self.base_thresholds = {}
                return

            if isinstance(loaded, dict):
                self.base_thresholds = {str(k): float(v) for k, v in loaded.items()}
                break

        logger.debug(
            "OfflineModelBuilder.load_state восстановил %d символов", len(self.base_thresholds)
        )

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

    async def adjust_thresholds(self, symbol: str, prediction: float) -> tuple[float, float]:
        """Вернуть фиксированные пороги для офлайн-режима."""

        _ = (symbol, prediction)
        return 0.6, 0.4

    async def retrain_symbol(self, symbol: str) -> None:
        """Имитировать переобучение модели для указанного символа."""

        self.last_update_at = time.time()
        self.predictive_models[symbol] = {"retrained": True, "timestamp": self.last_update_at}

    async def prepare_dataset(self, *args: Any, **kwargs: Any) -> None:
        """Заглушка для совместимости с основным интерфейсом."""

        return None

    def __repr__(self) -> str:  # pragma: no cover - диагностический хелпер
        return "<OfflineModelBuilder stub>"
