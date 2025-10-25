"""Offline fallback implementations for :mod:`bot.model_builder`."""

from __future__ import annotations

import logging
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
        self.last_update_at: float | None = None
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

    def save_state(self, *args: Any, **kwargs: Any) -> None:
        """Сохранение состояния в офлайн-режиме опускается."""

        logger.debug("OfflineModelBuilder.save_state игнорирован")

    def load_state(self, *args: Any, **kwargs: Any) -> None:
        """Загрузка состояния в офлайн-режиме опускается."""

        logger.debug("OfflineModelBuilder.load_state игнорирован")

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

        return 0.5

    def __repr__(self) -> str:  # pragma: no cover - диагностический хелпер
        return "<OfflineModelBuilder stub>"
