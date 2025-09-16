import asyncio
import logging
from typing import List, Optional, Tuple

import httpx

from bot.config import OFFLINE_MODE

logger = logging.getLogger("TradingBot")

_MODEL_VERSION = 0


async def _fetch_training_data(
    data_url: str, symbol: str, limit: int = 50
) -> Tuple[List[List[float]], List[int]]:
    """Fetch recent OHLCV data and derive direction labels."""

    if OFFLINE_MODE:
        logger.info(
            "Offline mode: skipping training data fetch for %s", symbol
        )
        return [], []

    features: List[List[float]] = []
    labels: List[int] = []
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            resp = await client.get(
                f"{data_url.rstrip('/')}/ohlcv/{symbol}",
                params={"limit": limit},
                timeout=5.0,
            )
        if resp.status_code != 200:
            logger.error("Failed to fetch OHLCV: HTTP %s", resp.status_code)
            return features, labels
        ohlcv = resp.json().get("ohlcv", [])
    except httpx.HTTPError as exc:  # pragma: no cover - network errors
        logger.error("Training data request error: %s", exc)
        return features, labels

    for i in range(1, len(ohlcv)):
        try:
            _ts, open_price, high_price, low_price, close_price, volume = ohlcv[i]
            prev_close = float(ohlcv[i - 1][4])
            direction = 1 if float(close_price) > prev_close else 0
            features.append(
                [
                    float(open_price),
                    float(high_price),
                    float(low_price),
                    float(close_price),
                    float(volume),
                ]
            )
            labels.append(direction)
        except (ValueError, TypeError, IndexError):
            continue
    return features, labels


async def train(url: str, features: List[List[float]], labels: List[int]) -> bool:
    """Send training data to the model_builder service."""

    if OFFLINE_MODE:
        logger.info("Offline mode: skipping model training request")
        return True

    payload = {"features": features, "labels": labels}
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.post(
                f"{url.rstrip('/')}/train", json=payload, timeout=5.0
            )
        if response.status_code == 200:
            return True
        logger.error("Model training failed: HTTP %s", response.status_code)
    except httpx.HTTPError as exc:  # pragma: no cover - network errors
        logger.error("Model training request error: %s", exc)
    return False


async def retrain(model_url: str, data_url: str, symbol: str = "BTCUSDT") -> Optional[int]:
    """Retrain the model using data from ``data_url``."""

    global _MODEL_VERSION
    if OFFLINE_MODE:
        logger.info("Offline mode: retrain skipped")
        return _MODEL_VERSION

    features, labels = await _fetch_training_data(data_url, symbol)
    if not features or not labels:
        logger.error("No training data available for retrain")
        return None
    if await train(model_url, features, labels):
        _MODEL_VERSION += 1
        logger.info("Model retrained, new version %s", _MODEL_VERSION)
        return _MODEL_VERSION
    return None


async def _retrain_loop(
    model_url: str, data_url: str, interval: float, symbol: str
) -> None:
    if OFFLINE_MODE:
        logger.info("Offline mode: retrain loop disabled")
        return
    while True:
        await retrain(model_url, data_url, symbol)
        await asyncio.sleep(interval)


def schedule_retrain(
    model_url: str, data_url: str, interval: float, symbol: str = "BTCUSDT"
) -> asyncio.Task:
    """Start periodic retraining task."""

    return asyncio.create_task(_retrain_loop(model_url, data_url, interval, symbol))
