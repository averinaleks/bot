import asyncio
import logging
from typing import List, Optional

import httpx

logger = logging.getLogger("TradingBot")

_MODEL_VERSION = 0


async def train(url: str, features: List[List[float]], labels: List[int]) -> bool:
    """Send training data to the model_builder service.

    Parameters
    ----------
    url:
        Base URL of the model builder service.
    features:
        Training feature vectors.
    labels:
        Corresponding labels.

    Returns
    -------
    bool
        True on success, False otherwise.
    """
    payload = {"features": features, "labels": labels}
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.post(f"{url.rstrip('/')}/train", json=payload, timeout=5.0)
        if response.status_code == 200:
            return True
        logger.error("Model training failed: HTTP %s", response.status_code)
    except httpx.HTTPError as exc:  # pragma: no cover - network errors
        logger.error("Model training request error: %s", exc)
    return False


async def retrain(url: str) -> Optional[int]:
    """Retrain the model and return the new version number."""
    global _MODEL_VERSION
    # Minimal training payload; real implementation would use real data
    if await train(url, [[0.0], [1.0]], [0, 1]):
        _MODEL_VERSION += 1
        logger.info("Model retrained, new version %s", _MODEL_VERSION)
        return _MODEL_VERSION
    return None


async def _retrain_loop(url: str, interval: float) -> None:
    while True:
        await retrain(url)
        await asyncio.sleep(interval)


def schedule_retrain(url: str, interval: float) -> asyncio.Task:
    """Start periodic retraining task."""
    return asyncio.create_task(_retrain_loop(url, interval))
