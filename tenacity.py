"""Minimal tenacity stub providing retry decorator for tests."""

from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, Callable, Coroutine

class stop_after_attempt:
    def __init__(self, attempt_number: int):
        self.attempt_number = attempt_number

class wait_exponential:
    def __init__(self, multiplier: float = 1, min: float = 0, max: float | None = None):
        self.multiplier = multiplier
        self.min = min
        self.max = max

    def __call__(self, attempt: int) -> float:
        delay = self.multiplier * (2 ** (attempt - 1))
        if delay < self.min:
            delay = self.min
        if self.max is not None:
            delay = min(delay, self.max)
        return delay

def retry(*, stop: stop_after_attempt, wait: wait_exponential, reraise: bool = True) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                attempt = 0
                while True:
                    try:
                        return await func(*args, **kwargs)
                    except Exception:
                        attempt += 1
                        if attempt >= stop.attempt_number:
                            raise
                        await asyncio.sleep(wait(attempt))
            return async_wrapper
        else:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                attempt = 0
                while True:
                    try:
                        return func(*args, **kwargs)
                    except Exception:
                        attempt += 1
                        if attempt >= stop.attempt_number:
                            raise
                        time.sleep(wait(attempt))
            return wrapper
    return decorator

__all__ = ["retry", "stop_after_attempt", "wait_exponential"]
