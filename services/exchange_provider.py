"""Thread-safe lazy provider for exchange-like singletons."""

from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from typing import Callable, Generator, Generic, TypeVar

__all__ = ["ExchangeProvider"]

T = TypeVar("T")


class ExchangeProvider(Generic[T]):
    """Provide a lazily constructed singleton instance with safe reuse.

    The provider wraps a ``factory`` callable that creates the underlying
    exchange instance on demand. The resulting object is cached until
    :meth:`close` is called, at which point an optional ``close`` callback is
    invoked and the cached instance is discarded.

    The implementation is intentionally minimal yet thread-safe so it can be
    reused both in Flask's multithreaded environment and in unit tests that
    spawn concurrent requests.
    """

    def __init__(
        self,
        factory: Callable[[], T],
        *,
        close: Callable[[T], None] | None = None,
    ) -> None:
        self._factory = factory
        self._close_cb = close
        self._instance: T | None = None
        self._lock = RLock()

    def get(self) -> T:
        """Return the cached instance, creating it on first access."""

        instance = self._instance
        if instance is not None:
            return instance

        with self._lock:
            if self._instance is None:
                self._instance = self._factory()
            return self._instance

    def peek(self) -> T | None:
        """Return the cached instance without triggering initialization."""

        return self._instance

    def create(self) -> T:
        """Create a brand-new exchange instance without caching it."""

        return self._factory()

    def close_instance(self, instance: T) -> None:
        """Close a specific instance produced by the factory."""

        if self._close_cb is not None:
            self._close_cb(instance)

    def close(self) -> None:
        """Dispose of the cached instance if present."""

        instance: T | None
        with self._lock:
            instance = self._instance
            self._instance = None

        if instance is not None:
            self.close_instance(instance)

    @contextmanager
    def lifespan(self) -> Generator[T, None, None]:
        """Context manager that yields the active instance.

        Upon exiting the context the cached instance is automatically
        discarded. This helper is mainly used in tests to verify that the
        provider cleans up correctly.
        """

        instance = self.get()
        try:
            yield instance
        finally:
            self.close()

    def override(self, instance: T | None) -> None:
        """Manually set the cached instance for tests.

        Passing ``None`` clears the provider without executing the close
        callback. This method is intentionally lightweight and is only used
        inside unit tests that monkeypatch service dependencies.
        """

        with self._lock:
            self._instance = instance

