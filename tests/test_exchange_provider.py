from __future__ import annotations

import threading

from services.exchange_provider import ExchangeProvider


def test_exchange_provider_lazy_initialization_and_close():
    created = []
    closed = []

    def factory():
        obj = object()
        created.append(obj)
        return obj

    provider: ExchangeProvider[object] = ExchangeProvider(factory, close=closed.append)

    first = provider.get()
    second = provider.get()

    assert first is second
    assert len(created) == 1

    provider.close()
    assert closed == [first]

    third = provider.get()
    assert third is not first
    assert len(created) == 2


def test_exchange_provider_thread_safe_initialization():
    barrier = threading.Barrier(5)
    created = []

    def factory():
        obj = object()
        created.append(obj)
        return obj

    provider: ExchangeProvider[object] = ExchangeProvider(factory)
    results: list[object] = []

    def worker():
        barrier.wait()
        results.append(provider.get())

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(created) == 1
    assert len(set(id(result) for result in results)) == 1


def test_exchange_provider_override_for_tests():
    provider: ExchangeProvider[object] = ExchangeProvider(object)

    fake = object()
    provider.override(fake)
    assert provider.get() is fake

    provider.override(None)
    new_obj = provider.get()
    assert new_obj is not fake
