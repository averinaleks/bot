import io
import threading
from bot import config


def test_load_defaults_thread_safe(monkeypatch):
    calls = 0

    def fake_open(*args, **kwargs):
        nonlocal calls
        calls += 1
        return io.StringIO("{}")

    monkeypatch.setattr(config, "DEFAULTS", None)
    monkeypatch.setattr("builtins.open", fake_open)

    barrier = threading.Barrier(5)

    def worker():
        barrier.wait()
        config.load_defaults()

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls == 1
