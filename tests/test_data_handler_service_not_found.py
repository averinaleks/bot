import importlib
import sys
import types


def test_unknown_route_returns_404(monkeypatch):
    class DummyExchange:
        def fetch_ticker(self, symbol):
            return {'last': 1.0}

    ccxt = types.ModuleType("ccxt")
    ccxt.bybit = lambda *args, **kwargs: DummyExchange()
    monkeypatch.setitem(sys.modules, "ccxt", ccxt)

    monkeypatch.delitem(sys.modules, "bot.services.data_handler_service", raising=False)
    monkeypatch.delitem(sys.modules, "services.data_handler_service", raising=False)
    monkeypatch.delitem(sys.modules, "flask", raising=False)
    importlib.import_module("flask")
    import bot.services as services_pkg
    monkeypatch.delattr(services_pkg, "data_handler_service", raising=False)
    data_handler_service = importlib.import_module("bot.services.data_handler_service")

    client = data_handler_service.app.test_client()
    resp = client.get("/missing")
    assert resp.status_code == 404
