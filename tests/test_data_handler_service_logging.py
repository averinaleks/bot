import logging
import importlib
import sys
import types


def test_data_handler_service_does_not_configure_logging_on_import(monkeypatch):
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.setLevel(logging.WARNING)
    assert root.handlers == []

    ccxt = types.ModuleType('ccxt')
    ccxt.bybit = lambda *a, **kw: types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'ccxt', ccxt)

    flask = types.ModuleType('flask')

    class DummyFlask:
        def __init__(self, name):
            self.config = {}

        def route(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

        def errorhandler(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

        def before_request(self, *args, **kwargs):
            """No-op decorator."""
            def decorator(func):
                return func
            return decorator

        def before_first_request(self, *args, **kwargs):
            """No-op decorator."""
            def decorator(func):
                return func
            return decorator

        def run(self, *args, **kwargs):
            pass

    flask.Flask = DummyFlask
    flask.jsonify = lambda obj: obj
    flask.request = types.SimpleNamespace(endpoint='ping')
    monkeypatch.setitem(sys.modules, 'flask', flask)

    monkeypatch.delitem(sys.modules, 'bot.services.data_handler_service', raising=False)
    importlib.import_module('bot.services.data_handler_service')

    assert root.level == logging.WARNING
    assert root.handlers == []
