import os
import requests
import multiprocessing
import sys
import signal
from contextlib import ExitStack
from flask import Flask, request, jsonify
import pytest

from tests.helpers import get_free_port, service_process

if sys.platform == "win32":
    multiprocessing.set_start_method("spawn", force=True)
ctx = multiprocessing.get_context("spawn")






# Minimal stubs for services to avoid heavy dependencies
dh_app = Flask('data_handler')

@dh_app.route('/price/<symbol>')
def price(symbol: str):
    return jsonify({'price': 100.0})

@dh_app.route('/ping')
def dh_ping():
    return jsonify({'status': 'ok'})


mb_app = Flask('model_builder')

@mb_app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    assert 'features' in data
    features = data.get('features') or []
    price = float(features[0]) if isinstance(features, list) and features else 0.0
    signal = 'buy' if price > 0 else None
    return jsonify({'signal': signal})

@mb_app.route('/ping')
def mb_ping():
    return jsonify({'status': 'ok'})


tm_app = Flask('trade_manager')
POSITIONS = []

@tm_app.route('/open_position', methods=['POST'])
def open_position():
    info = request.get_json(force=True)
    POSITIONS.append(info)
    return jsonify({'status': 'ok'})

@tm_app.route('/positions')
def positions_route():
    return jsonify({'positions': POSITIONS})

@tm_app.route('/ping')
def tm_ping():
    return jsonify({'status': 'ok'})


@tm_app.route('/ready')
def tm_ready():
    return jsonify({'status': 'ok'})


def _shutdown(*_):
    raise SystemExit(0)


def _run_dh(port: int):
    signal.signal(signal.SIGTERM, _shutdown)
    host = os.environ.get("HOST", "127.0.0.1")
    dh_app.run(host=host, port=port)


def _run_mb(port: int):
    signal.signal(signal.SIGTERM, _shutdown)
    host = os.environ.get("HOST", "127.0.0.1")
    mb_app.run(host=host, port=port)


def _run_tm(port: int):
    signal.signal(signal.SIGTERM, _shutdown)
    host = os.environ.get("HOST", "127.0.0.1")
    tm_app.run(host=host, port=port)


@pytest.mark.integration
def test_services_communicate(monkeypatch):
    from bot import trading_bot  # noqa: E402
    monkeypatch.setenv('HOST', '127.0.0.1')
    dh_port = get_free_port()
    mb_port = get_free_port()
    tm_port = get_free_port()
    with ExitStack() as stack:
        stack.enter_context(
            service_process(ctx.Process(target=_run_dh, args=(dh_port,)), url=f'http://localhost:{dh_port}/ping')
        )
        stack.enter_context(
            service_process(ctx.Process(target=_run_mb, args=(mb_port,)), url=f'http://localhost:{mb_port}/ping')
        )
        stack.enter_context(
            service_process(ctx.Process(target=_run_tm, args=(tm_port,)), url=f'http://localhost:{tm_port}/ready')
        )
        monkeypatch.setenv('DATA_HANDLER_URL', f'http://localhost:{dh_port}')
        monkeypatch.setenv('MODEL_BUILDER_URL', f'http://localhost:{mb_port}')
        monkeypatch.setenv('TRADE_MANAGER_URL', f'http://localhost:{tm_port}')
        trading_bot.run_once()
        resp = requests.get(f'http://localhost:{tm_port}/positions', timeout=5)
        data = resp.json()
        assert data['positions'], 'position was not created'


@pytest.mark.integration
def test_service_availability_check(monkeypatch):
    from bot import trading_bot  # noqa: E402
    monkeypatch.setenv('HOST', '127.0.0.1')
    dh_port = get_free_port()
    mb_port = get_free_port()
    tm_port = get_free_port()
    with ExitStack() as stack:
        stack.enter_context(
            service_process(ctx.Process(target=_run_dh, args=(dh_port,)), url=f'http://localhost:{dh_port}/ping')
        )
        stack.enter_context(
            service_process(ctx.Process(target=_run_mb, args=(mb_port,)), url=f'http://localhost:{mb_port}/ping')
        )
        stack.enter_context(
            service_process(ctx.Process(target=_run_tm, args=(tm_port,)), url=f'http://localhost:{tm_port}/ready')
        )
        resp = requests.get(f'http://localhost:{dh_port}/ping', timeout=5)
        assert resp.status_code == 200
        resp = requests.get(f'http://localhost:{mb_port}/ping', timeout=5)
        assert resp.status_code == 200
        resp = requests.get(f'http://localhost:{tm_port}/ready', timeout=5)
        assert resp.status_code == 200


@pytest.mark.integration
def test_check_services_success(monkeypatch):
    from bot import trading_bot  # noqa: E402
    monkeypatch.setenv('HOST', '127.0.0.1')
    dh_port = get_free_port()
    mb_port = get_free_port()
    tm_port = get_free_port()
    with ExitStack() as stack:
        stack.enter_context(
            service_process(ctx.Process(target=_run_dh, args=(dh_port,)), url=f'http://localhost:{dh_port}/ping')
        )
        stack.enter_context(
            service_process(ctx.Process(target=_run_mb, args=(mb_port,)), url=f'http://localhost:{mb_port}/ping')
        )
        stack.enter_context(
            service_process(ctx.Process(target=_run_tm, args=(tm_port,)), url=f'http://localhost:{tm_port}/ready')
        )
        monkeypatch.setenv('DATA_HANDLER_URL', f'http://localhost:{dh_port}')
        monkeypatch.setenv('MODEL_BUILDER_URL', f'http://localhost:{mb_port}')
        monkeypatch.setenv('TRADE_MANAGER_URL', f'http://localhost:{tm_port}')
        monkeypatch.setenv('SERVICE_CHECK_RETRIES', '2')
        monkeypatch.setenv('SERVICE_CHECK_DELAY', '0.1')
        trading_bot.check_services()


@pytest.mark.integration
def test_check_services_failure(monkeypatch):
    from bot import trading_bot  # noqa: E402
    monkeypatch.setenv('HOST', '127.0.0.1')
    dh_port = get_free_port()
    mb_port = get_free_port()
    tm_port = get_free_port()
    with ExitStack() as stack:
        stack.enter_context(
            service_process(ctx.Process(target=_run_dh, args=(dh_port,)), url=f'http://localhost:{dh_port}/ping')
        )
        stack.enter_context(
            service_process(ctx.Process(target=_run_mb, args=(mb_port,)), url=f'http://localhost:{mb_port}/ping')
        )
        monkeypatch.setenv('DATA_HANDLER_URL', f'http://localhost:{dh_port}')
        monkeypatch.setenv('MODEL_BUILDER_URL', f'http://localhost:{mb_port}')
        monkeypatch.setenv('TRADE_MANAGER_URL', f'http://localhost:{tm_port}')
        monkeypatch.setenv('SERVICE_CHECK_RETRIES', '2')
        monkeypatch.setenv('SERVICE_CHECK_DELAY', '0.1')
        with pytest.raises(SystemExit):
            trading_bot.check_services()


@pytest.mark.integration
def test_check_services_host_only(monkeypatch):
    from bot import trading_bot  # noqa: E402
    monkeypatch.setenv('HOST', '127.0.0.1')
    for var in ('DATA_HANDLER_URL', 'MODEL_BUILDER_URL', 'TRADE_MANAGER_URL'):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv('SERVICE_CHECK_RETRIES', '2')
    monkeypatch.setenv('SERVICE_CHECK_DELAY', '0.1')
    dh_port = get_free_port()
    mb_port = get_free_port()
    tm_port = get_free_port()
    with ExitStack() as stack:
        stack.enter_context(
            service_process(ctx.Process(target=_run_dh, args=(dh_port,)), url=f'http://localhost:{dh_port}/ping')
        )
        stack.enter_context(
            service_process(ctx.Process(target=_run_mb, args=(mb_port,)), url=f'http://localhost:{mb_port}/ping')
        )
        stack.enter_context(
            service_process(ctx.Process(target=_run_tm, args=(tm_port,)), url=f'http://localhost:{tm_port}/ready')
        )
        monkeypatch.setenv('DATA_HANDLER_URL', f'http://localhost:{dh_port}')
        monkeypatch.setenv('MODEL_BUILDER_URL', f'http://localhost:{mb_port}')
        monkeypatch.setenv('TRADE_MANAGER_URL', f'http://localhost:{tm_port}')
        trading_bot.check_services()
