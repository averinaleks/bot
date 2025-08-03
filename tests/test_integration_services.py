import os
import requests
import multiprocessing
import socket
import signal
from flask import Flask, request, jsonify
import pytest

from tests.helpers import wait_for_service

# Ensure processes use the spawn start method on all platforms
multiprocessing.set_start_method("spawn", force=True)
ctx = multiprocessing.get_context("spawn")


def _get_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port




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
    host = os.environ.get("HOST", "0.0.0.0")
    dh_app.run(host=host, port=port)


def _run_mb(port: int):
    signal.signal(signal.SIGTERM, _shutdown)
    host = os.environ.get("HOST", "0.0.0.0")
    mb_app.run(host=host, port=port)


def _run_tm(port: int):
    signal.signal(signal.SIGTERM, _shutdown)
    host = os.environ.get("HOST", "0.0.0.0")
    tm_app.run(host=host, port=port)


@pytest.mark.integration
def test_services_communicate():
    from bot import trading_bot  # noqa: E402
    os.environ['HOST'] = '127.0.0.1'
    dh_port = _get_free_port()
    mb_port = _get_free_port()
    tm_port = _get_free_port()
    processes = [
        ctx.Process(target=_run_dh, args=(dh_port,)),
        ctx.Process(target=_run_mb, args=(mb_port,)),
        ctx.Process(target=_run_tm, args=(tm_port,)),
    ]
    try:
        for p in processes:
            p.start()
        for url in (
            f'http://localhost:{dh_port}/ping',
            f'http://localhost:{mb_port}/ping',
            f'http://localhost:{tm_port}/ready',
        ):
            wait_for_service(url)
        os.environ.update({
            'DATA_HANDLER_URL': f'http://localhost:{dh_port}',
            'MODEL_BUILDER_URL': f'http://localhost:{mb_port}',
            'TRADE_MANAGER_URL': f'http://localhost:{tm_port}',
        })
        trading_bot.run_once()
        resp = requests.get(f'http://localhost:{tm_port}/positions', timeout=5)
        data = resp.json()
        assert data['positions'], 'position was not created'
    finally:
        for p in processes:
            p.terminate()
            p.join()
            assert p.exitcode == 0


@pytest.mark.integration
def test_service_availability_check():
    from bot import trading_bot  # noqa: E402
    os.environ['HOST'] = '127.0.0.1'
    dh_port = _get_free_port()
    mb_port = _get_free_port()
    tm_port = _get_free_port()
    processes = [
        ctx.Process(target=_run_dh, args=(dh_port,)),
        ctx.Process(target=_run_mb, args=(mb_port,)),
        ctx.Process(target=_run_tm, args=(tm_port,)),
    ]
    try:
        for p in processes:
            p.start()
        for url in (
            f'http://localhost:{dh_port}/ping',
            f'http://localhost:{mb_port}/ping',
            f'http://localhost:{tm_port}/ready',
        ):
            wait_for_service(url)
        resp = requests.get(f'http://localhost:{dh_port}/ping', timeout=5)
        assert resp.status_code == 200
        resp = requests.get(f'http://localhost:{mb_port}/ping', timeout=5)
        assert resp.status_code == 200
        resp = requests.get(f'http://localhost:{tm_port}/ready', timeout=5)
        assert resp.status_code == 200
    finally:
        for p in processes:
            p.terminate()
            p.join()
            assert p.exitcode == 0


@pytest.mark.integration
def test_check_services_success():
    from bot import trading_bot  # noqa: E402
    os.environ['HOST'] = '127.0.0.1'
    dh_port = _get_free_port()
    mb_port = _get_free_port()
    tm_port = _get_free_port()
    processes = [
        ctx.Process(target=_run_dh, args=(dh_port,)),
        ctx.Process(target=_run_mb, args=(mb_port,)),
        ctx.Process(target=_run_tm, args=(tm_port,)),
    ]
    try:
        for p in processes:
            p.start()
        for url in (
            f'http://localhost:{dh_port}/ping',
            f'http://localhost:{mb_port}/ping',
            f'http://localhost:{tm_port}/ready',
        ):
            wait_for_service(url)
        os.environ.update({
            'DATA_HANDLER_URL': f'http://localhost:{dh_port}',
            'MODEL_BUILDER_URL': f'http://localhost:{mb_port}',
            'TRADE_MANAGER_URL': f'http://localhost:{tm_port}',
            'SERVICE_CHECK_RETRIES': '2',
            'SERVICE_CHECK_DELAY': '0.1',
        })
        trading_bot.check_services()
    finally:
        for p in processes:
            p.terminate()
            p.join()
            assert p.exitcode == 0


@pytest.mark.integration
def test_check_services_failure():
    from bot import trading_bot  # noqa: E402
    os.environ['HOST'] = '127.0.0.1'
    dh_port = _get_free_port()
    mb_port = _get_free_port()
    tm_port = _get_free_port()
    processes = [
        ctx.Process(target=_run_dh, args=(dh_port,)),
        ctx.Process(target=_run_mb, args=(mb_port,)),
    ]
    try:
        for p in processes:
            p.start()
        for url in (
            f'http://localhost:{dh_port}/ping',
            f'http://localhost:{mb_port}/ping',
        ):
            wait_for_service(url)
        os.environ.update({
            'DATA_HANDLER_URL': f'http://localhost:{dh_port}',
            'MODEL_BUILDER_URL': f'http://localhost:{mb_port}',
            'TRADE_MANAGER_URL': f'http://localhost:{tm_port}',
            'SERVICE_CHECK_RETRIES': '2',
            'SERVICE_CHECK_DELAY': '0.1',
        })
        with pytest.raises(SystemExit):
            trading_bot.check_services()
    finally:
        for p in processes:
            p.terminate()
            p.join()
            assert p.exitcode == 0


@pytest.mark.integration
def test_check_services_host_only():
    from bot import trading_bot  # noqa: E402
    os.environ['HOST'] = '127.0.0.1'
    for var in ('DATA_HANDLER_URL', 'MODEL_BUILDER_URL', 'TRADE_MANAGER_URL'):
        os.environ.pop(var, None)
    os.environ.update({
        'SERVICE_CHECK_RETRIES': '2',
        'SERVICE_CHECK_DELAY': '0.1',
    })
    dh_port = _get_free_port()
    mb_port = _get_free_port()
    tm_port = _get_free_port()
    processes = [
        ctx.Process(target=_run_dh, args=(dh_port,)),
        ctx.Process(target=_run_mb, args=(mb_port,)),
        ctx.Process(target=_run_tm, args=(tm_port,)),
    ]
    try:
        for p in processes:
            p.start()
        for url in (
            f'http://localhost:{dh_port}/ping',
            f'http://localhost:{mb_port}/ping',
            f'http://localhost:{tm_port}/ready',
        ):
            wait_for_service(url)
        os.environ.update({
            'DATA_HANDLER_URL': f'http://localhost:{dh_port}',
            'MODEL_BUILDER_URL': f'http://localhost:{mb_port}',
            'TRADE_MANAGER_URL': f'http://localhost:{tm_port}',
        })
        trading_bot.check_services()
    finally:
        for p in processes:
            p.terminate()
            p.join()
            assert p.exitcode == 0
