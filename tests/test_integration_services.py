import os
import time
import requests
import multiprocessing
from flask import Flask, request, jsonify
import pytest

# Ensure processes use the spawn start method on all platforms
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


def _run_dh():
    host = os.environ.get("HOST", "0.0.0.0")
    dh_app.run(host=host, port=8000)


def _run_mb():
    host = os.environ.get("HOST", "0.0.0.0")
    mb_app.run(host=host, port=8001)


def _run_tm():
    host = os.environ.get("HOST", "0.0.0.0")
    tm_app.run(host=host, port=8002)


def test_services_communicate():
    import trading_bot  # noqa: E402
    os.environ['HOST'] = '127.0.0.1'
    processes = [
        ctx.Process(target=_run_dh),
        ctx.Process(target=_run_mb),
        ctx.Process(target=_run_tm),
    ]
    for p in processes:
        p.start()
    time.sleep(1)
    os.environ.update({
        'DATA_HANDLER_URL': 'http://localhost:8000',
        'MODEL_BUILDER_URL': 'http://localhost:8001',
        'TRADE_MANAGER_URL': 'http://localhost:8002',
    })
    try:
        trading_bot.run_once()
        resp = requests.get('http://localhost:8002/positions', timeout=5)
        data = resp.json()
        assert data['positions'], 'position was not created'
    finally:
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()


def test_service_availability_check():
    import trading_bot  # noqa: E402
    os.environ['HOST'] = '127.0.0.1'
    processes = [
        ctx.Process(target=_run_dh),
        ctx.Process(target=_run_mb),
        ctx.Process(target=_run_tm),
    ]
    for p in processes:
        p.start()
    time.sleep(1)
    try:
        resp = requests.get('http://localhost:8000/ping', timeout=5)
        assert resp.status_code == 200
        resp = requests.get('http://localhost:8001/ping', timeout=5)
        assert resp.status_code == 200
        resp = requests.get('http://localhost:8002/ready', timeout=5)
        assert resp.status_code == 200
    finally:
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()


def test_check_services_success():
    import trading_bot  # noqa: E402
    os.environ['HOST'] = '127.0.0.1'
    processes = [
        ctx.Process(target=_run_dh),
        ctx.Process(target=_run_mb),
        ctx.Process(target=_run_tm),
    ]
    for p in processes:
        p.start()
    time.sleep(1)
    os.environ.update({
        'DATA_HANDLER_URL': 'http://localhost:8000',
        'MODEL_BUILDER_URL': 'http://localhost:8001',
        'TRADE_MANAGER_URL': 'http://localhost:8002',
        'SERVICE_CHECK_RETRIES': '2',
        'SERVICE_CHECK_DELAY': '0.1',
    })
    try:
        trading_bot.check_services()
    finally:
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()


def test_check_services_failure():
    import trading_bot  # noqa: E402
    os.environ['HOST'] = '127.0.0.1'
    processes = [
        ctx.Process(target=_run_dh),
        ctx.Process(target=_run_mb),
    ]
    for p in processes:
        p.start()
    time.sleep(1)
    os.environ.update({
        'DATA_HANDLER_URL': 'http://localhost:8000',
        'MODEL_BUILDER_URL': 'http://localhost:8001',
        'TRADE_MANAGER_URL': 'http://localhost:8002',
        'SERVICE_CHECK_RETRIES': '2',
        'SERVICE_CHECK_DELAY': '0.1',
    })
    try:
        with pytest.raises(SystemExit):
            trading_bot.check_services()
    finally:
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()


def test_check_services_host_only():
    import trading_bot  # noqa: E402
    os.environ['HOST'] = '127.0.0.1'
    for var in ('DATA_HANDLER_URL', 'MODEL_BUILDER_URL', 'TRADE_MANAGER_URL'):
        os.environ.pop(var, None)
    os.environ.update({
        'SERVICE_CHECK_RETRIES': '2',
        'SERVICE_CHECK_DELAY': '0.1',
    })
    processes = [
        ctx.Process(target=_run_dh),
        ctx.Process(target=_run_mb),
        ctx.Process(target=_run_tm),
    ]
    for p in processes:
        p.start()
    time.sleep(1)
    try:
        trading_bot.check_services()
    finally:
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
