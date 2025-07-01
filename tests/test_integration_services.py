import os
import time
import requests
import multiprocessing
from flask import Flask, request, jsonify
import sys
import types
import pytest

# Ensure the project root is on the Python path so that 'trading_bot' can be imported
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Stub heavy dependencies before importing trading_bot
numba_mod = types.ModuleType('numba')
numba_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
numba_mod.jit = lambda *a, **k: (lambda f: f)
numba_mod.prange = range
sys.modules.setdefault('numba', numba_mod)
sys.modules.setdefault('numba.cuda', numba_mod.cuda)
sys.modules.setdefault('httpx', types.ModuleType('httpx'))
telegram_error_mod = types.ModuleType('telegram.error')
telegram_error_mod.RetryAfter = Exception
sys.modules.setdefault('telegram', types.ModuleType('telegram'))
sys.modules.setdefault('telegram.error', telegram_error_mod)
pybit_mod = types.ModuleType('pybit')
ut_mod = types.ModuleType('unified_trading')
ut_mod.HTTP = object
pybit_mod.unified_trading = ut_mod
sys.modules.setdefault('pybit', pybit_mod)
sys.modules.setdefault('pybit.unified_trading', ut_mod)
psutil_mod = types.ModuleType('psutil')
psutil_mod.cpu_percent = lambda interval=1: 0
psutil_mod.virtual_memory = lambda: type('mem', (), {'percent': 0})
sys.modules.setdefault('psutil', psutil_mod)

import trading_bot  # noqa: E402


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
    price = float(data.get('price', 0))
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


def _run(app, port):
    app.run(port=port)


def test_services_communicate():
    processes = [
        multiprocessing.Process(target=_run, args=(dh_app, 8000)),
        multiprocessing.Process(target=_run, args=(mb_app, 8001)),
        multiprocessing.Process(target=_run, args=(tm_app, 8002)),
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
    processes = [
        multiprocessing.Process(target=_run, args=(dh_app, 8000)),
        multiprocessing.Process(target=_run, args=(mb_app, 8001)),
        multiprocessing.Process(target=_run, args=(tm_app, 8002)),
    ]
    for p in processes:
        p.start()
    time.sleep(1)
    try:
        for port in (8000, 8001, 8002):
            resp = requests.get(f'http://localhost:{port}/ping', timeout=5)
            assert resp.status_code == 200
    finally:
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()


def test_check_services_success():
    processes = [
        multiprocessing.Process(target=_run, args=(dh_app, 8000)),
        multiprocessing.Process(target=_run, args=(mb_app, 8001)),
        multiprocessing.Process(target=_run, args=(tm_app, 8002)),
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
    processes = [
        multiprocessing.Process(target=_run, args=(dh_app, 8000)),
        multiprocessing.Process(target=_run, args=(mb_app, 8001)),
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
