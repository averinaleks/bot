import os
import time
import requests
import multiprocessing
from flask import Flask, request, jsonify
import trading_bot


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
    os.environ.update({
        'DATA_HANDLER_URL': 'http://localhost:8000',
        'MODEL_BUILDER_URL': 'http://localhost:8001',
        'TRADE_MANAGER_URL': 'http://localhost:8002',
    })
    try:
        assert trading_bot.check_services(), 'services not reachable'
    finally:
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
