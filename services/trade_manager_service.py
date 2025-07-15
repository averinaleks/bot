"""Minimal trade manager service using Bybit via ccxt.

This reference service exposes endpoints to open and close positions on
Bybit. API keys are taken from ``BYBIT_API_KEY`` and ``BYBIT_API_SECRET``
environment variables.
"""

from flask import Flask, request, jsonify
import ccxt
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

exchange = ccxt.bybit({
    'apiKey': os.getenv('BYBIT_API_KEY', ''),
    'secret': os.getenv('BYBIT_API_SECRET', ''),
})

POSITIONS: list[dict] = []


def _record(order: dict, symbol: str, side: str, amount: float, action: str) -> None:
    POSITIONS.append({
        'id': order.get('id'),
        'symbol': symbol,
        'side': side,
        'amount': amount,
        'action': action,
    })


@app.route('/open_position', methods=['POST'])
def open_position() -> tuple:
    data = request.get_json(force=True)
    symbol = data.get('symbol')
    side = str(data.get('side', 'buy')).lower()
    amount = float(data.get('amount', 0))
    if not symbol or amount <= 0:
        return jsonify({'error': 'invalid order'}), 400
    try:
        order = exchange.create_order(symbol, 'market', side, amount)
        _record(order, symbol, side, amount, 'open')
        return jsonify({'status': 'ok', 'order_id': order.get('id')})
    except Exception as exc:  # pragma: no cover - network errors
        return jsonify({'error': str(exc)}), 500


@app.route('/close_position', methods=['POST'])
def close_position() -> tuple:
    data = request.get_json(force=True)
    symbol = data.get('symbol')
    side = str(data.get('side', 'buy')).lower()
    side = 'sell' if side == 'buy' else 'buy'
    amount = float(data.get('amount', 0))
    if not symbol or amount <= 0:
        return jsonify({'error': 'invalid order'}), 400
    params = {'reduce_only': True}
    try:
        order = exchange.create_order(symbol, 'market', side, amount, params=params)
        _record(order, symbol, side, amount, 'close')
        return jsonify({'status': 'ok', 'order_id': order.get('id')})
    except Exception as exc:  # pragma: no cover - network errors
        return jsonify({'error': str(exc)}), 500


@app.route('/positions')
def positions():
    return jsonify({'positions': POSITIONS})


@app.route('/ping')
def ping():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8002'))
    host = os.environ.get('HOST', '0.0.0.0')
    app.run(host=host, port=port)
