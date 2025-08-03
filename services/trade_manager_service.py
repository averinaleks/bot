"""Minimal trade manager service using Bybit via ccxt.

This reference service exposes endpoints to open and close positions on
Bybit. API keys are taken from ``BYBIT_API_KEY`` and ``BYBIT_API_SECRET``
environment variables.
"""

from flask import Flask, request, jsonify
import ccxt
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)

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
    price = float(data.get('price', 0) or 0)
    amount = float(data.get('amount', 0))
    tp = data.get('tp')
    sl = data.get('sl')
    tp = float(tp) if tp is not None else None
    sl = float(sl) if sl is not None else None
    if amount <= 0:
        risk_usd = float(os.getenv('TRADE_RISK_USD', '0') or 0)
        if risk_usd > 0 and price > 0:
            amount = risk_usd / price
    if not symbol or amount <= 0:
        return jsonify({'error': 'invalid order'}), 400
    try:
        if (tp is not None or sl is not None) and hasattr(
            exchange, 'create_order_with_take_profit_and_stop_loss'
        ):
            order = exchange.create_order_with_take_profit_and_stop_loss(
                symbol, 'market', side, amount, None, tp, sl, None
            )
        else:
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
        for i, rec in enumerate(POSITIONS):
            if (
                rec.get('symbol') == symbol
                and rec.get('side') == data.get('side')
                and rec.get('action') == 'open'
            ):
                POSITIONS.pop(i)
                break
        return jsonify({'status': 'ok', 'order_id': order.get('id')})
    except Exception as exc:  # pragma: no cover - network errors
        logging.exception("Error closing position")
        return jsonify({'error': 'An internal error has occurred.'}), 500


@app.route('/positions')
def positions():
    return jsonify({'positions': POSITIONS})


@app.route('/ping')
def ping():
    return jsonify({'status': 'ok'})


@app.route('/ready')
def ready():
    """Health check endpoint used by docker-compose."""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8002'))
    host = os.environ.get('HOST', '0.0.0.0')
    app.run(host=host, port=port)
