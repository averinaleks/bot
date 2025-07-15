"""Simple reference data handler service fetching real prices from Bybit.

This version also exposes cached OHLCV data for basic backtesting or model
training.  Prices are refreshed periodically in the background for any symbols
listed in ``STREAM_SYMBOLS``.  Historical bars can be retrieved via the new
``/ohlcv/<symbol>`` endpoint.  The cache lifetime is controlled by
``CACHE_TTL``.
"""
from flask import Flask, jsonify
import ccxt
import os
import time
import threading
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

exchange = ccxt.bybit({
    'apiKey': os.getenv('BYBIT_API_KEY', ''),
    'secret': os.getenv('BYBIT_API_SECRET', ''),
})

TIMEFRAME = os.getenv('TIMEFRAME', '1m')
OHLCV_LIMIT = int(os.getenv('OHLCV_LIMIT', '100'))
CACHE_TTL = int(os.getenv('CACHE_TTL', '60'))
STREAM_SYMBOLS = [s for s in os.getenv('STREAM_SYMBOLS', '').split(',') if s]
UPDATE_INTERVAL = int(os.getenv('UPDATE_INTERVAL', '60'))

_cache = {}


def _update_symbol(symbol: str) -> None:
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=OHLCV_LIMIT)
        _cache[symbol] = {'timestamp': time.time(), 'data': data}
    except Exception:  # pragma: no cover - network errors
        pass


def _background_worker():
    while True:
        for sym in STREAM_SYMBOLS:
            _update_symbol(sym)
        time.sleep(UPDATE_INTERVAL)


if STREAM_SYMBOLS:
    thread = threading.Thread(target=_background_worker, daemon=True)
    thread.start()

@app.route('/price/<symbol>')
def price(symbol: str):
    try:
        ticker = exchange.fetch_ticker(symbol)
        last = float(ticker.get('last') or 0.0)
    except Exception as exc:  # pragma: no cover - network errors
        last = 0.0
    return jsonify({'price': last})


@app.route('/ohlcv/<symbol>')
def ohlcv(symbol: str):
    now = time.time()
    cached = _cache.get(symbol)
    if not cached or now - cached['timestamp'] > CACHE_TTL:
        _update_symbol(symbol)
        cached = _cache.get(symbol, {'data': []})
    return jsonify({'ohlcv': cached['data']})

@app.route('/ping')
def ping():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8000'))
    host = os.environ.get('HOST', '0.0.0.0')
    app.run(host=host, port=port)
