"""Simple reference data handler service fetching real prices from Bybit."""
from flask import Flask, jsonify
import ccxt
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

exchange = ccxt.bybit({
    'apiKey': os.getenv('BYBIT_API_KEY', ''),
    'secret': os.getenv('BYBIT_API_SECRET', ''),
})

# Correct price endpoint without trailing whitespace
@app.route('/price/<symbol>')
def price(symbol: str):
    try:
        ticker = exchange.fetch_ticker(symbol)
        last = float(ticker.get('last') or 0.0)
        return jsonify({'price': last})
    except Exception as exc:  # pragma: no cover - network errors
        # Surface exchange errors to clients so callers can react accordingly.
        return jsonify({'error': str(exc)}), 503

@app.route('/ping')
def ping():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8000'))
    host = os.environ.get('HOST', '0.0.0.0')
    app.run(host=host, port=port)
