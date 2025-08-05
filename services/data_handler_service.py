"""Simple reference data handler service fetching real prices from Bybit."""
from flask import Flask, jsonify
import logging
import ccxt
import os
from dotenv import load_dotenv

load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

exchange = ccxt.bybit({
    'apiKey': os.getenv('BYBIT_API_KEY', ''),
    'secret': os.getenv('BYBIT_API_SECRET', ''),
})

CCXT_BASE_ERROR = getattr(ccxt, 'BaseError', Exception)
CCXT_NETWORK_ERROR = getattr(ccxt, 'NetworkError', CCXT_BASE_ERROR)

# Correct price endpoint without trailing whitespace
@app.route('/price/<symbol>')
def price(symbol: str):
    try:
        ticker = exchange.fetch_ticker(symbol)
        last = float(ticker.get('last') or 0.0)
        return jsonify({'price': last})
    except CCXT_NETWORK_ERROR as exc:  # pragma: no cover - network errors
        logging.exception("Network error fetching price for '%s': %s", symbol, exc)
        return jsonify({'error': 'network error contacting exchange'}), 503
    except CCXT_BASE_ERROR as exc:
        logging.exception("Exchange error fetching price for '%s': %s", symbol, exc)
        return jsonify({'error': 'exchange error fetching price'}), 502

@app.route('/ping')
def ping():
    return jsonify({'status': 'ok'})


@app.errorhandler(Exception)
def handle_unexpected_error(exc: Exception) -> tuple:
    """Log unexpected errors and return a 500 response."""
    logging.exception("Unhandled error: %s", exc)
    return jsonify({'error': 'internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8000'))
    # Default to localhost; set HOST=0.0.0.0 to expose the service externally.
    host = os.environ.get('HOST', '127.0.0.1')
    if host != '127.0.0.1':
        logging.warning(
            "Using non-local host %s; ensure this exposure is intended", host
        )
    else:
        logging.info("HOST not set, defaulting to %s", host)
    app.run(host=host, port=port)
