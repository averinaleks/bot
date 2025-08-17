"""Simple reference data handler service fetching real prices from Bybit."""
from flask import Flask, jsonify
import logging
import ccxt
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB limit

exchange = None


def init_exchange() -> None:
    """Initialize the global ccxt Bybit exchange instance."""
    global exchange
    try:
        exchange = ccxt.bybit(
            {
                'apiKey': os.getenv('BYBIT_API_KEY', ''),
                'secret': os.getenv('BYBIT_API_SECRET', ''),
            }
        )
    except Exception as exc:  # pragma: no cover - config errors
        logging.exception("Failed to initialize Bybit client: %s", exc)
        raise RuntimeError("Invalid Bybit configuration") from exc


@app.before_first_request
def _startup() -> None:
    if exchange is None:
        init_exchange()

CCXT_BASE_ERROR = getattr(ccxt, 'BaseError', Exception)
CCXT_NETWORK_ERROR = getattr(ccxt, 'NetworkError', CCXT_BASE_ERROR)

# Correct price endpoint without trailing whitespace
@app.route('/price/<symbol>')
def price(symbol: str):
    if exchange is None:
        return jsonify({'error': 'exchange not initialized'}), 503
    try:
        ticker = exchange.fetch_ticker(symbol)
        last_raw = ticker.get('last')
        try:
            last = float(last_raw)
        except (TypeError, ValueError):
            last = None
        if not last or last <= 0:
            return jsonify({'error': 'invalid price'}), 502
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


@app.errorhandler(413)
def too_large(_):
    return jsonify({'error': 'payload too large'}), 413

@app.errorhandler(Exception)
def handle_unexpected_error(exc: Exception) -> tuple:
    """Log unexpected errors and return a 500 response."""
    logging.exception("Unhandled error: %s", exc)
    return jsonify({'error': 'internal server error'}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get('PORT', '8000'))
    # По умолчанию слушаем только локальный интерфейс.
    host = os.environ.get('HOST', '127.0.0.1')
    # Prevent binding to all interfaces.
    if host.strip() == '0.0.0.0':  # nosec B104
        raise ValueError('HOST=0.0.0.0 запрещён из соображений безопасности')
    if host != '127.0.0.1':
        logging.warning(
            'Используется не локальный хост %s; убедитесь, что это намеренно',
            host,
        )
    else:
        logging.info('HOST не установлен, используется %s', host)
    init_exchange()
    logging.info('Запуск сервиса DataHandler на %s:%s', host, port)
    app.run(host=host, port=port)  # nosec B104  # хост проверен выше
