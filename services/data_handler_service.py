"""Simple reference data handler service fetching real prices from Bybit."""
from flask import Flask, jsonify, request
from typing import Any
try:  # optional dependency
    from flask.typing import ResponseReturnValue
except Exception:  # pragma: no cover - fallback when flask.typing missing
    ResponseReturnValue = Any  # type: ignore
import logging
import threading
import ccxt
import os
from dotenv import load_dotenv
try:  # optional dependency
    import pandas as pd
except Exception:  # pragma: no cover - pandas not installed
    pd = None  # type: ignore
try:
    from bot.cache import HistoricalDataCache
except Exception:  # pragma: no cover - cache module missing
    HistoricalDataCache = None  # type: ignore
try:  # optional dependency
    from werkzeug.exceptions import HTTPException
except Exception:  # pragma: no cover - fallback when werkzeug absent
    class HTTPException(Exception):  # type: ignore[no-redef]
        pass

load_dotenv()

app = Flask(__name__)
# Минимальный shim Flask, используемый в тестах, может не иметь атрибута
# ``config``. Пропускаем установку лимита, если объекта нет, чтобы импорт
# модуля не падал и тесты, подменяющие ``flask`` на упрощённую реализацию,
# проходили корректно.
if hasattr(app, "config"):
    app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB limit

exchange = None
history_cache = (
    HistoricalDataCache(os.getenv("CACHE_DIR", "/tmp/cache"))
    if HistoricalDataCache
    else None
)
_init_lock = threading.Lock()


def _load_initial_history() -> None:
    """Fetch and cache initial OHLCV history for configured symbols."""
    if exchange is None or history_cache is None or pd is None:
        return
    symbols = [
        s.strip()
        for s in os.getenv("STREAM_SYMBOLS", "").split(",")
        if s.strip()
    ]
    if not symbols:
        return
    timeframe = os.getenv("HISTORY_TIMEFRAME", "1m")
    limit = int(os.getenv("HISTORY_LIMIT", "200"))
    for sym in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ],
            )
            history_cache.save_cached_data(sym, timeframe, df)
        except Exception as exc:  # pragma: no cover - unexpected fetch errors
            logging.exception("Failed to prefetch history for %s: %s", sym, exc)


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
        _load_initial_history()
    except Exception as exc:  # pragma: no cover - config errors
        logging.exception("Failed to initialize Bybit client: %s", exc)
        raise RuntimeError("Invalid Bybit configuration") from exc


if hasattr(app, "before_first_request"):
    app.before_first_request(init_exchange)
else:
    @app.before_request
    def _ensure_exchange() -> None:
        if exchange is None:
            with _init_lock:
                if exchange is None:
                    init_exchange()

def close_exchange(_: BaseException | None = None) -> None:
    """Закрыть соединение с биржей при завершении контекста приложения."""
    global exchange
    if exchange is not None:
        close_method = getattr(exchange, "close", None)
        if callable(close_method):
            close_method()
        exchange = None

if hasattr(app, "teardown_appcontext"):
    app.teardown_appcontext(close_exchange)

CCXT_BASE_ERROR = getattr(ccxt, 'BaseError', Exception)
CCXT_NETWORK_ERROR = getattr(ccxt, 'NetworkError', CCXT_BASE_ERROR)

# Correct price endpoint without trailing whitespace
@app.route('/price/<symbol>', methods=['GET'])
def price(symbol: str) -> ResponseReturnValue:
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


@app.route('/history/<symbol>', methods=['GET'])
def history(symbol: str) -> ResponseReturnValue:
    """Return OHLCV history for ``symbol``."""
    if exchange is None:
        return jsonify({'error': 'exchange not initialized'}), 503
    timeframe = request.args.get('timeframe', '1m')
    limit_str = request.args.get('limit')
    try:
        limit = int(limit_str) if limit_str is not None else 200
    except ValueError:
        limit = 200
    try:
        ohlcv = None
        if history_cache is not None and pd is not None:
            try:
                cached = history_cache.load_cached_data(symbol, timeframe)
            except Exception:
                cached = None
            if cached is not None and hasattr(cached, 'values'):
                ohlcv = (
                    cached[
                        ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    ]
                    .values.tolist()
                )
        if ohlcv is None:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if history_cache is not None and pd is not None:
                try:
                    df = pd.DataFrame(
                        ohlcv,
                        columns=[
                            'timestamp',
                            'open',
                            'high',
                            'low',
                            'close',
                            'volume',
                        ],
                    )
                    history_cache.save_cached_data(symbol, timeframe, df)
                except Exception:
                    pass
        return jsonify({'history': ohlcv})
    except CCXT_NETWORK_ERROR as exc:  # pragma: no cover - network errors
        logging.exception("Network error fetching history for '%s': %s", symbol, exc)
        return jsonify({'error': 'network error contacting exchange'}), 503
    except CCXT_BASE_ERROR as exc:
        logging.exception("Exchange error fetching history for '%s': %s", symbol, exc)
        return jsonify({'error': 'exchange error fetching history'}), 502

@app.route('/ping')
def ping() -> ResponseReturnValue:
    return jsonify({'status': 'ok'})


@app.route('/health')
def health() -> ResponseReturnValue:
    return jsonify({'status': 'ok'})

if hasattr(app, "errorhandler"):
    @app.errorhandler(413)
    def too_large(_) -> ResponseReturnValue:
        return jsonify({'error': 'payload too large'}), 413

    @app.errorhandler(Exception)
    def handle_unexpected_error(exc: Exception) -> ResponseReturnValue:
        """Log unexpected errors and return a 500 response."""
        if isinstance(exc, HTTPException):
            return exc
        logging.exception("Unhandled error: %s", exc)
        return jsonify({'error': 'internal server error'}), 500
else:
    def too_large(_) -> ResponseReturnValue:
        return jsonify({'error': 'payload too large'}), 413

    def handle_unexpected_error(exc: Exception) -> ResponseReturnValue:
        """Log unexpected errors and return a 500 response."""
        if isinstance(exc, HTTPException):
            return exc
        logging.exception("Unhandled error: %s", exc)
        return jsonify({'error': 'internal server error'}), 500


if __name__ == "__main__":
    from bot.utils import configure_logging

    configure_logging()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    app.logger.info('Запуск сервиса DataHandlerService на %s:%s', host, port)
    app.run(host=host, port=port)
