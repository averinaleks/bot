"""Simple reference data handler service fetching real prices from Bybit."""

import hmac
import logging
import os
import tempfile
import threading
from types import SimpleNamespace
from typing import Any

from flask import Flask, jsonify, request

from bot.dotenv_utils import load_dotenv
from bot.host_utils import validate_host
from services.logging_utils import sanitize_log_value

load_dotenv()

try:  # optional dependency
    from flask.typing import ResponseReturnValue
except Exception:  # pragma: no cover - fallback when flask.typing missing
    ResponseReturnValue = Any  # type: ignore

_ALLOW_OFFLINE = (
    os.getenv("OFFLINE_MODE") == "1"
    or os.getenv("TEST_MODE") == "1"
    or os.getenv("DATA_HANDLER_USE_STUB") == "1"
)

try:
    import ccxt  # type: ignore
except ImportError as exc:  # pragma: no cover - optional in offline mode
    logger = logging.getLogger(__name__)
    if _ALLOW_OFFLINE:
        logger.warning(
            "`ccxt` не найден: DataHandlerService использует OfflineBybit. "
            "Для онлайн-запуска установите зависимость `pip install ccxt`."
        )
        from services.offline import OfflineBybit

        ccxt = SimpleNamespace(  # type: ignore[assignment]
            bybit=OfflineBybit,
            BaseError=Exception,
            NetworkError=Exception,
        )
    else:
        logger.critical(
            "Библиотека `ccxt` не установлена. Установите `pip install ccxt` "
            "или активируйте офлайн-режим (OFFLINE_MODE=1)."
        )
        raise ImportError(
            "Не удалось импортировать `ccxt`, необходимый для работы с биржей."
        ) from exc

try:  # optional dependency
    import pandas as pd
except ImportError as exc:  # pragma: no cover - pandas not installed
    logging.getLogger(__name__).warning(
        "Библиотека `pandas` не найдена: %s. Установите `pip install pandas` "
        "или используйте альтернативу на базе стандартных структур данных.",
        exc,
    )
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

app = Flask(__name__)
# Минимальный shim Flask, используемый в тестах, может не иметь атрибута
# ``config``. Пропускаем установку лимита, если объекта нет, чтобы импорт
# модуля не падал и тесты, подменяющие ``flask`` на упрощённую реализацию,
# проходили корректно.
if hasattr(app, "config"):
    app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB limit

exchange = None


def _require_api_key() -> "ResponseReturnValue | None":
    """Ensure protected endpoints require a shared token when configured."""

    token = os.getenv("DATA_HANDLER_API_KEY", "").strip()
    if not token:
        return None

    provided = (request.headers.get("X-API-KEY") or "").strip()
    reason: str | None = None
    if not provided:
        reason = "missing token"
    elif not hmac.compare_digest(provided, token):
        reason = "token mismatch"

    if reason is None:
        return None

    remote = request.headers.get("X-Forwarded-For") or request.remote_addr or "unknown"
    logging.getLogger(__name__).warning(
        "Запрос к %s от %s отклонён: проверка API-ключа не пройдена",
        sanitize_log_value(request.path),
        sanitize_log_value(remote),
    )
    return jsonify({'error': 'unauthorized'}), 401


def _create_history_cache() -> "HistoricalDataCache | None":
    if HistoricalDataCache is None:
        return None

    logger = logging.getLogger(__name__)
    candidates: list[str] = []
    env_path = os.getenv("CACHE_DIR")
    if env_path:
        candidates.append(env_path)
    tmp_path = os.path.join(tempfile.gettempdir(), "cache")
    if not env_path or env_path != tmp_path:
        candidates.append(tmp_path)

    for path in candidates:
        try:
            return HistoricalDataCache(path)
        except PermissionError:
            logger.warning(
                "Нет прав на запись в каталог кэша %s, пробуем временную директорию",
                path,
            )
        except Exception:
            logger.exception(
                "Не удалось инициализировать кэш исторических данных в %s",
                path,
            )
    return None


history_cache = _create_history_cache()
if os.getenv("TEST_MODE") == "1":
    history_cache = None
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
    auth_error = _require_api_key()
    if auth_error is not None:
        return auth_error
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
        logging.exception(
            "Network error fetching price for %s: %s",
            sanitize_log_value(symbol),
            exc,
        )
        return jsonify({'error': 'network error contacting exchange'}), 503
    except CCXT_BASE_ERROR as exc:
        logging.exception(
            "Exchange error fetching price for %s: %s",
            sanitize_log_value(symbol),
            exc,
        )
        return jsonify({'error': 'exchange error fetching price'}), 502


@app.route('/history/<symbol>', methods=['GET'])
def history(symbol: str) -> ResponseReturnValue:
    """Return OHLCV history for ``symbol``."""
    auth_error = _require_api_key()
    if auth_error is not None:
        return auth_error
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
                except Exception as exc:  # pragma: no cover - logging best effort
                    logging.exception(
                        "Failed to cache history for %s on %s: %s",
                        sanitize_log_value(symbol),
                        sanitize_log_value(timeframe),
                        exc,
                    )
        return jsonify({'history': ohlcv})
    except CCXT_NETWORK_ERROR as exc:  # pragma: no cover - network errors
        logging.exception(
            "Network error fetching history for %s: %s",
            sanitize_log_value(symbol),
            exc,
        )
        return jsonify({'error': 'network error contacting exchange'}), 503
    except CCXT_BASE_ERROR as exc:
        logging.exception(
            "Exchange error fetching history for %s: %s",
            sanitize_log_value(symbol),
            exc,
        )
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


def get_bind_host() -> str:
    """Return the validated host value for ``app.run``."""

    return validate_host()


def main() -> None:
    from bot.utils import configure_logging

    configure_logging()
    host = get_bind_host()
    port = int(os.getenv("PORT", "8000"))
    app.logger.info('Запуск сервиса DataHandlerService на %s:%s', host, port)
    app.run(host=host, port=port)


if __name__ == "__main__":
    main()
