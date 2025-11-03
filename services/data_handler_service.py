"""Simple reference data handler service fetching real prices from Bybit."""

import hmac
import logging
import os
import re
import tempfile
from collections import deque
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

from bot.dotenv_utils import load_dotenv
from bot.host_utils import validate_host
from services.logging_utils import sanitize_log_value
from bot.utils_loader import require_utils
from services.exchange_provider import ExchangeProvider

_utils = require_utils("reset_tempdir_cache")
reset_tempdir_cache = _utils.reset_tempdir_cache


_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{1,20}/[A-Z0-9]{1,20}$")
_SYMBOL_SIMPLE_PATTERN = re.compile(r"^[A-Z0-9]{1,20}$")
_SIMPLE_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{1,20}$")
_KNOWN_QUOTE_SUFFIXES: tuple[str, ...] = (
    "USDT",
    "USDC",
    "BTC",
    "ETH",
    "BUSD",
    "DAI",
    "EUR",
    "USD",
    "TRY",
    "GBP",
    "JPY",
)


load_dotenv()

API_KEY_ENV_VAR = "DATA_HANDLER_API_KEY"
ALLOW_ANONYMOUS_ENV_VAR = "DATA_HANDLER_ALLOW_ANONYMOUS"

try:  # optional dependency
    import flask
except Exception:  # pragma: no cover - flask missing entirely
    flask = None  # type: ignore

if flask is None:  # pragma: no cover - flask absent
    raise ImportError("Flask is required for the data handler service")

Flask = flask.Flask  # type: ignore[attr-defined]
jsonify = flask.jsonify  # type: ignore[attr-defined]
request = flask.request  # type: ignore[attr-defined]
current_app = getattr(flask, "current_app", None)

try:  # optional dependency
    from flask.typing import ResponseReturnValue
except Exception:  # pragma: no cover - fallback when flask.typing missing
    ResponseReturnValue = Any  # type: ignore

_ALLOW_OFFLINE = (
    os.getenv("OFFLINE_MODE") == "1"
    or os.getenv("TEST_MODE") == "1"
    or os.getenv("DATA_HANDLER_USE_STUB") == "1"
)
_FORCE_OFFLINE = os.getenv("DATA_HANDLER_USE_STUB") == "1"

try:
    if _FORCE_OFFLINE:
        raise ImportError("Offline stub requested via DATA_HANDLER_USE_STUB=1")
    import ccxt  # type: ignore
except ImportError as exc:  # pragma: no cover - optional in offline mode
    logger = logging.getLogger(__name__)
    if _ALLOW_OFFLINE:
        if _FORCE_OFFLINE:
            logger.info(
                "DATA_HANDLER_USE_STUB=1: DataHandlerService использует OfflineBybit без загрузки ccxt"
            )
        else:
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

_serve_requests_with_auto_allow = False

if hasattr(app, "run"):
    _original_run = app.run

    def _run_with_auto_allow(*args, **kwargs):  # type: ignore[no-redef]
        global _serve_requests_with_auto_allow
        _serve_requests_with_auto_allow = True
        try:
            return _original_run(*args, **kwargs)
        finally:
            _serve_requests_with_auto_allow = False

    app.run = _run_with_auto_allow  # type: ignore[assignment]

_exchange_ctx: ContextVar[Any | None] = ContextVar("data_handler_exchange", default=None)
# ``_recently_closed`` хранит последние закрытые экземпляры биржи. Это предотвращает
# их немедленное уничтожение сборщиком мусора, из-за чего Python мог бы переиспользовать
# идентификаторы объектов (``id``) для только что созданных экземпляров. В тестах
# конкаррентности это выглядело как повторное использование клиента между запросами.
# Мы ограничиваем deque небольшим числом элементов, чтобы избежать утечек памяти,
# сохраняя при этом достаточно буфер для типичных нагрузок тестов и сервера.
_recently_closed: deque[Any] = deque(maxlen=128)
exchange_provider: ExchangeProvider[Any] | None = None


def _require_api_key() -> "ResponseReturnValue | None":
    """Ensure protected endpoints require a shared token when configured."""

    token = os.getenv(API_KEY_ENV_VAR, "").strip()
    allow_anonymous_raw = os.getenv(ALLOW_ANONYMOUS_ENV_VAR, "")
    allow_anonymous = allow_anonymous_raw.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    auto_allow_reason = None
    if os.getenv("TEST_MODE") == "1":
        auto_allow_reason = "TEST_MODE"
    elif os.getenv("OFFLINE_MODE") == "1":
        auto_allow_reason = "OFFLINE_MODE"
    logger = logging.getLogger(__name__)

    if not token:
        if allow_anonymous:
            logger.warning(
                "Ключ доступа к Data Handler не настроен. Анонимный доступ разрешён, потому что %s=%s. "
                "Используйте только для локальной отладки.",
                ALLOW_ANONYMOUS_ENV_VAR,
                sanitize_log_value(allow_anonymous_raw or "1"),
            )
            return None

        if auto_allow_reason is not None:
            is_testing_client = False
            flask_app: Any = app
            if current_app is not None:
                try:
                    flask_app = current_app._get_current_object()
                except Exception:
                    flask_app = app
            if getattr(flask_app, "testing", False):
                is_testing_client = True
            elif request.environ.get("werkzeug.test"):
                is_testing_client = True

            if not is_testing_client and (
                _serve_requests_with_auto_allow or auto_allow_reason == "OFFLINE_MODE"
            ):
                logger.info(
                    "Ключ доступа к Data Handler не настроен. Анонимный доступ разрешён, потому что %s=1.",
                    auto_allow_reason,
                )
                return None

        logger.warning(
            "Запрос к %s отклонён: ключ доступа к Data Handler не настроен. "
            "Настройте переменную окружения DATA_HANDLER_API_KEY или, только для локальной отладки, установите %s=1.",
            sanitize_log_value(request.path),
            ALLOW_ANONYMOUS_ENV_VAR,
        )
        return jsonify({'error': 'unauthorized'}), 401

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


def _iter_cache_dir_ancestors(path: Path) -> Iterator[Path]:
    """Yield *path* and each ancestor up to the filesystem root."""

    current = path
    while True:
        yield current
        parent = current.parent
        if parent == current:
            break
        current = parent


def _normalise_cache_dir(raw_path: str) -> Path | None:
    """Return a hardened cache directory for historical data."""

    if not raw_path or not raw_path.strip():
        logging.getLogger(__name__).warning(
            "Игнорируем CACHE_DIR: путь не должен быть пустым",
        )
        return None

    try:
        candidate = Path(raw_path).expanduser()
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Игнорируем CACHE_DIR %s: некорректный путь (%s)",
            sanitize_log_value(raw_path),
            sanitize_log_value(str(exc)),
        )
        return None

    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logging.getLogger(__name__).warning(
            "Игнорируем CACHE_DIR %s: не удалось создать каталог (%s)",
            sanitize_log_value(str(candidate)),
            sanitize_log_value(str(exc)),
        )
        return None

    try:
        for ancestor in _iter_cache_dir_ancestors(candidate):
            if ancestor.exists() and ancestor.is_symlink():
                logging.getLogger(__name__).warning(
                    "Игнорируем CACHE_DIR %s: путь содержит символьную ссылку %s",
                    sanitize_log_value(str(candidate)),
                    sanitize_log_value(str(ancestor)),
                )
                return None
        if not candidate.is_dir():
            logging.getLogger(__name__).warning(
                "Игнорируем CACHE_DIR %s: путь не является каталогом",
                sanitize_log_value(str(candidate)),
            )
            return None
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        logging.getLogger(__name__).warning(
            "Игнорируем CACHE_DIR %s: не удалось проверить каталог (%s)",
            sanitize_log_value(str(candidate)),
            sanitize_log_value(str(exc)),
        )
        return None

    if not os.access(resolved, os.R_OK | os.W_OK):
        logging.getLogger(__name__).warning(
            "Игнорируем CACHE_DIR %s: каталог недоступен для чтения/записи",
            sanitize_log_value(str(resolved)),
        )
        return None

    return resolved


def _create_history_cache() -> "HistoricalDataCache | None":
    if HistoricalDataCache is None:
        return None

    logger = logging.getLogger(__name__)
    candidates: list[Path] = []
    env_path = os.getenv("CACHE_DIR")
    if env_path:
        safe_env = _normalise_cache_dir(env_path)
        if safe_env is not None:
            candidates.append(safe_env)

    reset_tempdir_cache()
    fallback_raw = os.path.join(tempfile.gettempdir(), "cache")
    fallback = _normalise_cache_dir(fallback_raw)
    if fallback is not None and (not candidates or fallback != candidates[0]):
        candidates.append(fallback)

    for path in candidates:
        try:
            return HistoricalDataCache(str(path))
        except PermissionError:
            logger.warning(
                "Нет прав на запись в каталог кэша %s, пробуем временную директорию",
                sanitize_log_value(str(path)),
            )
        except Exception:
            logger.exception(
                "Не удалось инициализировать кэш исторических данных в %s",
                sanitize_log_value(str(path)),
            )
    return None


history_cache = _create_history_cache()
if os.getenv("TEST_MODE") == "1":
    history_cache = None

MIN_HISTORY_LIMIT = 1
MAX_HISTORY_LIMIT = 1000


def _load_initial_history(exchange: Any) -> None:
    """Fetch and cache initial OHLCV history for configured symbols."""
    if history_cache is None or pd is None:
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


def _allow_compact_symbol_format() -> bool:
    """Return ``True`` when compact symbols like ``BTCUSDT`` are allowed."""

    flask_app: Any = app
    if current_app is not None:
        try:
            flask_app = current_app._get_current_object()
        except Exception:  # pragma: no cover - fall back to global app
            flask_app = app
    return bool(getattr(flask_app, "testing", False))


def _close_exchange_instance(instance: Any) -> None:
    close_method = getattr(instance, "close", None)
    if callable(close_method):
        close_method()


def _create_exchange() -> Any:
    exchange = ccxt.bybit(
        {
            'apiKey': os.getenv('BYBIT_API_KEY', ''),
            'secret': os.getenv('BYBIT_API_SECRET', ''),
        }
    )
    _load_initial_history(exchange)
    return exchange


def _normalise_symbol(symbol: str) -> str | None:
    """Return a normalised ``BASE/QUOTE`` representation for *symbol*.

    The helper accepts the historical ``BTCUSDT`` style identifiers used by
    Bybit and converts them into the ``BTC/USDT`` format expected by ccxt.  A
    small set of common quote suffixes is recognised to keep the conversion
    deterministic and avoid expensive exchange lookups during Dependabot test
    runs.
    """

    raw = symbol.strip().upper()
    if not raw:
        return None

    if ":" in raw:
        main, _, suffix = raw.partition(":")
        normalised_main = _normalise_symbol(main)
        if normalised_main is None:
            return None
        if not _SYMBOL_SIMPLE_PATTERN.fullmatch(suffix):
            return None
        return f"{normalised_main}:{suffix}"

    if "/" in raw:
        return raw if _SYMBOL_PATTERN.fullmatch(raw) else None

    if not _SYMBOL_SIMPLE_PATTERN.fullmatch(raw):
        return None

    if not _allow_compact_symbol_format():
        return None

    for quote in _KNOWN_QUOTE_SUFFIXES:
        if raw.endswith(quote) and len(raw) > len(quote):
            base = raw[: -len(quote)]
            if _SYMBOL_SIMPLE_PATTERN.fullmatch(base):
                candidate = f"{base}/{quote}"
                if _SYMBOL_PATTERN.fullmatch(candidate):
                    return candidate
    return None


def validate_symbol(symbol: str) -> bool:
    """Return ``True`` when *symbol* can be normalised to a valid format."""

    return _normalise_symbol(symbol) is not None


def _allow_legacy_symbol_format() -> bool:
    """Return ``True`` when requests may omit the quote asset in symbols."""

    if getattr(app, "testing", False):  # type: ignore[arg-type]
        return True

    allow_flag = os.getenv("DATA_HANDLER_ALLOW_LEGACY_SYMBOLS", "")
    return allow_flag.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_symbol(symbol: str) -> str | None:
    """Return a sanitized trading symbol or ``None`` when invalid."""

    cleaned = symbol.strip().upper()
    if not cleaned:
        return None

    if validate_symbol(cleaned):
        return cleaned

    if "/" in cleaned or not _SIMPLE_SYMBOL_PATTERN.fullmatch(cleaned):
        return None

    if not _allow_legacy_symbol_format():
        return None

    default_quote = os.getenv("DATA_HANDLER_DEFAULT_QUOTE", "USDT").strip().upper()
    if not default_quote:
        default_quote = "USDT"

    candidate = f"{cleaned}/{default_quote}"
    return candidate if validate_symbol(candidate) else None


exchange_provider = ExchangeProvider(_create_exchange, close=_close_exchange_instance)


def _current_exchange() -> Any | None:
    exchange = _exchange_ctx.get()
    if exchange is not None:
        return exchange
    provider = exchange_provider
    if provider is None:
        return None
    exchange = provider.create()
    _exchange_ctx.set(exchange)
    return exchange


def init_exchange() -> None:
    """Ensure the exchange is initialized before serving requests."""

    provider = exchange_provider
    if provider is None:
        raise RuntimeError("Exchange provider is not configured")
    try:
        exchange = provider.create()
    except Exception as exc:  # pragma: no cover - config errors
        logging.exception("Failed to initialize Bybit client: %s", exc)
        raise RuntimeError("Invalid Bybit configuration") from exc
    else:
        _exchange_ctx.set(exchange)
        close_exchange(None)


if hasattr(app, "before_first_request"):
    app.before_first_request(init_exchange)


def close_exchange(_: BaseException | None = None) -> None:
    """Закрыть соединение с биржей при завершении контекста приложения."""
    provider = exchange_provider
    exchange = _exchange_ctx.get()
    if exchange is None:
        _exchange_ctx.set(None)
        return
    _exchange_ctx.set(None)
    if provider is None:
        _close_exchange_instance(exchange)
        _recently_closed.append(exchange)
        return
    try:
        provider.close_instance(exchange)
    except Exception:  # pragma: no cover - defensive logging
        logging.getLogger(__name__).exception("Failed to close exchange instance")
    finally:
        _recently_closed.append(exchange)

if hasattr(app, "teardown_appcontext"):
    app.teardown_appcontext(close_exchange)

if hasattr(app, "teardown_request"):
    app.teardown_request(close_exchange)

CCXT_BASE_ERROR = getattr(ccxt, 'BaseError', Exception)
CCXT_NETWORK_ERROR = getattr(ccxt, 'NetworkError', CCXT_BASE_ERROR)

# Correct price endpoint without trailing whitespace
@app.route('/price/<path:symbol>', methods=['GET'])
def price(symbol: str) -> ResponseReturnValue:
    auth_error = _require_api_key()
    if auth_error is not None:
        return auth_error
    normalized_symbol = _normalize_symbol(symbol)
    if normalized_symbol is None:
        return jsonify({'error': 'invalid symbol format'}), 400
    exchange = _current_exchange()
    if exchange is None:
        return jsonify({'error': 'exchange not initialized'}), 503
    try:
        ticker = exchange.fetch_ticker(normalized_symbol)
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
            sanitize_log_value(normalized_symbol),
            exc,
        )
        return jsonify({'error': 'network error contacting exchange'}), 503
    except CCXT_BASE_ERROR as exc:
        logging.exception(
            "Exchange error fetching price for %s: %s",
            sanitize_log_value(normalized_symbol),
            exc,
        )
        return jsonify({'error': 'exchange error fetching price'}), 502


@app.route('/history/<path:symbol>', methods=['GET'])
def history(symbol: str) -> ResponseReturnValue:
    """Return OHLCV history for ``symbol``."""
    auth_error = _require_api_key()
    if auth_error is not None:
        return auth_error
    normalized_symbol = _normalize_symbol(symbol)
    if normalized_symbol is None:
        return jsonify({'error': 'invalid symbol format'}), 400
    exchange = _current_exchange()
    if exchange is None:
        return jsonify({'error': 'exchange not initialized'}), 503
    timeframe = request.args.get('timeframe', '1m')
    limit_str = request.args.get('limit')
    warnings_payload: dict[str, Any] = {}
    try:
        limit = int(limit_str) if limit_str is not None else 200
    except ValueError:
        limit = 200
    else:
        if limit <= 0:
            requested_limit = limit
            limit = MIN_HISTORY_LIMIT
            warnings_payload['limit'] = {
                'message': f'limit raised to minimum {MIN_HISTORY_LIMIT}',
                'requested': requested_limit,
                'applied': limit,
            }
        elif limit > MAX_HISTORY_LIMIT:
            requested_limit = limit
            limit = MAX_HISTORY_LIMIT
            warnings_payload['limit'] = {
                'message': f'limit capped at {MAX_HISTORY_LIMIT}',
                'requested': requested_limit,
                'applied': limit,
            }
    try:
        ohlcv = None
        if history_cache is not None and pd is not None:
            try:
                cached = history_cache.load_cached_data(normalized_symbol, timeframe)
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
            ohlcv = exchange.fetch_ohlcv(normalized_symbol, timeframe=timeframe, limit=limit)
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
                    history_cache.save_cached_data(normalized_symbol, timeframe, df)
                except Exception as exc:  # pragma: no cover - logging best effort
                    logging.exception(
                        "Failed to cache history for %s on %s: %s",
                        sanitize_log_value(normalized_symbol),
                        sanitize_log_value(timeframe),
                        exc,
                    )
        response_payload = {'history': ohlcv}
        if warnings_payload:
            response_payload['warnings'] = warnings_payload
        return jsonify(response_payload)
    except CCXT_NETWORK_ERROR as exc:  # pragma: no cover - network errors
        logging.exception(
            "Network error fetching history for %s: %s",
            sanitize_log_value(normalized_symbol),
            exc,
        )
        return jsonify({'error': 'network error contacting exchange'}), 503
    except CCXT_BASE_ERROR as exc:
        logging.exception(
            "Exchange error fetching history for %s: %s",
            sanitize_log_value(normalized_symbol),
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
