"""Minimal trade manager service using Bybit via ccxt.

This reference service exposes endpoints to open and close positions on
Bybit. API keys are taken from ``BYBIT_API_KEY`` and ``BYBIT_API_SECRET``
environment variables.
"""

import json
import logging
import math
import os
import stat
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping, cast

from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest
try:  # optional dependency
    from flask.typing import ResponseReturnValue
except Exception:  # pragma: no cover - fallback when flask.typing missing
    ResponseReturnValue = Any  # type: ignore

from bot.trade_manager import server_common
from bot.utils_loader import require_utils
from services.logging_utils import sanitize_log_value

_utils = require_utils("validate_host", "safe_int")
validate_host = _utils.validate_host
safe_int = _utils.safe_int

server_common.load_environment()

app = Flask(__name__)
if hasattr(app, "config"):
    app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB limit

logger = logging.getLogger(__name__)
POSITIONS_LOCK = threading.RLock()

_exchange_runtime: server_common.ExchangeRuntime | None = None
exchange: Any | None = None


def _current_exchange() -> Any | None:
    global exchange
    runtime = _exchange_runtime
    if runtime is None:
        return exchange
    current = runtime.current()
    if current is not None:
        exchange = current
    return exchange


def init_exchange() -> None:
    """Ensure the exchange is initialized before serving requests."""

    global exchange
    runtime = _exchange_runtime
    if runtime is None:
        raise RuntimeError("Exchange runtime is not configured")
    exchange = runtime.init()


# Expected API token for simple authentication
API_TOKEN = (server_common.get_api_token() or '').strip()


if hasattr(app, "before_first_request"):
    app.before_first_request(init_exchange)


@app.before_request
def _bind_exchange() -> None:
    runtime = _exchange_runtime
    if runtime is not None:
        global exchange
        exchange = runtime.bind()


@app.before_request
def _require_api_token() -> ResponseReturnValue | None:
    """Simple token-based authentication middleware."""

    if request.method != 'POST' and request.path != '/positions':
        return None

    expected = API_TOKEN
    if not expected:
        return None

    headers: Mapping[str, str] = cast(Mapping[str, str], request.headers)
    reason = server_common.validate_token(headers, expected)

    if reason is not None:
        remote = request.headers.get('X-Forwarded-For') or request.remote_addr or 'unknown'
        logger.warning(
            'Rejected TradeManager request to %s from %s: %s',
            sanitize_log_value(request.path),
            sanitize_log_value(remote),
            reason,
        )
        return jsonify({'error': 'unauthorized'}), 401
    return None

POSITIONS: list[dict] = []
POSITIONS_FILE = Path('cache/positions.json')


def _positions_directory_is_safe(path: Path) -> bool:
    directory = path.parent
    if directory.exists():
        if directory.is_symlink():
            logger.warning(
                'Отказ записи позиций: каталог %s является символьной ссылкой',
                sanitize_log_value(str(directory)),
            )
            return False
        if not directory.is_dir():
            logger.warning(
                'Отказ записи позиций: путь %s не является каталогом',
                sanitize_log_value(str(directory)),
            )
            return False
    return True


def _positions_file_is_safe(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return True
    if stat.S_ISLNK(info.st_mode):
        logger.warning(
            'Отказ доступа к кэшу позиций: файл %s является символьной ссылкой',
            sanitize_log_value(str(path)),
        )
        return False
    if not stat.S_ISREG(info.st_mode):
        logger.warning(
            'Отказ доступа к кэшу позиций: путь %s не является обычным файлом',
            sanitize_log_value(str(path)),
        )
        return False
    return True


def _load_positions() -> None:
    """Load positions list from on-disk cache."""
    if not _positions_directory_is_safe(POSITIONS_FILE):
        with POSITIONS_LOCK:
            POSITIONS.clear()
        return
    if not _positions_file_is_safe(POSITIONS_FILE):
        with POSITIONS_LOCK:
            POSITIONS.clear()
        return
    loaded: list[dict] = []
    try:
        with POSITIONS_FILE.open('r', encoding='utf-8') as fh:
            data = json.load(fh)
        if isinstance(data, list):
            loaded = data
    except (OSError, json.JSONDecodeError):
        loaded = []

    with POSITIONS_LOCK:
        POSITIONS[:] = loaded


def _write_positions_locked() -> None:
    """Persist positions list to on-disk cache."""
    directory = POSITIONS_FILE.parent
    if not _positions_directory_is_safe(POSITIONS_FILE):
        return
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - disk errors
        logging.warning(
            'Failed to create positions cache directory %s: %s',
            sanitize_log_value(str(directory)),
            exc,
        )
        return
    if directory.is_symlink() or not directory.is_dir():
        logger.warning(
            'Refusing to persist positions: %s is not a regular directory',
            sanitize_log_value(str(directory)),
        )
        return
    if not _positions_file_is_safe(POSITIONS_FILE):
        return

    try:
        fd, tmp_name = tempfile.mkstemp(
            dir=str(directory), prefix='.positions.', suffix='.tmp'
        )
    except OSError as exc:  # pragma: no cover - tmp creation failures
        logging.warning('Failed to create temporary positions cache: %s', exc)
        return

    tmp_path = Path(tmp_name)
    snapshot = [dict(entry) if isinstance(entry, dict) else entry for entry in POSITIONS]
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            json.dump(snapshot, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, POSITIONS_FILE)
    except OSError as exc:  # pragma: no cover - disk errors
        logging.warning('Failed to save positions cache: %s', exc)
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _save_positions() -> None:
    """Public wrapper ensuring the positions lock is held while saving."""

    with POSITIONS_LOCK:
        _write_positions_locked()



def _sync_positions_with_exchange(exchange: Any | None = None) -> None:
    """Fetch open positions from the exchange and drop closed ones locally."""
    exchange = exchange or _current_exchange()
    if exchange is None or not hasattr(exchange, 'fetch_positions'):
        return

    try:
        remote_positions = exchange.fetch_positions()
    except Exception as exc:  # pragma: no cover - network/API issues
        logging.warning('fetch_positions failed: %s', exc)
        return

    remote_positions = remote_positions or []

    def _extract_from_dict(data: dict, keys: tuple[str, ...]) -> Any:
        for key in keys:
            if not isinstance(data, dict):
                continue
            value = data.get(key)
            if value not in (None, ''):
                return value
        return None

    def _extract_amount(entry: dict) -> float | None:
        amount_keys = ('contracts', 'contractSize', 'size', 'amount', 'positionAmt', 'qty')
        value = _extract_from_dict(entry, amount_keys)
        if value is None and isinstance(entry.get('info'), dict):
            value = _extract_from_dict(entry['info'], amount_keys)
        if value is None:
            return None
        try:
            return abs(float(value))
        except (TypeError, ValueError):
            return None

    def _extract_symbol(entry: dict) -> str | None:
        symbol = _extract_from_dict(entry, ('symbol', 'market'))
        if symbol is None and isinstance(entry.get('info'), dict):
            symbol = _extract_from_dict(entry['info'], ('symbol', 'market'))
        return str(symbol) if symbol is not None else None

    def _extract_side(entry: dict) -> str:
        side_value = _extract_from_dict(entry, ('side', 'direction', 'positionSide'))
        if side_value is None and isinstance(entry.get('info'), dict):
            side_value = _extract_from_dict(entry['info'], ('side', 'direction', 'positionSide'))
        if side_value is None:
            return ''
        side = str(side_value).lower()
        if side == 'long':
            return 'buy'
        if side == 'short':
            return 'sell'
        return side

    def _extract_id(entry: dict) -> str | None:
        identifier = _extract_from_dict(entry, ('id', 'positionId'))
        if identifier is None and isinstance(entry.get('info'), dict):
            identifier = _extract_from_dict(entry['info'], ('id', 'positionId'))
        return str(identifier) if identifier is not None else None

    active_ids: set[str] = set()
    active_pairs: set[tuple[str, str]] = set()

    for position in remote_positions:
        if not isinstance(position, dict):
            continue
        amount = _extract_amount(position)
        symbol = _extract_symbol(position)
        side = _extract_side(position)
        identifier = _extract_id(position)

        if identifier is not None:
            # Treat unknown amounts as active to avoid false removals.
            if amount is None or amount > 0:
                active_ids.add(identifier)
        if symbol and side and (amount is None or amount > 0):
            active_pairs.add((symbol.upper(), side))

    with POSITIONS_LOCK:
        before_count = len(POSITIONS)
        filtered: list[dict] = []
        for record in POSITIONS:
            if record.get('action') != 'open':
                filtered.append(record)
                continue

            record_id = record.get('id')
            if record_id is not None and str(record_id) in active_ids:
                filtered.append(record)
                continue

            record_symbol = record.get('symbol')
            record_side = str(record.get('side', '')).lower()
            if record_side == 'long':
                record_side = 'buy'
            elif record_side == 'short':
                record_side = 'sell'

            if (
                record_symbol
                and record_side
                and (str(record_symbol).upper(), record_side) in active_pairs
            ):
                filtered.append(record)
                continue

            # Position missing on the exchange – drop it locally.
            logger.info(
                'removing closed position %s %s (%s)',
                sanitize_log_value(record_symbol),
                record_side,
                sanitize_log_value(record_id),
            )

        removed = before_count - len(filtered)
        changed = filtered != POSITIONS
        if changed:
            POSITIONS[:] = filtered
        if removed > 0 or changed:
            _write_positions_locked()


_load_positions()


_exchange_runtime = server_common.ExchangeRuntime(
    service_name="TradeManagerService",
    context_name="trade_manager_exchange",
    after_create=_sync_positions_with_exchange,
)
exchange_provider = _exchange_runtime.provider
ccxt = _exchange_runtime.ccxt
CCXT_BASE_ERROR = _exchange_runtime.ccxt_base_error
CCXT_NETWORK_ERROR = _exchange_runtime.ccxt_network_error
CCXT_BAD_REQUEST = _exchange_runtime.ccxt_bad_request


def _record(
    order: dict,
    symbol: str,
    side: str,
    amount: float,
    action: str,
    trailing_stop: float | None = None,
    tp: float | None = None,
    sl: float | None = None,
    entry_price: float | None = None,
) -> None:
    record = {
        'id': order.get('id'),
        'symbol': symbol,
        'side': side,
        'amount': amount,
        'action': action,
        'trailing_stop': trailing_stop,
        'tp': tp,
        'sl': sl,
        'entry_price': entry_price,
    }
    with POSITIONS_LOCK:
        POSITIONS.append(record)
        _write_positions_locked()


@app.route('/open_position', methods=['POST'])
def open_position() -> ResponseReturnValue:
    try:
        data = request.get_json(force=True)
    except BadRequest:
        return jsonify({'error': 'invalid json'}), 400
    symbol = data.get('symbol')
    side = str(data.get('side', 'buy')).lower()
    try:
        price = float(data.get('price', 0) or 0)
    except (TypeError, ValueError):
        return jsonify({'error': 'invalid order'}), 400
    try:
        amount = float(data.get('amount', 0) or 0)
    except (TypeError, ValueError):
        return jsonify({'error': 'invalid order'}), 400
    tp = data.get('tp')
    sl = data.get('sl')
    trailing_stop = data.get('trailing_stop')
    tp = float(tp) if tp is not None else None
    sl = float(sl) if sl is not None else None
    trailing_stop = float(trailing_stop) if trailing_stop is not None else None
    risk_usd = 0.0
    if amount <= 0:
        raw_risk = os.getenv('TRADE_RISK_USD', '0') or 0
        try:
            risk_usd = float(raw_risk)
        except (TypeError, ValueError):
            risk_usd = 0.0
        if risk_usd > 0:
            if not math.isfinite(price) or price <= 0:
                return jsonify({'error': 'invalid order'}), 400
            amount = risk_usd / price
    if not math.isfinite(amount) or amount <= 0:
        return jsonify({'error': 'invalid order'}), 400
    exchange = _current_exchange()
    if exchange is None:
        return jsonify({'error': 'exchange not initialized'}), 503
    if not symbol:
        return jsonify({'error': 'invalid order'}), 400
    try:
        protective_failures: list[dict[str, str]] = []
        mitigation_actions: list[str] = []
        if trailing_stop is not None and hasattr(
            exchange, 'create_order_with_trailing_stop'
        ):
            app.logger.info('using create_order_with_trailing_stop')
            order = exchange.create_order_with_trailing_stop(
                symbol, 'market', side, amount, None, trailing_stop, None
            )
        elif (tp is not None or sl is not None) and hasattr(
            exchange, 'create_order_with_take_profit_and_stop_loss'
        ):
            app.logger.info('using create_order_with_take_profit_and_stop_loss')
            order = exchange.create_order_with_take_profit_and_stop_loss(
                symbol, 'market', side, amount, None, tp, sl, None
            )
        else:
            app.logger.info('using fallback order placement')
            order = exchange.create_order(symbol, 'market', side, amount)
            opp_side = 'sell' if side == 'buy' else 'buy'
            if sl is not None:
                stop_order = None
                delay = 0.1 if os.getenv("TEST_MODE") == "1" else 1.0
                for attempt in range(3):
                    try:
                        stop_order = exchange.create_order(
                            symbol, 'stop', opp_side, amount, sl
                        )
                    except CCXT_BASE_ERROR as exc:
                        app.logger.debug('stop order failed: %s', exc)
                        try:
                            stop_order = exchange.create_order(
                                symbol, 'stop_market', opp_side, amount, sl
                            )
                        except CCXT_BASE_ERROR as exc:
                            app.logger.debug('stop_market order failed: %s', exc)
                            stop_order = None
                    if stop_order and stop_order.get('id') is not None:
                        break
                    if attempt < 2:
                        time.sleep(delay)
                        delay *= 2
                if not stop_order or stop_order.get('id') is None:
                    safe_symbol = sanitize_log_value(symbol)
                    warn_msg = f"не удалось создать stop loss ордер для {safe_symbol}"
                    app.logger.warning(warn_msg)
                    logger.warning(warn_msg)
                    protective_failures.append({'type': 'stop_loss', 'message': warn_msg})
            if tp is not None:
                tp_order = None
                delay = 0.1 if os.getenv("TEST_MODE") == "1" else 1.0
                for attempt in range(3):
                    try:
                        tp_order = exchange.create_order(
                            symbol, 'limit', opp_side, amount, tp
                        )
                    except CCXT_BASE_ERROR as exc:
                        app.logger.debug('take profit order failed: %s', exc)
                        tp_order = None
                    if tp_order and tp_order.get('id') is not None:
                        break
                    if attempt < 2:
                        time.sleep(delay)
                        delay *= 2
                if not tp_order or tp_order.get('id') is None:
                    safe_symbol = sanitize_log_value(symbol)
                    warn_msg = f"не удалось создать take profit ордер для {safe_symbol}"
                    app.logger.warning(warn_msg)
                    logger.warning(warn_msg)
                    protective_failures.append({'type': 'take_profit', 'message': warn_msg})
            if trailing_stop is not None and price > 0:
                tprice = price - trailing_stop if side == 'buy' else price + trailing_stop
                stop_order = None
                try:
                    stop_order = exchange.create_order(
                        symbol, 'stop', opp_side, amount, tprice
                    )
                except CCXT_BASE_ERROR as exc:
                    app.logger.debug('trailing stop order failed: %s', exc)
                    try:
                        stop_order = exchange.create_order(
                            symbol, 'stop_market', opp_side, amount, tprice
                        )
                    except CCXT_BASE_ERROR as exc:
                        app.logger.debug('trailing stop_market failed: %s', exc)
                        stop_order = None
                if not stop_order or stop_order.get('id') is None:
                    safe_symbol = sanitize_log_value(symbol)
                    warn_msg = f"не удалось создать trailing stop ордер для {safe_symbol}"
                    app.logger.warning(warn_msg)
                    logger.warning(warn_msg)
                    protective_failures.append({'type': 'trailing_stop', 'message': warn_msg})
        if not order or order.get('id') is None:
            app.logger.error('failed to create primary order')
            return jsonify({'error': 'order creation failed'}), 500
        _record(
            order,
            symbol,
            side,
            amount,
            'open',
            trailing_stop,
            tp,
            sl,
            price if math.isfinite(price) and price > 0 else None,
        )
        response: dict[str, Any] = {'status': 'ok', 'order_id': order.get('id')}
        if protective_failures:
            safe_symbol = sanitize_log_value(symbol)
            app.logger.error(
                'protective order failures for %s: %s', safe_symbol, protective_failures
            )
            cancel_success = False
            if hasattr(exchange, 'cancel_order'):
                try:
                    exchange.cancel_order(order.get('id'), symbol)
                    cancel_success = True
                    mitigation_actions.append('primary_order_cancelled')
                    app.logger.info(
                        'primary order %s cancelled after protective failure',
                        order.get('id'),
                    )
                except CCXT_BASE_ERROR as exc:
                    app.logger.warning(
                        'failed to cancel primary order %s for %s: %s',
                        order.get('id'),
                        safe_symbol,
                        exc,
                    )
            if not cancel_success:
                emergency_close_success = False
                try:
                    close_order = exchange.create_order(
                        symbol,
                        'market',
                        'sell' if side == 'buy' else 'buy',
                        amount,
                        params={'reduce_only': True},
                    )
                    if close_order and close_order.get('id') is not None:
                        emergency_close_success = True
                        mitigation_actions.append('emergency_close_submitted')
                        app.logger.info(
                            'emergency close order submitted for %s after protective failure',
                            safe_symbol,
                        )
                except CCXT_BASE_ERROR as exc:
                    app.logger.exception(
                        'failed to place emergency close order for %s: %s',
                        safe_symbol,
                        exc,
                    )
                if not emergency_close_success:
                    mitigation_actions.append('emergency_close_failed')
        if protective_failures:
            warnings_payload: dict[str, Any] = {
                'protective_orders_failed': protective_failures,
            }
            if mitigation_actions:
                warnings_payload['mitigations'] = mitigation_actions
            response['warnings'] = warnings_payload
        return jsonify(response)
    except CCXT_NETWORK_ERROR as exc:  # pragma: no cover - network errors
        app.logger.exception('network error creating order: %s', exc)
        return jsonify({'error': 'network error contacting exchange'}), 503
    except CCXT_BAD_REQUEST as exc:
        app.logger.warning('bad request when creating order: %s', exc)
        return jsonify({'error': 'invalid order parameters'}), 400
    except CCXT_BASE_ERROR as exc:
        app.logger.exception('exchange error creating order: %s', exc)
        return jsonify({'error': 'exchange error creating order'}), 502


@app.route('/close_position', methods=['POST'])
def close_position() -> ResponseReturnValue:
    try:
        data = request.get_json(force=True)
    except BadRequest:
        return jsonify({'error': 'invalid json'}), 400
    order_id = data.get('order_id')
    side = str(data.get('side', '')).lower()
    close_amount = data.get('close_amount')
    if close_amount is not None:
        try:
            close_amount = float(close_amount)
        except (TypeError, ValueError):
            return jsonify({'error': 'invalid order'}), 400
        if not math.isfinite(close_amount) or close_amount <= 0:
            return jsonify({'error': 'invalid order'}), 400
    exchange = _current_exchange()
    if exchange is None:
        return jsonify({'error': 'exchange not initialized'}), 503
    if not order_id or not side:
        return jsonify({'error': 'invalid order'}), 400

    with POSITIONS_LOCK:
        rec = next(
            (
                dict(rec)
                for rec in POSITIONS
                if rec.get('id') == order_id and rec.get('action') == 'open'
            ),
            None,
        )
    if rec is None:
        return jsonify({'error': 'order not found'}), 404

    symbol = rec.get('symbol')
    amount = close_amount if close_amount is not None else float(rec.get('amount', 0) or 0)
    if not math.isfinite(amount) or amount <= 0:
        return jsonify({'error': 'invalid order'}), 400

    params = {'reduce_only': True}
    try:
        order = exchange.create_order(symbol, 'market', side, amount, params=params)
        if not order or order.get('id') is None:
            app.logger.error('failed to create close order')
            return jsonify({'error': 'order creation failed'}), 500
        with POSITIONS_LOCK:
            rec_index = next(
                (
                    i
                    for i, current in enumerate(POSITIONS)
                    if current.get('id') == order_id and current.get('action') == 'open'
                ),
                None,
            )
            if rec_index is not None:
                current = POSITIONS[rec_index]
                remaining = float(current.get('amount', 0) or 0) - amount
                if remaining <= 0:
                    POSITIONS.pop(rec_index)
                else:
                    current['amount'] = remaining
                _write_positions_locked()
        return jsonify({'status': 'ok', 'order_id': order.get('id')})
    except CCXT_NETWORK_ERROR as exc:  # pragma: no cover - network errors
        app.logger.exception('network error closing position: %s', exc)
        return jsonify({'error': 'network error contacting exchange'}), 503
    except CCXT_BAD_REQUEST as exc:
        app.logger.warning('bad request when closing position: %s', exc)
        return jsonify({'error': 'invalid order parameters'}), 400
    except CCXT_BASE_ERROR as exc:
        app.logger.exception('exchange error closing position: %s', exc)
        return jsonify({'error': 'exchange error closing position'}), 502


@app.route('/positions')
def positions() -> ResponseReturnValue:
    _sync_positions_with_exchange()
    with POSITIONS_LOCK:
        snapshot = [dict(entry) if isinstance(entry, dict) else entry for entry in POSITIONS]
    return jsonify({'positions': snapshot})


@app.route('/ping')
def ping() -> ResponseReturnValue:
    return jsonify({'status': 'ok'})


@app.route('/ready')
def ready() -> ResponseReturnValue:
    """Health check endpoint used by docker-compose."""
    return jsonify({'status': 'ok'})

if hasattr(app, "errorhandler"):
    @app.errorhandler(413)
    def too_large(_) -> ResponseReturnValue:
        return jsonify({'error': 'payload too large'}), 413

    @app.errorhandler(Exception)
    def handle_unexpected_error(exc: Exception) -> ResponseReturnValue:
        """Log unexpected errors and return a 500 response."""
        app.logger.exception('unhandled error: %s', exc)
        return jsonify({'error': 'internal server error'}), 500
else:
    def too_large(_) -> ResponseReturnValue:
        return jsonify({'error': 'payload too large'}), 413

    def handle_unexpected_error(exc: Exception) -> ResponseReturnValue:
        """Log unexpected errors and return a 500 response."""
        app.logger.exception('unhandled error: %s', exc)
        return jsonify({'error': 'internal server error'}), 500


if __name__ == '__main__':
    from utils import configure_logging

    configure_logging()
    host = validate_host()
    port = safe_int(os.getenv("PORT", "8002"))
    init_exchange()
    app.logger.info('Запуск сервиса TradeManager на %s:%s', host, port)
    app.run(host=host, port=port)  # host validated above
