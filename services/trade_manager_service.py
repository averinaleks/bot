"""Minimal trade manager service using Bybit via ccxt.

This reference service exposes endpoints to open and close positions on
Bybit. API keys are taken from ``BYBIT_API_KEY`` and ``BYBIT_API_SECRET``
environment variables.
"""

from flask import Flask, request, jsonify
from typing import Any
from pathlib import Path
import json
import logging
import os
import threading
import time
try:  # optional dependency
    from flask.typing import ResponseReturnValue
except Exception:  # pragma: no cover - fallback when flask.typing missing
    ResponseReturnValue = Any  # type: ignore

try:
    import ccxt
except ImportError as exc:  # pragma: no cover - critical dependency missing
    logging.getLogger(__name__).critical(
        "Библиотека `ccxt` обязательна для TradeManager. Установите её через "
        "`pip install ccxt` или подключите локальный mock-объект биржи."
    )
    raise ImportError(
        "TradeManager не может работать без зависимости `ccxt`."
    ) from exc

from bot.dotenv_utils import load_dotenv
from bot.utils import validate_host, safe_int
from services.logging_utils import sanitize_log_value

load_dotenv()
app = Flask(__name__)
if hasattr(app, "config"):
    app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB limit

logger = logging.getLogger(__name__)
exchange = None
_init_lock = threading.Lock()


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
    _sync_positions_with_exchange()


# Expected API token for simple authentication
API_TOKEN = os.getenv('TRADE_MANAGER_TOKEN')


if hasattr(app, "before_first_request"):
    app.before_first_request(init_exchange)
else:
    @app.before_request
    def _ensure_exchange() -> None:
        if exchange is None:
            with _init_lock:
                if exchange is None:
                    init_exchange()


@app.before_request
def _require_api_token() -> ResponseReturnValue | None:
    """Simple token-based authentication middleware."""
    if request.method == 'POST' or request.path == '/positions':
        token = request.headers.get('Authorization', '')
        if token.lower().startswith('bearer '):
            token = token[7:]
        else:
            token = request.headers.get('X-API-KEY', token)
        if not token or token != API_TOKEN:
            return jsonify({'error': 'unauthorized'}), 401
    return None

# Gracefully handle missing ccxt error classes when running under test stubs
CCXT_BASE_ERROR = getattr(ccxt, 'BaseError', Exception)
CCXT_NETWORK_ERROR = getattr(ccxt, 'NetworkError', CCXT_BASE_ERROR)
CCXT_BAD_REQUEST = getattr(ccxt, 'BadRequest', CCXT_BASE_ERROR)

POSITIONS: list[dict] = []
POSITIONS_FILE = Path('cache/positions.json')


def _load_positions() -> None:
    """Load positions list from on-disk cache."""
    global POSITIONS
    try:
        with POSITIONS_FILE.open('r', encoding='utf-8') as fh:
            POSITIONS = json.load(fh)
    except (OSError, json.JSONDecodeError):
        POSITIONS = []


def _save_positions() -> None:
    """Persist positions list to on-disk cache."""
    try:
        POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with POSITIONS_FILE.open('w', encoding='utf-8') as fh:
            json.dump(POSITIONS, fh)
    except OSError as exc:  # pragma: no cover - disk errors
        logging.warning('Failed to save positions cache: %s', exc)


def _sync_positions_with_exchange() -> None:
    """Fetch positions from exchange for verification."""
    if exchange is None or not hasattr(exchange, 'fetch_positions'):
        return
    try:
        exchange.fetch_positions()
    except Exception as exc:  # pragma: no cover - network/API issues
        logging.warning('fetch_positions failed: %s', exc)


_load_positions()


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
    POSITIONS.append(
        {
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
    )
    _save_positions()


@app.route('/open_position', methods=['POST'])
def open_position() -> ResponseReturnValue:
    data = request.get_json(force=True)
    symbol = data.get('symbol')
    side = str(data.get('side', 'buy')).lower()
    price = float(data.get('price', 0) or 0)
    amount = float(data.get('amount', 0))
    tp = data.get('tp')
    sl = data.get('sl')
    trailing_stop = data.get('trailing_stop')
    tp = float(tp) if tp is not None else None
    sl = float(sl) if sl is not None else None
    trailing_stop = float(trailing_stop) if trailing_stop is not None else None
    if amount <= 0:
        risk_usd = float(os.getenv('TRADE_RISK_USD', '0') or 0)
        if risk_usd > 0 and price > 0:
            amount = risk_usd / price
    if exchange is None:
        return jsonify({'error': 'exchange not initialized'}), 503
    if not symbol or amount <= 0:
        return jsonify({'error': 'invalid order'}), 400
    try:
        if trailing_stop is not None and hasattr(
            exchange, 'create_order_with_trailing_stop'
        ):
            app.logger.info('using create_order_with_trailing_stop')
            order = exchange.create_order_with_trailing_stop(
                symbol, 'market', side, amount, None, trailing_stop, None
            )
            orders = [order]
        elif (tp is not None or sl is not None) and hasattr(
            exchange, 'create_order_with_take_profit_and_stop_loss'
        ):
            app.logger.info('using create_order_with_take_profit_and_stop_loss')
            order = exchange.create_order_with_take_profit_and_stop_loss(
                symbol, 'market', side, amount, None, tp, sl, None
            )
            orders = [order]
        else:
            app.logger.info('using fallback order placement')
            orders = []
            order = exchange.create_order(symbol, 'market', side, amount)
            orders.append(order)
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
                orders.append(stop_order)
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
                orders.append(tp_order)
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
                orders.append(stop_order)
        if any(not o or o.get('id') is None for o in orders):
            app.logger.error('failed to create one or more orders')
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
            price if price > 0 else None,
        )
        return jsonify({'status': 'ok', 'order_id': order.get('id')})
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
    data = request.get_json(force=True)
    order_id = data.get('order_id')
    side = str(data.get('side', '')).lower()
    close_amount = data.get('close_amount')
    if close_amount is not None:
        close_amount = float(close_amount)
    if exchange is None:
        return jsonify({'error': 'exchange not initialized'}), 503
    if not order_id or not side:
        return jsonify({'error': 'invalid order'}), 400

    rec_index = next(
        (
            i
            for i, rec in enumerate(POSITIONS)
            if rec.get('id') == order_id and rec.get('action') == 'open'
        ),
        None,
    )
    if rec_index is None:
        return jsonify({'error': 'order not found'}), 404

    rec = POSITIONS[rec_index]
    symbol = rec.get('symbol')
    amount = close_amount if close_amount is not None else rec.get('amount', 0)
    if amount <= 0:
        return jsonify({'error': 'invalid order'}), 400

    params = {'reduce_only': True}
    try:
        order = exchange.create_order(symbol, 'market', side, amount, params=params)
        if not order or order.get('id') is None:
            app.logger.error('failed to create close order')
            return jsonify({'error': 'order creation failed'}), 500
        remaining = rec.get('amount', 0) - amount
        if remaining <= 0:
            POSITIONS.pop(rec_index)
        else:
            rec['amount'] = remaining
        _save_positions()
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
    return jsonify({'positions': POSITIONS})


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
