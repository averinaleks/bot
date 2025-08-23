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


# Expected API token for simple authentication
API_TOKEN = os.getenv('TRADE_MANAGER_TOKEN')

_exchange_initialized = False


@app.before_request
def _ensure_exchange() -> None:
    """Initialize the exchange on the first request."""
    global _exchange_initialized
    if not _exchange_initialized:
        init_exchange()
        _exchange_initialized = True


@app.before_request
def _require_api_token():
    """Simple token-based authentication middleware."""
    if request.method == 'POST' or request.path == '/positions':
        token = request.headers.get('Authorization', '')
        if token.lower().startswith('bearer '):
            token = token[7:]
        else:
            token = request.headers.get('X-API-KEY', token)
        if not token or token != API_TOKEN:
            return jsonify({'error': 'unauthorized'}), 401

# Gracefully handle missing ccxt error classes when running under test stubs
CCXT_BASE_ERROR = getattr(ccxt, 'BaseError', Exception)
CCXT_NETWORK_ERROR = getattr(ccxt, 'NetworkError', CCXT_BASE_ERROR)
CCXT_BAD_REQUEST = getattr(ccxt, 'BadRequest', CCXT_BASE_ERROR)

POSITIONS: list[dict] = []


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


@app.route('/open_position', methods=['POST'])
def open_position() -> tuple:
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
                orders.append(stop_order)
            if tp is not None:
                try:
                    tp_order = exchange.create_order(
                        symbol, 'limit', opp_side, amount, tp
                    )
                except CCXT_BASE_ERROR as exc:
                    app.logger.debug('take profit order failed: %s', exc)
                    tp_order = None
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
def close_position() -> tuple:
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
def positions():
    return jsonify({'positions': POSITIONS})


@app.route('/ping')
def ping():
    return jsonify({'status': 'ok'})


@app.route('/ready')
def ready():
    """Health check endpoint used by docker-compose."""
    return jsonify({'status': 'ok'})


@app.errorhandler(413)
def too_large(_):
    return jsonify({'error': 'payload too large'}), 413

@app.errorhandler(Exception)
def handle_unexpected_error(exc: Exception) -> tuple:
    """Log unexpected errors and return a 500 response."""
    app.logger.exception('unhandled error: %s', exc)
    return jsonify({'error': 'internal server error'}), 500


if __name__ == '__main__':
    from bot.utils import configure_logging

    configure_logging()
    port = int(os.environ.get('PORT', '8002'))
    # По умолчанию слушаем только локальный интерфейс.
    host = os.environ.get('HOST', '127.0.0.1')
    # Prevent binding to all interfaces.
    if host.strip() == '0.0.0.0':  # nosec B104
        raise ValueError('HOST=0.0.0.0 запрещён из соображений безопасности')
    if host != '127.0.0.1':
        app.logger.warning(
            'Используется не локальный хост %s; убедитесь, что это намеренно',
            host,
        )
    else:
        app.logger.info('HOST не установлен, используется %s', host)
    init_exchange()
    app.logger.info('Запуск сервиса TradeManager на %s:%s', host, port)
    app.run(host=host, port=port)  # nosec B104  # host validated above
