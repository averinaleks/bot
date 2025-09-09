"""Simple Flask API for price queries."""
from flask import Flask, jsonify
from .storage import price_storage, DEFAULT_PRICE

api_app = Flask(__name__)


@api_app.route('/price/<symbol>', methods=['GET'])
def price(symbol: str):
    price = price_storage.get(symbol)
    if price is None:
        price = DEFAULT_PRICE
    return jsonify({'price': price})
