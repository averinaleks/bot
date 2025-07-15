"""Reference model builder service using logistic regression.

The service now supports multiple input features and keeps separate models for
each trading symbol.  POST JSON data to ``/train`` with ``symbol``, ``features``
and ``labels``.  Predictions are returned from ``/predict`` for the provided
feature vectors.
"""
from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression

load_dotenv()

app = Flask(__name__)

MODEL_DIR = os.getenv('MODEL_DIR', '.')
os.makedirs(MODEL_DIR, exist_ok=True)

_models = {}


def _model_path(symbol: str) -> str:
    return os.path.join(MODEL_DIR, f'{symbol}_model.pkl')


def _load_model(symbol: str):
    model = _models.get(symbol)
    if model is None:
        path = _model_path(symbol)
        if os.path.exists(path):
            try:
                model = joblib.load(path)
            except Exception:  # pragma: no cover
                model = None
        _models[symbol] = model
    return model


@app.route('/train', methods=['POST'])
def train():
    data = request.get_json(force=True)
    symbol = data.get('symbol', 'default')
    feats = np.array(data.get('features', []), dtype=np.float32)
    labels = np.array(data.get('labels', []), dtype=np.float32)
    if feats.ndim != 2 or len(feats) == 0 or len(feats) != len(labels):
        return jsonify({'error': 'invalid training data'}), 400
    model = LogisticRegression()
    model.fit(feats, labels)
    joblib.dump(model, _model_path(symbol))
    _models[symbol] = model
    return jsonify({'status': 'trained'})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    symbol = data.get('symbol', 'default')
    feats = np.array(data.get('features', []), dtype=np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)
    model = _load_model(symbol)
    if model is None:
        val = feats[0, 0] if feats.size else 0.0
        prob = float(val > 0)
        signal = 'buy' if prob >= 0.5 else 'sell'
    else:
        probs = model.predict_proba(feats)[:, 1]
        prob = probs.tolist() if len(probs) > 1 else float(probs[0])
        if isinstance(prob, list):
            signal = ['buy' if p >= 0.5 else 'sell' for p in prob]
        else:
            signal = 'buy' if prob >= 0.5 else 'sell'
    return jsonify({'signal': signal, 'prob': prob})


@app.route('/ping')
def ping():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8001'))
    host = os.environ.get('HOST', '0.0.0.0')
    app.run(host=host, port=port)
