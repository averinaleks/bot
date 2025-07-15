"""Reference model builder service using logistic regression."""
from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression

load_dotenv()

app = Flask(__name__)

MODEL_FILE = os.getenv('MODEL_FILE', 'model.pkl')
_model = None


def _load_model():
    global _model
    if _model is None and os.path.exists(MODEL_FILE):
        try:
            _model = joblib.load(MODEL_FILE)
        except Exception:  # pragma: no cover
            _model = None


@app.route('/train', methods=['POST'])
def train():
    data = request.get_json(force=True)

    joblib.dump(model, MODEL_FILE)
    global _model
    _model = model
    return jsonify({'status': 'trained'})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

        signal = 'buy' if prob >= 0.5 else 'sell'
    return jsonify({'signal': signal, 'prob': prob})


@app.route('/ping')
def ping():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8001'))
    host = os.environ.get('HOST', '0.0.0.0')
    _load_model()
    app.run(host=host, port=port)
