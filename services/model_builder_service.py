"""Reference model builder service using logistic regression.

The service expects binary labels. The ``/train`` route will return a
``400`` error if the provided label array does not contain exactly two
unique classes.
"""

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


def _load_model() -> None:
    """Load model from ``MODEL_FILE`` if it exists."""
    global _model
    if os.path.exists(MODEL_FILE):
        try:
            _model = joblib.load(MODEL_FILE)
        except Exception:  # pragma: no cover - model may be corrupted
            _model = None
            raise


@app.route('/train', methods=['POST'])
def train() -> tuple:
    data = request.get_json(force=True)
    features = np.array(data.get('features', []), dtype=np.float32)
    labels = np.array(data.get('labels', []), dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if len(features) == 0 or len(features) != len(labels):
        return jsonify({'error': 'invalid training data'}), 400
    # Ensure training labels represent a binary classification problem
    if len(np.unique(labels)) != 2:
        return jsonify({'error': 'labels must contain two classes'}), 400
    model = LogisticRegression()
    model.fit(features, labels)
    joblib.dump(model, MODEL_FILE)
    global _model
    _model = model
    return jsonify({'status': 'trained'})


@app.route('/predict', methods=['POST'])
def predict() -> tuple:
    data = request.get_json(force=True)
    features = np.array(data.get('features', []), dtype=np.float32)
    if features.ndim == 0:
        features = np.array([[features]], dtype=np.float32)
    elif features.ndim == 1:
        features = features.reshape(1, -1)
    if _model is None:
        price = float(features[0, 0]) if features.size else 0.0
        signal = 'buy' if price > 0 else None
        prob = 1.0 if signal else 0.0
    else:
        prob = float(_model.predict_proba(features)[0, 1])
        signal = 'buy' if prob >= 0.5 else 'sell'
    return jsonify({'signal': signal, 'prob': prob})


@app.route('/ping')
def ping() -> tuple:
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8001'))
    host = os.environ.get('HOST', '0.0.0.0')
    _load_model()
    app.run(host=host, port=port)
