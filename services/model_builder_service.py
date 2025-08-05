"""Reference model builder service using logistic regression.

The service requires training labels to contain at least two unique
classes.  The ``/train`` route returns a ``400`` error when all labels
belong to a single class.
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
        except Exception as exc:  # pragma: no cover - model may be corrupted
            app.logger.exception("Failed to load model: %s", exc)
            _model = None


@app.route('/train', methods=['POST'])
def train() -> tuple:
    data = request.get_json(force=True)
    # ``features`` may contain multiple attributes such as price, volume and
    # technical indicators.  Ensure the array is always two-dimensional so the
    # logistic regression treats each row as one observation with ``n``
    # features.
    features = np.array(data.get('features', []), dtype=np.float32)
    labels = np.array(data.get('labels', []), dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    else:
        features = features.reshape(len(features), -1)
    if features.size == 0 or len(features) != len(labels):
        return jsonify({'error': 'invalid training data'}), 400
    # Ensure training labels contain at least two classes
    if len(np.unique(labels)) < 2:
        return jsonify({'error': 'labels must contain at least two classes'}), 400
    model = LogisticRegression(multi_class="auto")
    model.fit(features, labels)
    joblib.dump(model, MODEL_FILE)
    global _model
    _model = model
    return jsonify({'status': 'trained'})


@app.route('/predict', methods=['POST'])
def predict() -> tuple:
    data = request.get_json(force=True)
    features = data.get('features')
    if features is None:
        # Backwards compatibility – allow a single ``price`` value.
        price_val = float(data.get('price', 0.0))
        features = [price_val]
    features = np.array(features, dtype=np.float32)
    if features.ndim == 0:
        features = np.array([[features]], dtype=np.float32)
    elif features.ndim == 1:
        features = features.reshape(1, -1)
    else:
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
    # Default to localhost; set HOST=0.0.0.0 to expose the service externally.
    host = os.environ.get('HOST', '127.0.0.1')
    if host != '127.0.0.1':
        app.logger.warning(
            'Using non-local host %s; ensure this exposure is intended', host
        )
    else:
        app.logger.info('HOST not set, defaulting to %s', host)
    app.logger.info('Starting model builder reference service on %s:%s', host, port)
    _load_model()
    app.run(host=host, port=port)
