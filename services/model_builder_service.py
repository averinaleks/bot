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
_model: LogisticRegression | None = None


def _load_model() -> None:
    global _model
    if _model is None and os.path.exists(MODEL_FILE):
        try:
            _model = joblib.load(MODEL_FILE)
        except Exception:  # pragma: no cover
            _model = None


@app.route('/train', methods=['POST'])
def train():
    data = request.get_json(force=True)
    features = np.array(data.get('features', []))
    labels = np.array(data.get('labels', []))
    model = LogisticRegression(max_iter=100)
    if features.size and labels.size:
        model.fit(features, labels)
    joblib.dump(model, MODEL_FILE)
    global _model
    _model = model
    return jsonify({'status': 'trained'})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data.get('features', []))
    # Accept either a single feature vector or a batch of vectors
    features = np.atleast_2d(features)
    _load_model()
    if _model is None:
        return jsonify({'error': 'model not trained'}), 400
    prob = float(_model.predict_proba(features)[0, 1])
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
