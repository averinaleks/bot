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
    features = np.array(data.get('features', []), dtype=np.float32).reshape(-1, 1)
    labels = np.array(data.get('labels', []), dtype=np.float32)
    if len(features) == 0 or len(features) != len(labels):
        return jsonify({'error': 'invalid training data'}), 400
    model = LogisticRegression()
    model.fit(features, labels)
    joblib.dump(model, MODEL_FILE)
    global _model
    _model = model
    return jsonify({'status': 'trained'})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    feature = float(np.array(data.get('features', [0]), dtype=np.float32)[0])
    _load_model()
    if _model is None:
        signal = 'buy' if feature > 0 else None
        prob = 1.0 if signal else 0.0
    else:
        prob = float(_model.predict_proba([[feature]])[0, 1])
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
