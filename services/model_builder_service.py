"""Reference model builder service using logistic regression.

The service requires training labels to contain at least two unique
classes.  The ``/train`` route returns a ``400`` error when all labels
belong to a single class.
"""

from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
from pathlib import Path
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB limit

BASE_DIR = Path.cwd().resolve()
MODEL_FILE = Path(os.getenv('MODEL_FILE', 'model.pkl'))
_model = None


def _validate_model_path(path: Path) -> Path:
    """Ensure ``path`` stays inside ``BASE_DIR`` and resolve it."""
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(BASE_DIR):
        app.logger.error(
            "Попытка доступа к файлу вне допустимого каталога: %s", resolved
        )
        raise ValueError("некорректный путь к модели")
    return resolved


def _load_model() -> None:
    """Load model from ``MODEL_FILE`` if it exists."""
    global _model
    try:
        model_path = _validate_model_path(MODEL_FILE)
    except ValueError:
        _model = None
        return
    if model_path.exists():
        try:
            _model = joblib.load(model_path)
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
    try:
        model_path = _validate_model_path(MODEL_FILE)
    except ValueError:
        return jsonify({'error': 'invalid model path'}), 400
    if not model_path.parent.exists():
        app.logger.error(
            "Каталог для файла модели не существует: %s", model_path.parent
        )
        return jsonify({'error': 'invalid model path'}), 400
    joblib.dump(model, model_path)
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


@app.errorhandler(413)
def too_large(_):
    return jsonify({'error': 'payload too large'}), 413

if __name__ == '__main__':
    from bot.utils import configure_logging

    configure_logging()
    port = int(os.environ.get('PORT', '8001'))
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
    app.logger.info('Запуск сервиса ModelBuilder на %s:%s', host, port)
    _load_model()
    app.run(host=host, port=port)  # nosec B104  # host validated above
