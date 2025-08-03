# Этап сборки
FROM nvidia/cuda:12.1.0-cudnn-devel-ubuntu22.04 AS builder
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Установка необходимых пакетов для сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    build-essential \
    libssl-dev \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && python3.12 --version

WORKDIR /app

# Копируем requirements.txt
COPY requirements.txt .

# Создаем виртуальное окружение
ENV VIRTUAL_ENV=/app/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Устанавливаем зависимости
RUN pip install --no-cache-dir pip==24.0 setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    find /app/venv -type d -name '__pycache__' -exec rm -rf {} + && \
    find /app/venv -type f -name '*.pyc' -delete

# Этап выполнения
FROM nvidia/cuda:12.1.0-cudnn-devel-ubuntu22.04
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /app

# Установка минимальных пакетов для выполнения
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    curl \
    python3.12 \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && python3.12 --version

# Копируем виртуальное окружение из этапа сборки
COPY --from=builder /app/venv /app/venv

# Копируем исходный код в /app/bot
COPY . /app/bot

# Устанавливаем переменные окружения для виртуального окружения
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Добавляем PYTHONPATH, чтобы модуль bot был доступен
ENV PYTHONPATH=/app

# Проверяем версии библиотек и доступность CUDA с отладкой
RUN echo "Checking library versions and CUDA availability..." && \
    /app/venv/bin/python3.12 -c "import ccxt; print('CCXT Version:', ccxt.__version__)" || echo "CCXT check failed" && \
    /app/venv/bin/python3.12 -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('CUDA Device Count:', torch.cuda.device_count()); print('CUDA Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')" || echo "PyTorch check failed" && \
    /app/venv/bin/python3.12 -c "import ray; print('Ray Version:', ray.__version__)" || echo "Ray check failed" && \
    /app/venv/bin/python3.12 -c "import optuna; print('Optuna Version:', optuna.__version__)" || echo "Optuna check failed" && \
    /app/venv/bin/python3.12 -c "import shap; print('SHAP Version:', shap.__version__)" || echo "SHAP check failed" && \
    /app/venv/bin/python3.12 -c "import numba; print('Numba Version:', numba.__version__)" || echo "Numba check failed" && \
    /app/venv/bin/python3.12 -c "import tensorflow as tf; print('TF Version:', tf.__version__)" || echo "TensorFlow check failed" && \
    /app/venv/bin/python3.12 -c "import stable_baselines3 as sb3; print('SB3 Version:', sb3.__version__)" || echo "SB3 check failed" && \
    /app/venv/bin/python3.12 -c "import pytorch_lightning as pl; print('Lightning Version:', pl.__version__)" || echo "Lightning check failed" && \
    /app/venv/bin/python3.12 -c "import mlflow; print('MLflow Version:', mlflow.__version__)" || echo "MLflow check failed"

# Указываем команду для запуска
CMD ["/app/venv/bin/python3.12", "-m", "bot.trading_bot"]
