# Этап сборки
FROM nvidia/cuda:12.5.0-cudnn8-devel-ubuntu22.04 AS builder
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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

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
FROM nvidia/cuda:12.5.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Установка минимальных пакетов для выполнения
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    curl \
    python3.12 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Копируем виртуальное окружение из этапа сборки
COPY --from=builder /app/venv /app/venv

# Копируем исходный код
COPY . .

# Устанавливаем переменные окружения для виртуального окружения
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Проверяем версии библиотек и доступность CUDA с отладкой
RUN echo "Checking library versions and CUDA availability..." && \
    /app/venv/bin/python3.12 -c "import ccxt; print('CCXT Version:', ccxt.__version__)" || echo "CCXT check failed" && \
    /app/venv/bin/python3.12 -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('CUDA Device Count:', torch.cuda.device_count()); print('CUDA Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')" || echo "PyTorch check failed" && \
    /app/venv/bin/python3.12 -c "import ray; print('Ray Version:', ray.__version__)" || echo "Ray check failed" && \
    /app/venv/bin/python3.12 -c "import optuna; print('Optuna Version:', optuna.__version__)" || echo "Optuna check failed" && \
    /app/venv/bin/python3.12 -c "import shap; print('SHAP Version:', shap.__version__)" || echo "SHAP check failed" && \
    /app/venv/bin/python3.12 -c "import numba; print('Numba Version:', numba.__version__)" || echo "Numba check failed"

# Указываем команду для запуска
CMD ["/app/venv/bin/python3.12", "trading_bot.py"]
