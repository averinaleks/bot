# Этап сборки
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04 AS builder
ARG ZLIB_VERSION=1.3.1
ARG TAR_VERSION=1.36
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Установка необходимых пакетов для сборки и обновление критических библиотек
# Обновление linux-libc-dev устраняет CVE-2024-50217 и CVE-2025-21976, а libgcrypt20 — CVE-2024-2236
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    linux-libc-dev \
    libgcrypt20 \
    build-essential \
    curl \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    libssl-dev \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    && curl --netrc-file /dev/null -L https://zlib.net/zlib-${ZLIB_VERSION}.tar.gz -o zlib.tar.gz \
    && tar -xf zlib.tar.gz \
    && cd zlib-${ZLIB_VERSION} && ./configure --prefix=/usr && make -j"$(nproc)" && make install && cd .. \
    && rm -rf zlib.tar.gz zlib-${ZLIB_VERSION} \
    && curl --netrc-file /dev/null -L https://ftp.gnu.org/gnu/tar/tar-${TAR_VERSION}.tar.gz -o gnu-tar.tar.gz \
    && tar -xf gnu-tar.tar.gz \
    && cd tar-${TAR_VERSION} && ./configure --prefix=/usr && make -j"$(nproc)" && make install && cd .. \
    && rm -rf gnu-tar.tar.gz tar-${TAR_VERSION} \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && ldconfig \
    && python3 --version

WORKDIR /app

# Копируем requirements.txt
COPY requirements.txt .

# Создаем виртуальное окружение
ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Устанавливаем зависимости (pip >=25.2 включает requests >=2.32.4 и устраняет CVE-2023-32681)
RUN pip install --no-cache-dir pip==25.2 'setuptools<81' wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    find /app/venv -type d -name '__pycache__' -exec rm -rf {} + && \
    find /app/venv -type f -name '*.pyc' -delete

# Этап выполнения (минимальный образ)
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /app

# Установка минимальных пакетов для выполнения и обновление критических библиотек
# Обновление linux-libc-dev устраняет CVE-2024-50217 и CVE-2025-21976, а libgcrypt20 — CVE-2024-2236
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    linux-libc-dev \
    libgcrypt20 \
    python3 \
    python3-venv \
    zlib1g \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && ldconfig \
    && python3 --version

# Копируем виртуальное окружение из этапа сборки
COPY --from=builder /app/venv /app/venv
# Копируем обновлённый GNU tar для устранения CVE-2025-45582
COPY --from=builder /usr/bin/tar /usr/bin/tar
RUN tar --version

# Копируем исходный код в /app/bot
COPY . /app/bot

# Устанавливаем переменные окружения для виртуального окружения
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Добавляем PYTHONPATH, чтобы модуль bot был доступен
ENV PYTHONPATH=/app

# Optionally enable TensorFlow checks during build
ARG ENABLE_TF=0

# Проверяем версии библиотек и доступность CUDA с отладкой
RUN echo "Checking library versions and CUDA availability..." && \
    /app/venv/bin/python3 -c "import ccxt; print('CCXT Version:', ccxt.__version__)" || echo "CCXT check failed" && \
    /app/venv/bin/python3 -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('CUDA Device Count:', torch.cuda.device_count()); print('CUDA Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')" || echo "PyTorch check failed" && \
    /app/venv/bin/python3 -c "import ray; print('Ray Version:', ray.__version__)" || echo "Ray check failed" && \
    /app/venv/bin/python3 -c "import optuna; print('Optuna Version:', optuna.__version__)" || echo "Optuna check failed" && \
    /app/venv/bin/python3 -c "import shap; print('SHAP Version:', shap.__version__)" || echo "SHAP check failed" && \
    /app/venv/bin/python3 -c "import numba; print('Numba Version:', numba.__version__)" || echo "Numba check failed" && \
    if [ "$ENABLE_TF" = "1" ]; then /app/venv/bin/python3 -c "import tensorflow as tf; print('TF Version:', tf.__version__)" || echo "TensorFlow check failed"; else echo "TensorFlow check skipped"; fi && \
    /app/venv/bin/python3 -c "import stable_baselines3 as sb3; print('SB3 Version:', sb3.__version__)" || echo "SB3 check failed" && \
    /app/venv/bin/python3 -c "import pytorch_lightning as pl; print('Lightning Version:', pl.__version__)" || echo "Lightning check failed" && \
    /app/venv/bin/python3 -c "import mlflow; print('MLflow Version:', mlflow.__version__)" || echo "MLflow check failed"

# Указываем команду для запуска
CMD ["/app/venv/bin/python3", "-m", "bot.trading_bot"]
