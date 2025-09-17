# Этап сборки
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04 AS builder
ARG ZLIB_VERSION=1.3.1
ARG ZLIB_SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV TF_CPP_MIN_LOG_LEVEL=3

# Установка необходимых пакетов для сборки и обновление критических библиотек
# Обновление linux-libc-dev устраняет CVE-2024-50217 и CVE-2025-21976, а libgcrypt20 — CVE-2024-2236.
# Дополнительно собираем пропатченные пакеты PAM, чтобы закрыть CVE-2024-10963 (HIGH).
COPY docker/patches/linux-pam-CVE-2024-10963.patch /tmp/security/linux-pam-CVE-2024-10963.patch
COPY docker/scripts/update_pam_changelog.py /tmp/security/update_pam_changelog.py
RUN set -eux; \
    apt-get update; \
    apt-get dist-upgrade -y; \
    apt-get install -y --no-install-recommends \
        tzdata \
        linux-libc-dev \
        build-essential \
        ca-certificates \
        wget \
        curl \
        python3-dev \
        python3-venv \
        python3-pip \
        libssl-dev \
        libffi-dev \
        libblas-dev \
        liblapack-dev \
        tar \
        devscripts \
        equivs; \
    python3 -m pip install --no-compile --no-cache-dir --break-system-packages \
        'pip>=24.0' \
        'setuptools>=78.1.1,<81' \
        wheel; \
    if command -v python3.11 >/dev/null 2>&1; then \
        python3.11 -m ensurepip --upgrade; \
        python3.11 -m pip install --no-compile --no-cache-dir --break-system-packages \
            'setuptools>=78.1.1,<81'; \
    fi; \
    curl --netrc-file /dev/null -L https://zlib.net/zlib-${ZLIB_VERSION}.tar.gz -o zlib.tar.gz; \
    echo "${ZLIB_SHA256}  zlib.tar.gz" | sha256sum -c -; \
    (find /usr -type l -lname "*..*" -print 2>/dev/null || true); \
    # Используем --keep-directory-symlink для предотвращения CVE-2025-45582
    tar --keep-directory-symlink --no-overwrite-dir -xf zlib.tar.gz; \
    cd zlib-${ZLIB_VERSION} && ./configure --prefix=/usr && make -j"$(nproc)" && make install && cd ..; \
    rm -rf zlib.tar.gz zlib-${ZLIB_VERSION}; \
    printf 'deb-src http://archive.ubuntu.com/ubuntu noble main restricted universe multiverse\n' \
           'deb-src http://archive.ubuntu.com/ubuntu noble-updates main restricted universe multiverse\n' \
           'deb-src http://security.ubuntu.com/ubuntu noble-security main restricted universe multiverse\n' \
           > /etc/apt/sources.list.d/noble-src.list; \
    apt-get update; \
    apt-get build-dep -y pam; \
    apt-get source -y pam; \
    cd pam-*; \
    patch -p1 < /tmp/security/linux-pam-CVE-2024-10963.patch; \
    python3 /tmp/security/update_pam_changelog.py; \
    export DEB_BUILD_OPTIONS=nocheck; \
    dpkg-buildpackage -b -uc -us; \
    cd ..; \
    mkdir -p /tmp/pam-fixed; \
    cp libpam-modules_* libpam-modules-bin_* libpam-runtime_* libpam0g_* /tmp/pam-fixed/; \
    dpkg -i /tmp/pam-fixed/*.deb; \
    rm -rf pam-* /etc/apt/sources.list.d/noble-src.list; \
    apt-get purge -y --auto-remove devscripts equivs; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*; \
    ldconfig; \
    python3 --version; \
    openssl version; \
    curl --version; \
    gpg --version; \
    dirmngr --version

WORKDIR /app

# Copy requirements files
COPY requirements-core.txt requirements-gpu.txt ./

# Создаем виртуальное окружение
ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Устанавливаем зависимости (pip >=24.0 устраняет CVE-2023-32681, setuptools>=78.1.1 закрывает свежие уязвимости)
RUN pip install --no-compile --no-cache-dir 'pip>=24.0' 'setuptools>=78.1.1,<81' wheel && \
    pip install --no-compile --no-cache-dir -r requirements-core.txt -r requirements-gpu.txt && \
    RAY_JARS_DIR=$($VIRTUAL_ENV/bin/python -c "import os, ray; print(os.path.join(os.path.dirname(ray.__file__), 'jars'))") && \
    rm -f "$RAY_JARS_DIR"/commons-lang3-*.jar && \
    curl --retry 5 --retry-delay 5 -fsSL https://repo1.maven.org/maven2/org/apache/commons/commons-lang3/3.18.0/commons-lang3-3.18.0.jar -o "$RAY_JARS_DIR"/commons-lang3-3.18.0.jar && \
    find /app/venv -type d -name '__pycache__' -exec rm -rf {} + && \
    find /app/venv -type f -name '*.pyc' -delete && \
    pip uninstall -y pip setuptools wheel && \
    find /app/venv -name "*.so" -exec strip --strip-unneeded {} +

# Этап выполнения (минимальный образ)
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04
ARG PYTHON_VERSION=3.12.3-1ubuntu0.7
ARG PYTHON_META=3.12.3-0ubuntu2
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV TF_CPP_MIN_LOG_LEVEL=3

WORKDIR /app

# Копируем виртуальное окружение из этапа сборки
COPY --from=builder /app/venv /app/venv
COPY --from=builder /tmp/pam-fixed /tmp/pam-fixed

# Установка минимальных пакетов выполнения
RUN apt-get update && apt-get dist-upgrade -y && apt-get install -y --no-install-recommends \
    python3 \
    libpython3.12-stdlib \
    coreutils \
    zlib1g \
    libpam0g \
    libpam-modules \
    && dpkg -i /tmp/pam-fixed/*.deb \
    && if command -v python3.11 >/dev/null 2>&1; then \
        python3.11 -m ensurepip --upgrade; \
        python3.11 -m pip install --no-cache-dir --break-system-packages 'setuptools>=78.1.1,<81'; \
    fi \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/pam-fixed \
    && ldconfig \
    && /app/venv/bin/python --version \
    && openssl version

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
    /app/venv/bin/python3 -c "import stable_baselines3; print('SB3 Version:', stable_baselines3.__version__)" || echo "SB3 check failed" && \
    if [ "$ENABLE_TF" = "1" ]; then /app/venv/bin/python3 -c "import tensorflow as tf; print('TF Version:', tf.__version__)" || echo "TensorFlow check failed"; else echo "TensorFlow check skipped"; fi && \
    /app/venv/bin/python3 -c "import pytorch_lightning as pl; print('Lightning Version:', pl.__version__)" || echo "Lightning check failed" && \
    /app/venv/bin/python3 -c "import mlflow; print('MLflow Version:', mlflow.__version__)" || echo "MLflow check failed"

# Use a dedicated non-root user at runtime
RUN groupadd --system bot && useradd --system --gid bot --home-dir /home/bot --shell /bin/bash bot \
    && mkdir -p /home/bot \
    && chown -R bot:bot /app /home/bot

USER bot

# Указываем команду для запуска
CMD ["/app/venv/bin/python3", "-m", "bot.trading_bot"]
