# syntax=docker/dockerfile:1
# Security stage with a reusable hardened base layer
ARG SECURITY_BASE_IMAGE=nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04
FROM ${SECURITY_BASE_IMAGE} AS security_base
ARG ZLIB_VERSION=1.3.1
ARG ZLIB_SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install build dependencies and refresh critical libraries.
# When a prepared SECURITY_BASE_IMAGE is available, the rebuild block is skipped
# via the `/opt/security-layer/.ready` marker.
COPY docker/patches/linux-pam-CVE-2024-10963.patch /tmp/security/linux-pam-CVE-2024-10963.patch
COPY docker/patches/linux-pam-CVE-2024-10041.patch /tmp/security/linux-pam-CVE-2024-10041.patch
COPY docker/patches/linux-pam-hardening.patch /tmp/security/linux-pam-hardening.patch
COPY docker/scripts/update_pam_changelog.py /tmp/security/update_pam_changelog.py
COPY docker/scripts/setup_zlib_and_pam.sh /tmp/security/setup_zlib_and_pam.sh
COPY docker/scripts/build_patched_pam.sh /tmp/security/build_patched_pam.sh
COPY docker/scripts/harden_gnutar.sh /tmp/security/harden_gnutar.sh

WORKDIR /tmp/build

RUN <<'EOSHELL'
set -eux
mkdir -p /opt/security-layer
if [ -f /opt/security-layer/.ready ]; then
  echo "Security layer already provisioned, skipping zlib/PAM rebuild"
else
  apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg
  mkdir -p /etc/apt/keyrings
  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
    | gpg --dearmor -o /etc/apt/keyrings/cuda-archive-keyring.gpg
  if [ -f /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list ]; then
    sed -i 's#\\[signed-by=[^]]*\\]#[signed-by=/etc/apt/keyrings/cuda-archive-keyring.gpg]#g' \
      /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list
    if ! grep -q 'signed-by' /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list; then
      sed -i 's#^deb #deb [signed-by=/etc/apt/keyrings/cuda-archive-keyring.gpg] #' \
        /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list
    fi
  fi
  apt-get update \
    && apt-get install -y --no-install-recommends \
    tzdata \
    linux-libc-dev \
    build-essential \
    patch \
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
    equivs
  /bin/bash /tmp/security/harden_gnutar.sh
  PYTOOLS_VENV=/opt/security-layer/pytools
  python3 -m venv "$PYTOOLS_VENV"
  PATH="$PYTOOLS_VENV/bin:$PATH"
  "$PYTOOLS_VENV/bin/pip" install --no-compile --no-cache-dir \
    'pip>=25.3' \
    'setuptools>=80.9.0,<81' \
    wheel
  if command -v python3.11 >/dev/null 2>&1; then
    PYTOOLS_VENV311=/opt/security-layer/pytools-py311
    python3.11 -m venv "$PYTOOLS_VENV311"
    "$PYTOOLS_VENV311/bin/pip" install --no-compile --no-cache-dir \
      'setuptools>=80.9.0,<81'
  fi
  curl --netrc-file /dev/null -L "https://zlib.net/zlib-${ZLIB_VERSION}.tar.gz" -o zlib.tar.gz
  echo "${ZLIB_SHA256}  zlib.tar.gz" | sha256sum -c -
  /bin/bash /tmp/security/setup_zlib_and_pam.sh
  cd /tmp/build/zlib-src
  ./configure --prefix=/usr
  make -j"$(nproc)"
  make install
  cd /tmp/build
  rm -rf zlib.tar.gz zlib-src
  /tmp/security/build_patched_pam.sh \
    "/tmp/security/linux-pam-hardening.patch /tmp/security/linux-pam-CVE-2024-10963.patch /tmp/security/linux-pam-CVE-2024-10041.patch" \
    /tmp/security/update_pam_changelog.py noble /tmp/security/pam-build /tmp/pam-fixed
  ldconfig
  python3 --version
  openssl version
  curl --version
  gpg --version
  dirmngr --version
  mkdir -p /opt/security-layer/pam-fixed
  rm -rf /opt/security-layer/pam-fixed/*
  cp /tmp/pam-fixed/*.deb /opt/security-layer/pam-fixed/
  rm -rf /tmp/pam-fixed /tmp/security/pam-build
  apt-get clean
  rm -rf /var/lib/apt/lists/*
  touch /opt/security-layer/.ready
fi
EOSHELL

RUN rm -rf /tmp/build /tmp/security

# Application build stage
FROM security_base AS builder
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3

WORKDIR /app

# Copy requirements files
COPY requirements-core.txt requirements-gpu.txt ./

# Create an isolated virtual environment
ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY docker/scripts/update_commons_lang3.py /tmp/security/update_commons_lang3.py

# Install dependencies (pip >=25.3 mitigates CVE-2025-8869 and CVE-2023-32681; setuptools>=80.9.0
# addresses recent vulnerabilities)
RUN pip install --no-compile --no-cache-dir 'pip>=25.3' 'setuptools>=80.9.0,<81' wheel && \
    pip install --no-compile --no-cache-dir -r requirements-core.txt -r requirements-gpu.txt && \
    $VIRTUAL_ENV/bin/python /tmp/security/update_commons_lang3.py

RUN find /app/venv -type d -name '__pycache__' -exec rm -rf {} + && \
    find /app/venv -type f -name '*.pyc' -delete && \
    pip uninstall -y pip setuptools wheel && \
    find /app/venv -name "*.so" -exec strip --strip-unneeded {} +

# Runtime stage (minimal image)
FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04
ARG PYTHON_VERSION=3.12.3-1ubuntu0.7
ARG PYTHON_META=3.12.3-0ubuntu2
ARG APP_UID=1000
ARG APP_GID=1000
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3

WORKDIR /app

# Copy virtual environment from the build stage
COPY --from=builder /app/venv /app/venv
COPY --from=security_base /opt/security-layer/pam-fixed /tmp/pam-fixed
COPY docker/scripts/harden_gnutar.sh /tmp/security/harden_gnutar.sh

# Install minimal runtime packages.
# ``apt-get upgrade`` is intentionally skipped to satisfy the
# ``dockerfile.security.apt-get-upgrade`` Semgrep rule and keep builds
# reproducible.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libpython3.12-stdlib \
    python3.12-venv \
    python3-pip \
    coreutils \
    zlib1g \
    libpam0g \
    libpam-modules \
    libssl3 \
    openssl \
    ca-certificates \
    && python3 -m venv /tmp/runtime-packaging \
    && . /tmp/runtime-packaging/bin/activate \
    && pip install --no-cache-dir 'setuptools>=80.9.0,<81' \
    && deactivate \
    && apt-get install -y --no-install-recommends \
        /tmp/pam-fixed/libpam0g_* \
        /tmp/pam-fixed/libpam-runtime_* \
        /tmp/pam-fixed/libpam-modules-bin_* \
        /tmp/pam-fixed/libpam-modules_* \
    && /bin/bash /tmp/security/harden_gnutar.sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/pam-fixed /tmp/runtime-packaging \
    && ldconfig \
    && /app/venv/bin/python --version \
    && openssl version

# Copy project source into /app/bot
COPY . /app/bot

# Configure environment variables for the virtual environment
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Extend PYTHONPATH so the bot package is importable
ENV PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD /app/venv/bin/python -c "import bot" || exit 1

RUN mkdir -p /app/logs

# Optionally enable TensorFlow checks during build
ARG ENABLE_TF=0

# Verify library versions and CUDA availability for diagnostics
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

# Use a dedicated non-root user at runtime aligned with typical host UID/GID
RUN groupadd --system --gid ${APP_GID} bot && useradd --system --uid ${APP_UID} --gid bot --home-dir /home/bot --shell /bin/bash bot \
    && mkdir -p /home/bot \
    && chown -R bot:bot /app /home/bot

USER bot

# Define the default entrypoint command
CMD ["/app/venv/bin/python3", "-m", "bot.trading_bot"]
