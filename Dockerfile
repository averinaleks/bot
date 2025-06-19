# Этап сборки
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
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

COPY requirements.txt .

ENV VIRTUAL_ENV=/app/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir -r requirements.txt \
    && find /app/venv -type d -name '__pycache__' -exec rm -rf {} + \
    && find /app/venv -type f -name '*.pyc' -delete

# Этап выполнения
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/venv /app/venv
COPY . .

ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN /app/venv/bin/python3.12 -c "import ccxt; print('CCXT Version:', ccxt.__version__)" \
    && /app/venv/bin/python3.12 -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())" \
    && /app/venv/bin/python3.12 -c "import ray; print('Ray Version:', ray.__version__)" \
    && /app/venv/bin/python3.12 -c "import optuna; print('Optuna Version:', optuna.__version__)" \
    && /app/venv/bin/python3.12 -c "import shap; print('SHAP Version:', shap.__version__)" \
    && /app/venv/bin/python3.12 -c "import numba; print('Numba Version:', numba.__version__)"

CMD ["/app/venv/bin/python3.12", "trading_bot.py"]