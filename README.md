# Trading Bot

Этот репозиторий содержит пример торгового бота на Python. Для запуска необходимы файлы `config.json` и `.env` с ключами API.

**Disclaimer**: This project is provided for educational purposes only and does not constitute financial advice. Use at your own risk.

## Быстрый старт

1. Установите зависимости. Варианты для GPU и CPU:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   # Вариант с GPU (по умолчанию)
   python -m pip install -r requirements.txt
   # Или CPU‑сборки без CUDA
   python -m pip install -r requirements-cpu.txt
   ```
   Список `requirements-cpu.txt` содержит версии `torch` и `tensorflow` без поддержки GPU. Его можно использовать для установки зависимостей и запуска тестов на машинах без CUDA.
- После обновления зависимостей пакет `optuna-integration[botorch]` больше не используется.
- Библиотека `catalyst` закреплена на версии `21.4`, так как новые версии не устанавливаются с `pip>=24.1`. Если требуется `catalyst>=22.2`, понизьте `pip` ниже 24.1.
2. Отредактируйте файл `.env`, указав в нём свои значения. При запуске основные
    скрипты вызывают `load_dotenv()` из библиотеки `python-dotenv`, поэтому
    переменные из файла подхватываются автоматически. При необходимости задайте
    торговую пару через переменную `SYMBOL` (по умолчанию используется
    `BTCUSDT`). Через переменную `HOST` можно задать адрес, на котором
    запускаются сервисы Flask (по умолчанию `0.0.0.0`). Переменная `LOG_LEVEL`
    задаёт уровень логирования (например, `DEBUG`, `INFO`, `WARNING`), а
    `LOG_DIR` определяет каталог для файлов логов. После инициализации бот не
    печатает `INFO`‑сообщения при обычной работе. Для подробного вывода
    запустите контейнеры с `LOG_LEVEL=DEBUG`:

    ```bash
    LOG_LEVEL=DEBUG docker compose up --build
    ```

    Непрерывный вывод смотрите в файлах внутри `./logs/`.

    Дополнительные переменные для вспомогательных сервисов:

    - `STREAM_SYMBOLS` — список пар через запятую, которые `data_handler_service`
      обновляет в фоне.
    - `CACHE_TTL` и `UPDATE_INTERVAL` — время жизни кэша OHLCV и интервал
      фонового обновления (в секундах).
    - `MODEL_DIR` — каталог, где `model_builder_service` хранит обученные модели
      по символам.
    - `BYBIT_API_KEY` и `BYBIT_API_SECRET` — ключи API. Их читают как
      `trade_manager_service`, так и `data_handler_service`, поэтому укажите их
      для обоих контейнеров. Переменные `TELEGRAM_BOT_TOKEN` и
      `TELEGRAM_CHAT_ID` нужны для уведомлений. Убедитесь, что все значения
      доступны сервисам, например через `env_file: .env` или секцию
      `environment:` в `docker-compose.yml`. `DataHandler` и `TradeManager`
      проверяют эти переменные при запуске и выводят предупреждение, если они
      отсутствуют. В этом случае Telegram уведомления отправляться не будут.

    Пример `docker-compose` с передачей этих переменных обоим сервисам:

    ```yaml
    services:
      data_handler:
        env_file: .env
        # Или задайте переменные напрямую:
        # environment:
        #   BYBIT_API_KEY: ${BYBIT_API_KEY}
        #   BYBIT_API_SECRET: ${BYBIT_API_SECRET}

      trade_manager:
        env_file: .env
    ```
3. Отредактируйте `config.json` под свои нужды. Помимо основных настроек можно
   задать параметры адаптации порогов:
   - `loss_streak_threshold` и `win_streak_threshold` контролируют количество
     подряд убыточных или прибыльных сделок, после которого базовый порог
     вероятности будет соответственно повышен или понижен.
   - `threshold_adjustment` задаёт величину изменения порога.
   - `target_change_threshold` задаёт минимальный процент изменения цены для положительной метки при обучении модели.
  - `backtest_interval` определяет, как часто выполняется автоматический бектест стратегии.
  - `optimization_interval` задаёт базовый интервал оптимизации параметров. Фактический запуск происходит динамически и может сокращаться при росте волатильности.
  - `enable_grid_search` включает дополнительную проверку лучших параметров через GridSearchCV после оптимизации Optuna.
  - `max_symbols` задаёт количество наиболее ликвидных торговых пар, которые бот выберет из доступных.
   - `max_subscriptions_per_connection` определяет, сколько символов подписывается через одно WebSocket‑соединение.
   - `telegram_queue_size` ограничивает размер очереди сообщений Telegram.
4. Запустите бота. При локальном запуске без Docker Compose задайте адреса сервисов переменными `DATA_HANDLER_URL`, `MODEL_BUILDER_URL` и `TRADE_MANAGER_URL`:
```bash
DATA_HANDLER_URL=http://localhost:8000 \
MODEL_BUILDER_URL=http://localhost:8001 \
TRADE_MANAGER_URL=http://localhost:8002 \
python trading_bot.py
```
Эти переменные задают URL-адреса сервисов `data_handler`, `model_builder` и `trade_manager`. В Compose они не требуются, так как сервисы обнаруживаются по имени.
Перед запуском убедитесь, что сервисы отвечают на `/ping`. В Docker Compose это происходит автоматически через встроенные health check'и, так что дополнительных настроек не требуется. При запуске вне Compose бот использует функцию `check_services`, которая повторяет запросы к `/ping`. Количество попыток и пауза между ними настраиваются переменными `SERVICE_CHECK_RETRIES` и `SERVICE_CHECK_DELAY`. По умолчанию бот делает 30 попыток с задержкой 2 секунды.
Также можно использовать `docker-compose up --build` для запуска в контейнере.

В зависимости от версии Docker команда может называться `docker compose` или
`docker-compose`.
По умолчанию используется образ с поддержкой GPU. Если она не требуется,
запустите compose с переменной `DOCKERFILE` и отключите NVIDIA-переменные:

```bash
RUNTIME= DOCKERFILE=Dockerfile.cpu NVIDIA_VISIBLE_DEVICES= NVIDIA_DRIVER_CAPABILITIES= docker-compose up --build
```

Set `RUNTIME=` if you want to run these CPU images without the NVIDIA runtime.

GPU acceleration with Numba requires the `libnvvm.so` library. The default
`Dockerfile` uses the `nvidia/cuda:*‑cudnn-devel` image so NVVM is available
at runtime. If you switch to a runtime-only base (for example
`*-runtime` or `*-cudnn-runtime`), copy the NVVM libraries from the build stage
or Numba will fall back to CPU mode. Build and run the GPU image with:

```bash
DOCKERFILE=Dockerfile docker compose up --build
```

Setting `FORCE_CPU=1` disables all CUDA checks, which helps avoid crashes such as
`free(): double free detected in tcache 2` when CUDA drivers are missing or
misconfigured.

When running with GPUs the `trade_manager` service automatically configures
Python's multiprocessing start method to ``"spawn"``. Forking workers can lead to
CUDA initialization errors, so the module switches to ``"spawn"`` on import. The
default should work for most setups, but you may call
``multiprocessing.set_start_method()`` yourself before launching the service if
you need a different policy.

The `trade_manager` container needs extra shared memory. The compose file
allocates 8GB via `shm_size: '8gb'` to enlarge `/dev/shm`.

The `model_builder` service sets `TF_CPP_MIN_LOG_LEVEL=3` to hide verbose TensorFlow
GPU warnings. Adjust or remove this variable if you need more detailed logs.

When both TensorFlow and PyTorch start in the same container you might see
messages like `Unable to register cuDNN factory` or `computation placer already
registered`. These lines appear while each framework loads CUDA plugins and
tries to register them more than once. They are warnings, not fatal errors, and
can be safely ignored. Building the image with `Dockerfile.cpu` avoids them
entirely.

Earlier revisions started lightweight stubs for the supporting services.  This
repository now includes simple reference implementations in the `services`
directory. `data_handler_service.py` fetches live prices from Bybit using
`ccxt`, `model_builder_service.py` trains a small logistic regression
model when you POST data to `/train`, and `trade_manager_service.py` can
place market orders on Bybit when you POST to `/open_position` or
`/close_position`.  Start it with:

```
python services/trade_manager_service.py
```
It also exposes `/positions` and `/ping` routes for status checks.

The data handler exposes two endpoints:

``/price/<symbol>``
    Return the latest ticker price.

``/ohlcv/<symbol>``
    Return cached OHLCV bars for ``symbol``. The service periodically refreshes
    data for symbols listed in ``STREAM_SYMBOLS``.  Cache lifetime is controlled
    by ``CACHE_TTL``.


The reference scripts exist purely for demonstration and integration testing.
They expose the same HTTP routes as the real services but avoid heavy
dependencies.  Use them when you just want to verify the bot's basic workflow
without setting up full ML libraries.

The model builder maintains separate models per trading pair.  POST JSON data
of the form::

    {"symbol": "BTC/USDT", "features": [[...], [...]], "labels": [0, 1]}

### Switching implementations

`docker-compose.yml` uses the full implementations in `data_handler.py` and
`model_builder.py`. They depend on heavy packages like TensorFlow and PyTorch
which are installed in the Docker image. For lightweight testing you can run
the reference services instead. Replace the `command` entries for each service
with the scripts from the `services` directory:

```yaml
data_handler:
  command: python services/data_handler_service.py
model_builder:
  command: python services/model_builder_service.py
trade_manager:
  command: python services/trade_manager_service.py
```

Restore the Gunicorn commands when you want to launch the full services.

### Running the full services

Running these full modules requires TensorFlow, PyTorch and related
libraries.  They may take noticeably longer to start while the frameworks
initialise and will use a GPU if one is available.  Ensure the compose file
sets `RUNTIME=nvidia` and that your system has the NVIDIA container runtime
installed.  Without a GPU you can still run the services with
`DOCKERFILE=Dockerfile.cpu` but startup will remain slower than the lightweight
scripts above.

The default `docker-compose.yml` already points to the full-featured
implementations.  If you replaced the `command` entries with the minimal scripts
earlier, simply revert those lines or copy the compose file from the repository
again.  After restoring the Gunicorn commands, run:

```bash
docker compose up --build
```

so the heavy frameworks load and the services expose their production APIs.






## Docker Compose logs

Просмотреть вывод контейнеров можно через `docker compose logs` (или
`docker-compose logs` в более старых версиях Docker).
Выберите нужный сервис или добавьте флаг `-f` для режима слежения:

```bash
docker compose logs data_handler          # логи DataHandler
docker-compose logs data_handler         # для docker-compose
docker compose logs -f trade_manager     # следить за выводом TradeManager
docker-compose logs -f trade_manager
docker compose logs model_builder
docker-compose logs model_builder
```

Команда без аргументов печатает логи всех сервисов.
Если GPU недоступен, собирайте образ через переменную `DOCKERFILE=Dockerfile.cpu`
и укажите `RUNTIME=`, чтобы отключить NVIDIA‑runtime:

```bash
RUNTIME= DOCKERFILE=Dockerfile.cpu docker compose up --build
```

## Troubleshooting service health

If `trading_bot.py` exits with `dependent services are unavailable`,
use these steps to diagnose the problem:

1. Inspect container logs to see why a service failed to start:

   ```bash
   docker compose logs data_handler
   docker compose logs model_builder
   docker compose logs trade_manager
   ```

2. GPU images require the NVIDIA runtime (`nvidia-container-toolkit` or
   `nvidia-docker2`). Verify that the GPU is visible from a container:

   ```bash
   docker compose run --rm data_handler nvidia-smi
   ```

   If no GPU is detected, rebuild with the CPU Dockerfile:

   ```bash
   RUNTIME= DOCKERFILE=Dockerfile.cpu docker compose up --build
   ```

3. If services require more time to initialize, increase
   `SERVICE_CHECK_RETRIES` or `SERVICE_CHECK_DELAY` in `.env`.
   By default, the bot performs 30 retries with a 2‑second delay.
4. If logs contain `gymnasium import failed`, install the package manually with `pip install gymnasium`.
5. When RL components start, they import `gymnasium`.
  If the package is missing, training will fail until you install it.
6. If `gunicorn` logs show `WORKER TIMEOUT` messages, the service
   might need more time to respond. Set `GUNICORN_TIMEOUT` in your
   environment to increase the timeout in seconds. The compose file
   defaults to `120` seconds.
7. Use an async-capable worker for `gunicorn` (e.g. `--worker-class
   uvicorn.workers.UvicornWorker`) so async views like the trade
   manager's `/open_position` route can schedule tasks properly.

## Telegram notifications

Set the `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` variables in `.env` to
enable notifications. Create the bot with `telegram.Bot` and pass it to both
`DataHandler` and `TradeManager`:

```python
from telegram import Bot
import json
import os

with open("config.json") as f:
    cfg = json.load(f)

bot = Bot(os.environ["TELEGRAM_BOT_TOKEN"])
chat_id = os.environ["TELEGRAM_CHAT_ID"]

data_handler = DataHandler(cfg, bot, chat_id)
model_builder = ModelBuilder(cfg, data_handler, None)
trade_manager = TradeManager(cfg, data_handler, model_builder, bot, chat_id)

You can limit the logger queue with `telegram_queue_size` in `config.json`.
```

Both services check the `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`
variables at startup. If either variable is missing, a warning is logged and
no Telegram alerts are sent.

To avoid processing old updates after a restart, store the `update_id` and pass
it to the `offset` parameter when calling `get_updates`. The helper class
`TelegramUpdateListener` handles this automatically and logs any errors.

Telegram enforces message limits per bot account, so duplicate notifications are
typically caused by bugs rather than global restrictions. Ensure that your code
filters repeated messages and checks that `send_message` returns HTTP 200.

You can run this bot either with long polling or a webhook using the
`Application` class from `python-telegram-bot`.

When deploying the `trade_manager` service with Gunicorn, use a single worker
(`-w 1`). This ensures only one `TelegramUpdateListener` polls the bot token,
preventing duplicated updates.

## Лимиты WebSocket-подписок

Количество подписок через одно соединение ограничивается параметром `max_subscriptions_per_connection`. Если список пар превышает это значение, бот откроет дополнительные WebSocket-соединения.

Подписки отправляются пакетами. Размер пакета определяется параметром `max_subscriptions_per_connection`. При необходимости бот делает паузу между отправками, чтобы не превышать ограничения биржи на частоту запросов.

Пример настроек в `config.json`:

```json
{
    "max_subscriptions_per_connection": 15
}
```

Так бот будет разбивать список подписок на пакеты по 10–15 символов и делать небольшую паузу между отправками. Для большинства бирж этого достаточно, чтобы не превышать ограничения на частоту запросов.

## Model training

`ModelBuilder` автоматически обучает и переобучает нейросетевую модель CNN‑LSTM.
Для организации процесса можно выбрать ML-фреймворк: `pytorch` (по умолчанию), `lightning` или `keras`.
Тип указывается параметром `nn_framework` в `config.json`.
Для каждого актива формируются признаки из OHLCV‑данных и технических
индикаторов. Метка `1` присваивается, если через
`lstm_timesteps` баров цена выросла больше, чем на значение
`target_change_threshold` (по умолчанию `0.001`). Иначе метка равна `0`.

Обучение запускается удалённо через `ray` и использует GPU при наличии.
- После обучения выполняется калибровка вероятностей с помощью логистической
регрессии. Затем `TradeManager` применяет откалиброванные прогнозы для
открытия и закрытия позиций согласно динамическим порогам.

Для интерпретации модели вычисляются SHAP‑значения, и краткий отчёт о наиболее
важных признаках отправляется в Telegram.

Пример запуска обучения вместе с ботом:

```bash
python trading_bot.py
```

### RL agents

Модуль `RLAgent` может обучать модели с помощью `stable-baselines3`, `Ray RLlib` или фреймворка `Catalyst` от Яндекса.
Выберите подходящий движок параметром `rl_framework` в `config.json` (`stable_baselines3`, `rllib` или `catalyst`).
Алгоритм указывается опцией `rl_model` (`PPO` или `DQN`), продолжительность обучения — `rl_timesteps`.
Для корректной работы `stable-baselines3` необходим пакет `gymnasium`. Он
устанавливается по умолчанию вместе с зависимостями проекта, поэтому
альтернативный `gym` не требуется.

Периодическое переобучение задаётся параметром `retrain_interval` в
`config.json`. Можно запускать бота по расписанию (например, через `cron`) для
регулярного обновления моделей:

```bash
0 3 * * * cd /path/to/bot && /usr/bin/python trading_bot.py
```

## MLflow

Если установить пакет `mlflow` и включить флаг `mlflow_enabled` в `config.json`,
бот будет сохранять параметры и метрики обучения в указанное хранилище.
По умолчанию используется локальная папка `mlruns`.
Чтобы задать другой адрес, укажите ключ `mlflow_tracking_uri` в `config.json`
или переменную окружения `MLFLOW_TRACKING_URI`.

Пример запуска с отслеживанием экспериментов:

```bash
MLFLOW_TRACKING_URI=mlruns python trading_bot.py
```

## Running tests

Running `pytest` requires the packages listed in `requirements-cpu.txt`.
Install them using the helper script:

```bash
./scripts/setup-tests.sh        # CPU packages only
```

To install the GPU-enabled packages from `requirements.txt` instead, run:

```bash
./scripts/install-test-deps.sh --full
```

If you skip this step and run `pytest` anyway, common imports like
`pandas`, `numpy` or `scipy` will be missing and the tests will abort
with `ModuleNotFoundError` errors.

For a clean environment you can create a virtualenv and install the
CPU requirements before running the tests:

```bash
    python3 -m venv venv
    source venv/bin/activate
    ./scripts/setup-tests.sh
    pytest
```

The `requirements-cpu.txt` file already bundles `pytest` and all other
packages needed by the test suite.

Unit tests automatically set the environment variable `TEST_MODE=1`.
This disables the Telegram logger's background worker thread so tests
run without spawning extra threads.

As noted above, make sure to run `./scripts/setup-tests.sh` before
executing `pytest`; otherwise imports such as `numpy`, `pandas`, `scipy` and
`requests` will fail.

Чтобы выполнить тесты на машине без GPU, создайте виртуальное окружение и установите зависимости из `requirements-cpu.txt`.
Пакет включает CPU‑сборки `torch` и `tensorflow`, поэтому тесты не подтягивают CUDA‑библиотеки.

Пример полного процесса:

```bash
    python3 -m venv venv
    source venv/bin/activate
    ./scripts/setup-tests.sh
    pytest
```

Если у вас есть GPU и установленный CUDA, можно установить полный список
зависимостей командой `./scripts/install-test-deps.sh --full` и затем
  запустить те же тесты.

The `requirements.txt` file already includes test-only packages such as
`pytest`, `optuna` and `tenacity`, so no separate `requirements-dev.txt`
is required.

The test suite relies on the following packages:

- numpy
- pandas
- torch
- torchvision
- ccxt
- ccxtpro
- pybit
- python-telegram-bot
- aiohttp
- websockets
- ta
- scikit-learn
- optuna
- psutil
- python-dotenv
- joblib
- imbalanced-learn
- statsmodels
- scipy
- shap
- tenacity
- pyarrow
- jsonschema
- plotly
- numba
- ray
- stable-baselines3
- mlflow
- pytorch-lightning
- tensorflow
- catalyst
- pytest

GPU libraries such as CUDA-enabled torch or numba may be required for some tests.

## Linting


The project uses **flake8** for style checks. Install dependencies and enable
the pre-commit hook so linting runs automatically:

```bash
pip install -r requirements.txt  # or requirements-cpu.txt
pip install pre-commit
pre-commit install
```

This hook relies on `.pre-commit-config.yaml` to run `flake8` before each
commit. Trigger it manually with:

```bash
pre-commit run --all-files
```

Linting configuration is stored in `.flake8`. Run the checker manually:

```bash
python -m flake8
```

## Continuous integration

All pushes and pull requests trigger a GitHub Actions workflow that installs
dependencies via `scripts/install-test-deps.sh` (which also installs `flake8`),
runs `python -m flake8`, and executes `pytest`.
This ensures style checks and tests run automatically.
