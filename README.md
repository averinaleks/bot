# Trading Bot

Этот репозиторий содержит пример торгового бота на Python. Для запуска необходимы файлы `config.json` и `.env` с ключами API.
Бот также умеет запрашивать анализ у GPT‑сервиса через модуль `gpt_client`.
Ответ ожидается в формате JSON:

```json
{"signal": "buy", "tp_mult": 1.2, "sl_mult": 0.8}
```

где `signal` — рекомендация (`buy`, `sell` или `hold`), а `tp_mult` и `sl_mult`
задают множители для расчёта тейк‑профита и стоп‑лосса.

**Disclaimer**: This project is provided for educational purposes only and does not constitute financial advice. Use at your own risk.

## Зависимости

Основные пакеты устанавливаются из `requirements.txt` и включают, например,
`pandas`, `numpy`, `httpx`, `psutil` и `scipy`.
GPU-зависимости вынесены в отдельный файл `requirements-gpu.txt` и устанавливаются при необходимости.
Дополнительные библиотеки используются только для расширенного функционала и
устанавливаются при необходимости:

- `polars` — работа с данными в формате Polars при `use_polars=True`;
- `websockets` — потоковые обновления котировок;
- `ta` — расчёт технических индикаторов;
- `ray` — распределённые вычисления и оптимизация параметров;
- `numba` — ускорение критичных участков кода;
- `python-telegram-bot` — отправка уведомлений в Telegram.

При отсутствии любой из этих опциональных зависимостей соответствующие функции
недоступны и выдадут ошибку уровня `error` в логах.

### Работа с паролями

Модуль `password_utils.py` предоставляет функции `hash_password` и
`verify_password`, использующие библиотеку `bcrypt`. Пароли короче
8 символов, длиннее 64 символов или не содержащие цифр, букв разных
регистров и спецсимволов приводят к `ValueError`.

## Быстрый старт

1. Установите зависимости:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   python -m pip install -r requirements.txt
   # Дополнительно для GPU
   python -m pip install -r requirements-gpu.txt
   # Зависимости для тестов
   ./scripts/install-test-deps.sh
   # С GPU-зависимостями
   INSTALL_GPU_DEPS=1 ./scripts/install-test-deps.sh
   ```
   В CI применяются только стандартные колёса PyTorch без поддержки CUDA.
   Если вам нужна GPU‑сборка с CUDA 12.4, установите PyTorch
   отдельно, указав дополнительный индекс:

   ```bash
   pip install torch>=2.7.1 --extra-index-url https://download.pytorch.org/whl/cu124
   ```
   Эта команда ставит версии с поддержкой CUDA 12.4. Выполните её до
   `python -m pip install -r requirements.txt -r requirements-gpu.txt` (или вместо него, если
   остальные зависимости уже установлены).
   Эта утилита устанавливает пакеты, необходимые для запуска тестов. Выполните
   её перед `pytest`, чтобы все проверки прошли успешно.
    Файл `requirements-gpu.txt` уже включает `cupy-cuda12x` для систем с CUDA 12.x. Системы без GPU могут пропустить этот шаг и использовать только `requirements.txt`

- После обновления зависимостей пакет `optuna-integration[botorch]` больше не используется.
- Библиотека `catalyst` закреплена на версии `21.4`, так как новые версии не устанавливаются с `pip>=24.1`. Если требуется `catalyst>=22.2`, понизьте `pip` ниже 24.1.
2. Отредактируйте файл `.env`, указав в нём свои значения. При запуске основные
    скрипты вызывают `load_dotenv()` из библиотеки `python-dotenv`, поэтому
    переменные из файла подхватываются автоматически. При необходимости задайте
    торговую пару через переменную `SYMBOL` (по умолчанию используется
    `BTCUSDT`). Через переменную `HOST` можно задать адрес, на котором
    запускаются сервисы Flask. Переменная `SERVICE_SCHEME` позволяет выбрать
    схему (`http` или `https`) для автоматического формирования URL сервисов
    (по умолчанию используется `http`). Для безопасности сервис `model_builder` по
    умолчанию слушает `127.0.0.1`; чтобы открыть его наружу, укажите явное
    значение, например `HOST=0.0.0.0`. Переменная `LOG_LEVEL` задаёт уровень
    логирования (например, `DEBUG`, `INFO`, `WARNING`), а `LOG_DIR` определяет
    каталог для файлов логов. После инициализации бот не
    печатает `INFO`‑сообщения при обычной работе. Для подробного вывода
    запустите контейнеры с `LOG_LEVEL=DEBUG`:

    ```bash
    LOG_LEVEL=DEBUG docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
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
    - `TRADE_RISK_USD` — величина риска в долларах для расчёта размера позиции,
      если `/open_position` получает только `price`.
    - `GPT_MODEL` — имя модели (по умолчанию `openai/gpt-oss-20b`).
      Модель загружается лениво при запуске сервера, поэтому первые запросы к API
      могут получать код `503`, пока загрузка не завершена.
    - `API_KEYS` — список токенов через запятую для доступа к серверу FastAPI.
      Каждый запрос должен передавать заголовок `Authorization: Bearer <token>`,
      иначе сервер вернёт `401`.
    - `HF_COMMIT_SHA` — хеш коммита модели (например,
      `10e9d713f8e4a9281c59c40be6c58537480635ea`). Переменная обязательна и
      передаётся в параметр `revision` при загрузке модели; без неё сервис не
      запустится.
    - `GPT_OSS_API` — базовый адрес сервиса [GPT OSS](https://github.com/jina-ai/gpt-oss)
      (без суффикса `/completions`), к которому обращаются функции
      `query_gpt` и `query_gpt_async`. Переменная обязательна; при её отсутствии
      эти функции завершатся ошибкой. Например:

      ```python
      from bot.gpt_client import query_gpt
      result = query_gpt("Что ты видишь в этом коде:\n" + open("strategy_optimizer.py").read())
      print(result)
      ```

    - `GPT_OSS_WAIT_TIMEOUT` — время ожидания запуска сервера GPT OSS в
      секундах (по умолчанию `30`).
    - `GPT_OSS_TIMEOUT` — таймаут запроса к GPT OSS API в секундах (по
      умолчанию `5`). Используется функциями `query_gpt` и `query_gpt_async`.

     В `docker-compose.yml` теперь есть сервис `gptoss`,
     запускающий образ `ghcr.io/openaccess-ai-collective/gpt-oss:cpu-latest` на порту `8003`
     (внутри контейнера используется порт `8000`). Сервис `bot` зависит от него
     и получает переменную `GPT_OSS_API=http://localhost:8003`. Для кеширования
     моделей контейнеру
     `gptoss` требуется переменная `TRANSFORMERS_CACHE`, указывающая на путь в
     примонтированном томе.

      Дополнительно доступен сервис `gptoss_check`, который монтирует
      репозиторий и запускает `python3 gptoss_check/main.py` с тем же API.
      Поведение проверки настраивается через файл `gptoss_check.config` в
      корне проекта. В нём есть опция `skip_gptoss_check`, по умолчанию равная
      `true` — в этом случае скрипт лишь выводит сообщение о пропуске проверки.
      Установите её в `false`, чтобы выполнить анализ кода. Пути к
      анализируемым файлам задаются переменной `CHECK_CODE_PATH` — список,
      разделённый запятыми (по умолчанию `trading_bot.py`). Все эти сервисы
      подключены к общей сети `gptoss_net`.

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
   задать параметры адаптации порогов. Полное описание доступных ключей
   приведено в [CONFIG.md](CONFIG.md):
   - `loss_streak_threshold` и `win_streak_threshold` контролируют количество
     подряд убыточных или прибыльных сделок, после которого базовый порог
     вероятности будет соответственно повышен или понижен.
   - `threshold_adjustment` задаёт величину изменения порога.
   - `threshold_decay_rate` контролирует скорость возврата порога к базовому
     значению.
   - После каждого расчёта точности `ModelBuilder` обновляет словарь
     `base_thresholds`. Для каждой пары базовый порог постепенно
     приближается к фактической точности (значение ограничено диапазоном
     0.5–0.9) и сохраняется в состоянии.
   - `reversal_margin` задаёт запас отклонения от противоположного порога,
     при котором после закрытия позиции открывается новая в обратную сторону.
   - `risk_sharpe_loss_factor` и `risk_sharpe_win_factor` управляют снижением
     или повышением риска при отрицательном и высоком коэффициенте Шарпа.
     `risk_vol_min` и `risk_vol_max` ограничивают влияние волатильности на
     итоговый размер позиции.
   - `target_change_threshold` задаёт минимальный процент изменения цены для положительной метки при обучении модели.
  - `backtest_interval` определяет, как часто выполняется автоматический бектест стратегии.
  - `optimization_interval` задаёт базовый интервал оптимизации параметров. Фактический запуск происходит динамически и может сокращаться при росте волатильности.
  - `enable_grid_search` включает дополнительную проверку лучших параметров через GridSearchCV после оптимизации Optuna.
  - `use_strategy_optimizer` включает поиск общих параметров на уровне портфеля с помощью `StrategyOptimizer`.
  - `max_symbols` задаёт количество наиболее ликвидных торговых пар, которые бот выберет из доступных.
  - `secondary_timeframe` определяет дополнительный интервал (по умолчанию `2h`). Свечи этого таймфрейма бот запрашивает напрямую у биржи и не агрегирует из основного. См. `DataHandler.load_initial` и `_send_subscriptions`.
  - `max_subscriptions_per_connection` (или `ws_subscription_batch_size`) определяет, сколько символов подписывается через одно WebSocket‑соединение. Если `ws_subscription_batch_size` не указан, он приравнивается к `max_subscriptions_per_connection` (по умолчанию 15).
  - `backup_ws_urls` задаёт список альтернативных WebSocket‑адресов. При переподключении `DataHandler` перебирает их по очереди, что позволяет использовать другие дата‑центры.
  - `ws_inactivity_timeout` определяет, сколько секунд ждать сообщений после отправки ping. Если данных нет дольше этого времени, соединение закрывается и открывается заново.
  - `history_batch_size` задаёт число одновременных запросов истории. При нехватке памяти значение автоматически снижается.
  - `history_retention` ограничивает объём сохранённых данных после инициализации. Хранятся только последние `N` баров.
  - `ray_num_cpus` задаёт число потоков, которые Ray выделяет под задачи (по умолчанию 2). Убедитесь, что у хоста достаточно ядер или уменьшите значение.
  - `telegram_queue_size` ограничивает размер очереди сообщений Telegram.
  - `fine_tune_epochs` задаёт число эпох при дообучении модели.
  - `early_stopping_patience` задаёт, сколько эпох ждать улучшения ошибки на
    валидации перед остановкой обучения.
  - `n_splits` определяет число разбиений временного ряда для кросс-валидации при обучении и оптимизации (по умолчанию 3).
  - `performance_window` задаёт размер окна для оценки точности. Если она падает ниже `retrain_threshold`, модель переобучается.
  - `freeze_base_layers` при дообучении замораживает начальные слои и ускоряет обучение.
  - `use_transfer_learning` включает режим дообучения существующей модели вместо полного переобучения.
  - `model_type` задаёт архитектуру нейросети (`gru`, `tft` или `transformer`).
    Новый вариант `transformer` использует слои самовнимания и позиционное кодирование.
    Он лучше выявляет долгосрочные зависимости в финансовых рядах. При большой истории
    увеличьте `d_model` и `num_layers` в `model_builder.py`, а для малых выборок
    оставьте значения по умолчанию, чтобы избежать переобучения.
  - После расчёта индикаторов `DataHandler` кэширует признаки в памяти. Если
    установить его параметр `feature_callback`, например `ModelBuilder.precompute_features`,
    признаки заранее сохраняются в `feature_cache`. Это снижает задержки при
    обучении и прогнозах, поскольку готовые данные не приходится вычислять
    заново.
Пример фрагмента `config.json` с этими параметрами:

```json
{
    "n_splits": 10,
    "performance_window": 200,
    "retrain_threshold": 0.6,
    "freeze_base_layers": true,
    "early_stopping_patience": 5
}
```

Кросс-валидация повышает устойчивость оценки качества, автоматическое переобучение поддерживает модель актуальной,
ранняя остановка и заморозка базовых слоёв сокращают переобучение и ускоряют тренировку.

4. Запустите бота. При локальном запуске без Docker Compose задайте адреса сервисов переменными `DATA_HANDLER_URL`, `MODEL_BUILDER_URL` и `TRADE_MANAGER_URL`:
```bash
DATA_HANDLER_URL=http://localhost:8000 \
MODEL_BUILDER_URL=http://localhost:8001 \
TRADE_MANAGER_URL=http://localhost:8002 \
python -m bot.trading_bot
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
DOCKERFILE=Dockerfile docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
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

The services may build large data sets in memory. Compose allocates
8GB of shared memory for each container via `shm_size: '8gb'` so Polars
and other libraries do not run into the default `/dev/shm` limit.

The `model_builder` service sets `TF_CPP_MIN_LOG_LEVEL=3` to hide verbose TensorFlow
GPU warnings. Set the environment variable to `2` or `3` **before importing
TensorFlow** if you want to suppress extra CUDA messages entirely. Adjust or
remove this variable if you need more detailed logs.

The Docker image also skips importing TensorFlow during its build-time library
check unless explicitly enabled. Set `ENABLE_TF=1` as a build argument when
you require TensorFlow features:

```bash
docker build --build-arg ENABLE_TF=1 -t trading-bot-x .
```

Omit the argument to skip the import and avoid the associated warnings.

When both TensorFlow and PyTorch start in the same container you might see
messages like `Unable to register cuDNN factory` or `computation placer already
registered`. These lines appear while each framework loads CUDA plugins and
tries to register them more than once. They are warnings, not fatal errors, and
can be safely ignored. Building the image with `Dockerfile.cpu` avoids them
entirely.

## Lightweight service scripts

The `services` directory provides minimal versions of the microservices. They
avoid heavy dependencies so they start quickly and are useful for basic
testing.

Run each script directly from the project root:

```bash
gunicorn services.data_handler_service:app
python services/model_builder_service.py
python services/trade_manager_service.py
```

Для альтернативы без зависимостей можно запустить сервис через Waitress:

```bash
waitress-serve services.data_handler_service:app
```

`data_handler_service.py` fetches prices from Bybit using `ccxt` and exposes
`/price/<symbol>` and `/ohlcv/<symbol>`.
`model_builder_service.py` trains a small logistic regression when you POST
features to `/train`.  Predictions are requested via `/predict` using
`{"features": [...]}` where the first element usually represents the price.
For backward compatibility a single `price` value is also accepted.  The
service supports multi-class problems via
`LogisticRegression(multi_class="auto")` and returns an error if the labels
contain only a single class.
`trade_manager_service.py` opens and closes positions on Bybit via
`/open_position` and `/close_position` and also provides `/positions`, `/ping`
and `/ready` routes. The `/open_position` endpoint accepts either `amount` or
`price`, calculating the size from `TRADE_RISK_USD` when only a price is given.

`trade_manager_service.py` uses a simple token-based authentication. Set the
`TRADE_MANAGER_TOKEN` environment variable on the service and supply the same
token via an `Authorization: Bearer` header when calling any POST route or the
`/positions` endpoint.  For example:

```bash
export TRADE_MANAGER_TOKEN=supersecret
curl --netrc-file /dev/null -H "Authorization: Bearer $TRADE_MANAGER_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"symbol":"BTCUSDT","side":"buy","amount":1}' \
     http://localhost:8002/open_position
```

These reference scripts expose the same HTTP routes as the full services but
avoid heavy frameworks like TensorFlow and PyTorch, making them ideal for quick
tests.

Все Flask-сервисы ограничивают размер тела запроса 1 МБ. При превышении
лимита клиент получает ответ 413 с JSON‑сообщением `{"error": "payload too large"}`.

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
  command: gunicorn services.data_handler_service:app
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
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build    # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build    # GPU
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
RUNTIME= DOCKERFILE=Dockerfile.cpu docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
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
   docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm data_handler nvidia-smi
   ```

   If no GPU is detected, rebuild with the CPU Dockerfile:

   ```bash
   RUNTIME= DOCKERFILE=Dockerfile.cpu docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
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
# Optionally cache features ahead of time
# data_handler = DataHandler(cfg, bot, chat_id,
#                            feature_callback=model_builder.precompute_features)
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

Количество подписок через одно соединение ограничивается параметром `max_subscriptions_per_connection` (он же `ws_subscription_batch_size`). Если `ws_subscription_batch_size` не указан, используется `max_subscriptions_per_connection`. Если список пар превышает это значение, бот откроет дополнительные WebSocket‑соединения.

Подписки отправляются пакетами. Размер пакета определяется этим же параметром. При необходимости бот делает паузу между отправками, чтобы не превышать ограничения биржи на частоту запросов.

Пример настроек в `config.json`:

```json
{
    "max_subscriptions_per_connection": 15,
    "history_batch_size": 10,
    "history_retention": 200
}
```

Так бот будет разбивать список подписок на пакеты по 10–15 символов и делать небольшую паузу между отправками. Для большинства бирж этого достаточно, чтобы не превышать ограничения на частоту запросов.

Если сервер перестаёт присылать данные, `DataHandler` отправляет ping и ждёт `ws_inactivity_timeout` секунд. После тайм-аута соединение закрывается и бот пытается подключиться заново. В списке `backup_ws_urls` можно указать резервные WebSocket‑адреса — например, другие дата‑центры — которые будут пробоваться по очереди.

## Model training

`ModelBuilder` автоматически обучает и переобучает нейросетевую модель на базе Transformer.
Для организации процесса можно выбрать ML-фреймворк: `pytorch` (по умолчанию), `lightning` или `keras`.
Тип указывается параметром `nn_framework` в `config.json`.
Для каждого актива формируются признаки из OHLCV‑данных и технических
индикаторов. Метка `1` присваивается, если через
`lstm_timesteps` баров цена выросла больше, чем на значение
`target_change_threshold` (по умолчанию `0.001`). Иначе метка равна `0`.
При установке `prediction_target` в `"pnl"` модель обучается прогнозировать
доходность следующего бара, и сигнал формируется, только если предсказанный
PnL превышает торговые издержки.

Обучение запускается удалённо через `ray` и использует GPU при наличии.
- После обучения выполняется калибровка вероятностей с помощью логистической
регрессии. Затем `TradeManager` применяет откалиброванные прогнозы для
открытия и закрытия позиций согласно динамическим порогам.

Для интерпретации модели вычисляются SHAP‑значения, и краткий отчёт о наиболее
важных признаках отправляется в Telegram.

При включении `use_transfer_learning` повторные запуски обучения не создают модель с нуля,
а дообучают её на последних данных. Число эпох задаётся параметром `fine_tune_epochs`.
Модель переобучается, если точность за последние `performance_window` прогнозов опускается ниже `retrain_threshold`.
Обучение использует `n_splits` разбиений `TimeSeriesSplit`. Параметр `early_stopping_patience` прерывает цикл,
когда ошибка на валидации не улучшается, а `freeze_base_layers` помогает сохранить базовые представления при дообучении.


Пример запуска обучения вместе с ботом:

```bash
python -m bot.trading_bot
```

### Transformer architecture

Архитектура `transformer` использует позиционное кодирование и механизмы
самовнимания вместо рекуррентных слоёв. Это позволяет эффективнее улавливать
долгосрочные зависимости в финансовых временных рядах и повышает стабильность
прогнозов на волатильных рынках. Включите её, указав в `config.json`:

```json
{
    "model_type": "transformer"
}
```

При увеличении объёма обучающих данных имеет смысл повысить ёмкость модели,
поменяв параметры `d_model` и `num_layers` в `model_builder.py`. Небольшие
истории наоборот лучше обучать на минимальных значениях, чтобы не потерять
обобщающую способность.

### RL agents

Модуль `RLAgent` может обучать модели с помощью `stable-baselines3`, `Ray RLlib` или фреймворка `Catalyst` от Яндекса.
Выберите подходящий движок параметром `rl_framework` в `config.json` (`stable_baselines3`, `rllib` или `catalyst`).
Алгоритм указывается опцией `rl_model` (`PPO` или `DQN`), продолжительность обучения — `rl_timesteps`.
Если `rl_use_imitation` выставлен в `true`, перед обучением PPO/DQN политика
предобучается на действиях, полученных от базовой модели. Такой этап имитации
помогает быстрее сходиться модели, особенно когда исторических данных
недостаточно: политика сразу видит приближённые примеры и реже делает случайные
шаги.
Для корректной работы `stable-baselines3` необходим пакет `gymnasium`. Он
устанавливается по умолчанию вместе с зависимостями проекта, поэтому
альтернативный `gym` не требуется.
Параметр `drawdown_penalty` задаёт штраф за просадку капитала в среде RL:
из вознаграждения вычитается `коэффициент × текущая просадка`. Такое наказание
позволяет уменьшить глубокие просадки баланса и делает торговлю более
устойчивой.

Пример настроек для активации этих возможностей в `config.json`:

```json
{
    "rl_use_imitation": true,
    "drawdown_penalty": 0.1
}
```

При включённой имитации и разумном штрафе агент обычно быстрее сходится и
склонен избегать резких потерь капитала. Для работы `stable-baselines3` нужен
пакет `gymnasium`, который ставится вместе с зависимостями проекта.

Периодическое переобучение задаётся параметром `retrain_interval` в
`config.json`. Можно запускать бота по расписанию (например, через `cron`) для
регулярного обновления моделей:

```bash
0 3 * * * cd /path/to/bot && /usr/bin/python -m bot.trading_bot
```

## Signal ranking

During live trading the bot can rank entry signals from all symbols once per
`check_interval`. Signals are sorted by the predicted probability multiplied by
the expected ATR. Only the best `top_signals` trades are executed. Set
`top_signals` in `config.json` (defaults to `max_positions`).

## MLflow

Если установить пакет `mlflow` и включить флаг `mlflow_enabled` в `config.json`,
бот будет сохранять параметры и метрики обучения в указанное хранилище.
По умолчанию используется локальная папка `mlruns`.
Чтобы задать другой адрес, укажите ключ `mlflow_tracking_uri` в `config.json`
или переменную окружения `MLFLOW_TRACKING_URI`.

Пример запуска с отслеживанием экспериментов:

```bash
MLFLOW_TRACKING_URI=mlruns python -m bot.trading_bot
```

## Running tests

Running `pytest` requires the packages listed in `requirements.txt`.
Install them using the helper script:

```bash
./scripts/setup-tests.sh        # CPU packages only
```

To install the GPU-enabled packages from `requirements-gpu.txt` as well, run:

```bash
./scripts/install-test-deps.sh --gpu
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

Integration tests that require external services are marked with `integration` and are skipped by default. To run them separately:

```bash
pytest -m integration
```

The `requirements.txt` file already bundles `pytest` and all other
packages needed by the test suite.

Unit tests automatically set the environment variable `TEST_MODE=1`.
This disables the Telegram logger's background worker thread so tests
run without spawning extra threads.

When `TEST_MODE=1` is exported manually, heavy dependencies like Ray,
stable-baselines3 and the Bybit SDK are replaced with lightweight stubs.
`DataHandler` also generates synthetic OHLCV data instead of calling
external APIs. This mode is useful for running examples on systems
without the full requirements installed.

Set `DATA_HANDLER_PROFILE=1` to log how long `DataHandler` methods take.
Profiling information is printed to the standard log output.
For example:

```bash
DATA_HANDLER_PROFILE=1 python -m bot.trading_bot
```

If `scikit-learn` is not installed, tests marked with `requires_sklearn`
are skipped automatically. Install the package to run the full suite.

As noted above, make sure to run `./scripts/setup-tests.sh` before
executing `pytest`; otherwise imports such as `numpy`, `pandas`, `scipy` and
`httpx` will fail.

Чтобы выполнить тесты на машине без GPU, создайте виртуальное окружение и установите зависимости из `requirements.txt`.
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
pip install -r requirements.txt -r requirements-gpu.txt  # or requirements.txt
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

## State persistence

`TradeManager` now saves its open positions as a Parquet file and per-symbol
returns as JSON inside the directory specified by `cache_dir`. The files are
named `trade_manager_state.parquet` and `trade_manager_returns.json`
respectively. These non-executable formats replace the previous pickle-based
storage.

## Historical simulator

Replay past market data using the built in trading logic:

```bash
python scripts/run_simulation.py --start 2020-01-01 --end 2020-01-02 --speed 60
```

If you do not have the full GPU requirements installed, set `TEST_MODE=1` so
the simulator stubs heavy dependencies. Example:

```bash
TEST_MODE=1 python scripts/run_simulation.py --start ... --end ...
```

The simulator loads candles from the DataHandler cache and steps through them
rapidly while reusing ``TradeManager`` methods. Trailing stops and other
position logic behave exactly like in live trading.
If no cached candles exist for the chosen period, the simulator logs a warning
and exits.

## Continuous integration

All pushes and pull requests trigger a GitHub Actions workflow that installs
dependencies via `scripts/install-test-deps.sh` (which also installs `flake8`),
runs `python -m flake8`, and executes `pytest`.
This ensures style checks and tests run automatically.

## Security

Для снижения риска уязвимости [CVE-2024-56433](https://ubuntu.com/security/CVE-2024-56433)
проверьте, что диапазоны в `/etc/subuid` и `/etc/subgid` не пересекаются с UID
пользователей вашей сети. В репозитории добавлен скрипт
`scripts/check_subuid_conflict.sh`, который помогает выявить конфликтующие
диапазоны и предлагает безопасные значения.

Пример запуска:

```bash
./scripts/check_subuid_conflict.sh
```
