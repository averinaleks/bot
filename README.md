# Trading Bot

Этот репозиторий содержит пример торгового бота на Python. Для запуска необходимы файлы `config.json` и `.env` с ключами API.


**Disclaimer**: This project is provided for educational purposes only and does not constitute financial advice. Use at your own risk.

## Быстрый старт

1. Установите зависимости. Варианты для GPU и CPU:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   # Вариант с GPU (по умолчанию)
   pip install -r requirements.txt
   # Или CPU‑сборки без CUDA
   pip install -r requirements-cpu.txt
   ```
   Список `requirements-cpu.txt` содержит версии `torch` и `tensorflow` без поддержки GPU. Его можно использовать для установки зависимостей и запуска тестов на машинах без CUDA.
- После обновления зависимостей пакет `optuna-integration[botorch]` больше не используется.
- Библиотека `catalyst` закреплена на версии `21.4`, так как новые версии не устанавливаются с `pip>=24.1`. Если требуется `catalyst>=22.2`, понизьте `pip` ниже 24.1.
2. Создайте файл `.env` по примеру `.env.example` и укажите свои значения.
   Переменные из этого файла загружаются автоматически библиотекой
   `python-dotenv`, дополнительный вызов `load_dotenv()` не требуется.
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
4. Запустите бота:
  ```bash
  python trading_bot.py
  ```
  При старте бот проверяет доступность всех сервисов по маршруту `/ping`.
  Параметры проверки контролируются переменными окружения
  `SERVICE_CHECK_RETRIES` и `SERVICE_CHECK_DELAY` (в секундах).
  Если в логах появляется сообщение "dependent services are unavailable",
  увеличьте эти значения в `.env`. Например:

  ```env
  SERVICE_CHECK_RETRIES=20
  SERVICE_CHECK_DELAY=5
  ```

Также можно использовать `docker-compose up --build` для запуска в контейнере.
По умолчанию используется образ с поддержкой GPU. Если она не требуется,
запустите compose с переменной `DOCKERFILE` и отключите NVIDIA-переменные:

```bash
DOCKERFILE=Dockerfile.cpu NVIDIA_VISIBLE_DEVICES= NVIDIA_DRIVER_CAPABILITIES= docker-compose up --build
```

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

To avoid processing old updates after a restart, store the `update_id` and pass
it to the `offset` parameter when calling `get_updates`. The helper class
`TelegramUpdateListener` handles this automatically and logs any errors.

Telegram enforces message limits per bot account, so duplicate notifications are
typically caused by bugs rather than global restrictions. Ensure that your code
filters repeated messages and checks that `send_message` returns HTTP 200.

You can run this bot either with long polling or a webhook using the
`Application` class from `python-telegram-bot`.

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

Пример запуска с отслеживанием экспериментов:

```bash
MLFLOW_TRACKING_URI=mlruns python trading_bot.py
```

## Running tests

Чтобы выполнить тесты на машине без GPU, создайте виртуальное окружение и установите зависимости из `requirements-cpu.txt`.
Пакет включает CPU‑сборки `torch` и `tensorflow`, поэтому тесты не подтягивают CUDA‑библиотеки.

Пример полного процесса:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-cpu.txt
pytest
```

Если у вас есть GPU и установленный CUDA, можно установить полный список зависимостей из `requirements.txt` и затем запустить те же тесты.

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

The project uses **flake8** for style checks. Install dependencies and set up an optional Git pre-commit hook so linting runs automatically:

```bash
pip install -r requirements.txt  # or requirements-cpu.txt
flake8 --install-hook git
```

Linting configuration is stored in `.flake8`. Run the checker manually:

```bash
flake8
```
