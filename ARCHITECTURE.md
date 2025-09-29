# Архитектура Trading Bot

## Обзор

Система торгового бота состоит из набора сервисов и библиотек, которые взаимодействуют между собой по HTTP и через локальные очереди задач. Основные компоненты:

- **DataHandler** — сервис подготовки и предоставления ценовых данных.
- **ModelBuilder** — сервис обучения и инференса ML-моделей.
- **TradeManager** — оркестратор торговых стратегий и операций.
- **GPT API** — внешний сервис генерации сигналов (через модуль `bot.gpt_client`).

Код всех сервисов можно запускать как отдельные Flask-приложения либо интегрировать через модуль `run_bot.py`, который связывает компоненты в единый пайплайн.

### Текстовая схема взаимодействия

```
[Биржа / Источники котировок]
          │ REST / WebSocket (через ccxt)
          ▼
   DataHandler (Flask, services/data_handler_service.py)
          │ HTTP JSON (GET /price, GET /history, ...)
          ▼
   Trading Bot / TradeManager (bot/bot/trade_manager/core.py)
          │            │
          │            ├── HTTP JSON → ModelBuilder (Flask, /train, /predict)
          │            │                ↳ Файловое хранилище модели (joblib)
          │            │
          │            └── HTTPS JSON → GPT API (внешний сервис /gpt)
          │
          └── Telegram API, брокерские REST API (через httpx, ccxt)
```

## DataHandler

- **Назначение:** агрегирует рыночные данные, кеширует исторические свечи и предоставляет REST-эндпойнты для получения котировок и OHLC-последовательностей. Основная логика расположена в `services/data_handler_service.py` и `data_handler/`.
- **Протоколы:** HTTP/JSON через Flask (`/price/<symbol>`, `/history/<symbol>`, `/ping`, `/health`), опционально WebSocket-потоки от биржи через библиотеку `ccxt`.
- **Интеграции:** проверка API-ключа (`X-API-KEY`), использование `HistoricalDataCache`, обмен данными с Bybit/другими биржами, экспортирует API для TradeManager и внешних клиентов.

## ModelBuilder

- **Назначение:** обучает и обслуживает модели машинного обучения (scikit-learn, PyTorch). Реализован во `services/model_builder_service.py` и `model_builder/`.
- **Протоколы:** HTTP/JSON поверх Flask. Основные маршруты — `POST /train` (принимает набор признаков и метки) и `POST /predict` (возвращает сигнал и вероятность). Реализует health-check `GET /ping`.
- **Интеграции:** сохраняет модели через `joblib` в локальное файловое хранилище с подписью состояния (`security.write_model_state_signature`), использует `numpy`/`pandas` для подготовки данных. TradeManager обращается к API через `httpx`.

## TradeManager

- **Назначение:** главный исполнитель торговой логики (`bot/bot/trade_manager/core.py`). Координирует запросы к DataHandler, ModelBuilder и внешним API, управляет позицией и рисками, рассылает уведомления в Telegram.
- **Протоколы:**
  - HTTP/JSON запросы к DataHandler и ModelBuilder (через `bot.http_client` или прямой `httpx`).
  - Взаимодействие с биржами через `ccxt` (REST/WebSocket).
  - Telegram уведомления через HTTPS запросы (`TelegramLogger`).
  - Внутренние вычисления запускаются через `ray` или локальные процессы для инференса модели.
- **Интеграции:** использует `services.exchange_provider.ExchangeProvider`, `services.offline` для офлайн-режима, подключает `bot.utils_loader.require_utils` для вспомогательных утилит.

## GPT API

- **Назначение:** предоставляет текстовый анализ и дополнительные торговые сигналы. Модуль `bot.gpt_client` управляет запросами и валидацией ответов.
- **Протоколы:** HTTP(S) JSON-запросы (по умолчанию `POST` на `GPT_OSS_API`). Клиент ограничивает адреса (проверка `GPT_OSS_ALLOWED_HOSTS`) и перезапускает запросы с экспоненциальным бэкоффом.
- **Интеграции:**
  - Используется TradeManager и ModelBuilder как `gpt_client_factory`.
  - Поддерживает офлайн-заглушку `services.offline.OfflineGPT` при `OFFLINE_MODE=1`.
  - Управляет безопасностью URL (проверка схемы, хостов, размера payload).

## Потоки данных

1. **Получение рыночных данных:** TradeManager вызывает `GET /price` или `GET /history` у DataHandler. DataHandler обновляет кеш, при необходимости дергает биржу через `ccxt`.
2. **Обучение модели:** TradeManager планирует обучение, отправляет `POST /train` в ModelBuilder, который подготовляет признаки и сохраняет модель.
3. **Прогноз:** при необходимости TradeManager отправляет `POST /predict` с текущими признаками. ModelBuilder возвращает вероятность и сигнал.
4. **Запрос к GPT:** когда стратегия требует пояснений, TradeManager вызывает `query_gpt_json_async` из `bot.gpt_client`, передавая текущий контекст. Ответ нормализуется до структуры `Signal`.
5. **Исполнение сделки:** TradeManager рассчитывает объём и размещает ордера через `ccxt` (REST) или внутренний симулятор. TelegramLogger уведомляет об открытии/закрытии позиций.

## Безопасность и конфигурация

- Все сервисы читают переменные окружения через `bot.dotenv_utils.load_dotenv()`; токены доступа передаются через `.env`.
- DataHandler и TradeManager поддерживают проверку API-ключей, а GPT-клиент ограничивает список доверенных хостов и поддерживает только HTTPS или локальные HTTP.
- Для локальной разработки доступны офлайн-заглушки (`services.offline`), чтобы не обращаться к внешним сервисам.

## Развёртывание

- Рекомендуемый способ — docker-compose конфигурации в `services/docker-compose.yml`, где каждый сервис запускается отдельным контейнером, общение идёт по HTTP.
- При монолитном запуске `run_bot.py` стартует необходимые службы в одном процессе, используя встроенные фабрики (`ModelBuilder`, `DataHandler`).

