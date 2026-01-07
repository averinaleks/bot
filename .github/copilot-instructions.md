# Copilot instructions for Trading Bot

## Big picture architecture
- The system is split into **DataHandler**, **ModelBuilder**, and **TradeManager** services that communicate over HTTP/JSON; `run_bot.py` can wire them together in-process for a monolithic run. Key entry points: `services/data_handler_service.py`, `services/model_builder_service.py`, and `bot/trade_manager/core.py`. Refer to `ARCHITECTURE.md` for the interaction diagram and data flow.
- Market data flows from DataHandler (REST + ccxt/WebSocket) → TradeManager → ModelBuilder `/train`/`/predict`, with optional GPT signals via `bot/gpt_client.py` (expects JSON like `{signal,tp_mult,sl_mult}`).
- Trade execution and notifications live in TradeManager, using `ccxt` for exchange REST/WebSocket and `TelegramLogger` for alerts.

## Critical workflows & commands
- Quick offline run (no secrets required): `python run_bot.py --offline` (auto-offline is enabled by default when env vars are missing).
- Tests: `./scripts/setup-tests.sh` (CPU) or `./scripts/install-test-deps.sh --gpu`, then `pytest`. Integration tests are marked `-m integration`.
- Linting: `python -m flake8` (also run via pre-commit hooks).

## Project-specific conventions
- Config is primarily in `config.json` (see `config.example.json`), with `.env` loaded via `bot/dotenv_utils.py`. Service factories in `config.json` decide real vs offline implementations; when missing, `run_bot.py` prompts to use offline mode.
- `TEST_MODE=1` replaces heavy deps (Ray, RL libs, Bybit SDK) with lightweight stubs and disables Telegram background threads via `tests/conftest.py`.
- GPT calls are guarded by host allow-lists (`GPT_OSS_ALLOWED_HOSTS`) and strict JSON validation in `bot/gpt_client.py`.

## Integration points & data handling
- DataHandler endpoints (`/price`, `/history`, `/ping`) are defined in `services/data_handler_service.py` and backed by `data_handler/` caching logic.
- ModelBuilder persists models via `joblib` and signs state (see `services/model_builder_service.py` and `model_builder/`).
- TradeManager persists state in `cache_dir` as `trade_manager_state.parquet` and `trade_manager_returns.json`.

## Patterns to follow
- Prefer `bot/http_client.py` helpers for service HTTP calls.
- Respect offline stubs in `services/offline.py` when `OFFLINE_MODE=1` or `--offline` is set.
- When touching trading logic, start in `bot/trade_manager/core.py` and follow calls into `trade_manager/` submodules.
