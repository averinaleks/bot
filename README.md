# Trading Bot

Этот репозиторий содержит пример торгового бота на Python. Для запуска необходимы файлы `config.json` и `.env` с ключами API.

## Быстрый старт

1. Установите зависимости:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Создайте файл `.env` по примеру `.env.example` и укажите свои значения.
3. Отредактируйте `config.json` под свои нужды.
4. Запустите бота:
   ```bash
   python trading_bot.py
   ```

Также можно использовать `docker-compose up --build` для запуска в контейнере.
