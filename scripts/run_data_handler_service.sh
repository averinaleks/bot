#!/usr/bin/env sh

# Запуск data_handler_service через Gunicorn.
# Передаёт все аргументы командной строки непосредственно Gunicorn.

exec gunicorn services.data_handler_service:app "$@"

