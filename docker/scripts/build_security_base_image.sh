#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'USAGE'
Usage: build_security_base_image.sh [--tag <name>] [--push]

Скрипт собирает слой безопасности ``security-base`` из корневого Dockerfile.
По умолчанию образ публикуется локально с тегом ``bot-security-base:latest``.
Передаёте ``--push`` — buildx будет пушить в registry вместо локальной загрузки.
Устанавливайте переменную ``CACHE_DIR`` для переопределения директории кэша.
USAGE
}

TAG="bot-security-base:latest"
PUSH=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--tag)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 1
      fi
      TAG="$2"
      shift 2
      ;;
    --push)
      PUSH=true
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_help >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
CACHE_DIR=${CACHE_DIR:-"${PROJECT_ROOT}/.docker-cache/security-base"}
mkdir -p "${CACHE_DIR}"

BUILD_FLAGS=(
  --target security-base
  --cache-from "type=local,src=${CACHE_DIR}"
  --cache-to "type=local,dest=${CACHE_DIR},mode=max"
  --tag "${TAG}"
  --build-arg "SECURITY_BASE_IMAGE="
  --file Dockerfile
  "${PROJECT_ROOT}"
)

if ${PUSH}; then
  docker buildx build "${BUILD_FLAGS[@]}" --push
else
  docker buildx build "${BUILD_FLAGS[@]}" --load
fi
