#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"

PID="$(lsof -tiTCP:${PORT} -sTCP:LISTEN || true)"
if [[ -z "${PID}" ]]; then
  echo "No process listening on port ${PORT}."
  exit 0
fi

echo "Stopping Qwen3-VL-32B vLLM (PID ${PID}) on port ${PORT}..."
kill "${PID}"

sleep 2
if kill -0 "${PID}" 2>/dev/null; then
  echo "Process still running; sending SIGKILL..."
  kill -9 "${PID}"
fi
