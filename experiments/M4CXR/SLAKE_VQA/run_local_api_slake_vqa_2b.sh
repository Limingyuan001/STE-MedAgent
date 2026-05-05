#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/benchmark_local_api_slake_vqa.py"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/outputs/slake_vqa_2b_${TIMESTAMP}"

conda run -n ToolAPI python "${PY_SCRIPT}" \
  --base-url "http://127.0.0.1:8001/v1" \
  --api-key "mingyuan" \
  --model "qwen3-vl-2b-instruct-fp8" \
  --output-dir "${DEFAULT_OUTPUT_DIR}" \
  "$@"
