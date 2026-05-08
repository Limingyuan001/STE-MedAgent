#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
MODEL_REF="${MODEL_REF:-Qwen/Qwen3-VL-32B-Instruct}"
MODEL_DIR="${MODEL_DIR:-}"
LOG_DIR="${LOG_DIR:-${BASE_DIR}/logs}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-32b-instruct}"
TP="${TP:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
API_KEY="${API_KEY:-mingyuan}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-true}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
DISABLE_FRONTEND_MULTIPROCESSING="${DISABLE_FRONTEND_MULTIPROCESSING:-false}"
QUANTIZATION="${QUANTIZATION:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
COMPILATION_CONFIG="${COMPILATION_CONFIG:-}"

if [[ -z "${CUDA_VISIBLE_DEVICES-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0,1"
fi

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/qwen3_vl_32b_$(date +%Y%m%d_%H%M%S).log"

MODEL_SOURCE="${MODEL_DIR:-${MODEL_REF}}"
if [[ -n "${MODEL_DIR}" && ! -d "${MODEL_DIR}" ]]; then
  echo "Model directory not found: ${MODEL_DIR}" >&2
  exit 1
fi

if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "conda.sh not found under ${CONDA_BASE}. Set CONDA_BASE or activate VLM manually." >&2
  exit 1
fi

conda activate VLM

if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "0" || "${ENABLE_AUTO_TOOL_CHOICE}" == "false" || "${ENABLE_AUTO_TOOL_CHOICE}" == "False" || "${ENABLE_AUTO_TOOL_CHOICE}" == "FALSE" ]]; then
  AUTO_TOOL_FLAG="--no-enable-auto-tool-choice"
else
  AUTO_TOOL_FLAG="--enable-auto-tool-choice"
fi

EXTRA_ARGS=()
if [[ "${ENFORCE_EAGER}" == "1" || "${ENFORCE_EAGER}" == "true" || "${ENFORCE_EAGER}" == "True" || "${ENFORCE_EAGER}" == "TRUE" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi
if [[ "${DISABLE_FRONTEND_MULTIPROCESSING}" == "1" || "${DISABLE_FRONTEND_MULTIPROCESSING}" == "true" || "${DISABLE_FRONTEND_MULTIPROCESSING}" == "True" || "${DISABLE_FRONTEND_MULTIPROCESSING}" == "TRUE" ]]; then
  EXTRA_ARGS+=(--disable-frontend-multiprocessing)
fi
if [[ -n "${QUANTIZATION}" ]]; then
  EXTRA_ARGS+=(--quantization "${QUANTIZATION}")
fi
if [[ -n "${MAX_NUM_SEQS}" ]]; then
  EXTRA_ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
fi
if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
  EXTRA_ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
fi
if [[ -n "${COMPILATION_CONFIG}" ]]; then
  EXTRA_ARGS+=(--compilation-config "${COMPILATION_CONFIG}")
fi

echo "Starting Qwen3-VL-32B vLLM server..."
echo "Model: ${MODEL_SOURCE}"
echo "Host: ${HOST}  Port: ${PORT}  TP: ${TP}  DType: ${DTYPE}"
echo "Log: ${LOG_FILE}"

exec vllm serve "${MODEL_SOURCE}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --api-key "${API_KEY}" \
  "${AUTO_TOOL_FLAG}" \
  --tool-call-parser "${TOOL_CALL_PARSER}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"
