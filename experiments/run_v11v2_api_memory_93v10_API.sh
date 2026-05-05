#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[93v10_API] v11v2_API backend, 100% memory mode, all tools"
python experiments/benchmark_medrax_v11v2_API.py \
  --mode memory \
  --metadata-path chestagentbench/metadata.jsonl \
  --memory-db experiments/memory/memory_v11v2_93v10_api.jsonl \
  --memory-update on \
  --embedding-model bge-small-en-v1.5 \
  --top-k 3 \
  --sim-threshold 0.99 \
  --image-source local \
  --tools all \
  --memory-case-ratio 1 \
  --tool-backend api \
  --tool-api-base-url http://127.0.0.1:8010 \
  --log-tag 93v10_API

# To switch only the LLM core to the local 2B model, append:
#   --llm-core 2b
#
# Example:
# python experiments/benchmark_medrax_v11v2_API.py \
#   --mode memory \
#   --metadata-path chestagentbench/metadata.jsonl \
#   --memory-db experiments/memory/memory_v11v2_93v10_api.jsonl \
#   --memory-update on \
#   --embedding-model bge-small-en-v1.5 \
#   --top-k 3 \
#   --sim-threshold 0.99 \
#   --image-source local \
#   --tools all \
#   --memory-case-ratio 1 \
#   --tool-backend api \
#   --tool-api-base-url http://127.0.0.1:8010 \
#   --llm-core 2b \
#   --llm-2b-base-url http://127.0.0.1:8001/v1 \
#   --llm-2b-model qwen3-vl-2b-instruct-fp8 \
#   --llm-2b-api-key mingyuan \
#   --log-tag 93v10_API_2b

echo "Done."
