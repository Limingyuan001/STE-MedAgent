#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"


echo "[119 SLAKE_VQA_32B_inference] v12_SLAKE_VQA_API backend, pure inference mode, all tools, default 32B LLM core"
python experiments/benchmark_medrax_v12_SLAKE_VQA_API.py \
  --mode inference \
  --metadata-path experiments/M4CXR/SLAKE_VQA/slake_test_en_closed_xray_114.json \
  --slake-image-root /data/lmy/datasets/slake/imgs \
  --memory-db experiments/memory/memory_missing5_tmp.jsonl \
  --memory-update off \
  --embedding-model bge-small-en-v1.5 \
  --top-k 3 \
  --sim-threshold 2 \
  --image-source local \
  --tools all \
  --tool-backend api \
  --tool-api-base-url http://127.0.0.1:8010 \
  --log-tag 119_32B

echo "[120 SLAKE_VQA_32B_memory] v12_SLAKE_VQA_API backend, full memory mode, all tools, default 32B LLM core"
python experiments/benchmark_medrax_v12_SLAKE_VQA_API.py \
  --mode memory \
  --metadata-path experiments/M4CXR/SLAKE_VQA/slake_test_en_closed_xray_114.json \
  --slake-image-root /data/lmy/datasets/slake/imgs \
  --memory-db experiments/memory/memory_v12_120_32B.jsonl \
  --restore on \
  --memory-update on \
  --embedding-model bge-small-en-v1.5 \
  --top-k 3 \
  --sim-threshold 0.99 \
  --image-source local \
  --tools all \
  --memory-case-ratio 1 \
  --tool-backend api \
  --tool-api-base-url http://127.0.0.1:8010 \
  --log-tag 120_32B

echo "Done."
