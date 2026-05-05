#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[180 GPT-5.5] ablation study k=3，sim=0.99 v11v2_API backend, 100% memory mode, all tools, default 32B LLM core"
python experiments/benchmark_medrax_v11v2_API_GPT-5.5_flex.py \
  --mode memory \
  --metadata-path chestagentbench/metadata.jsonl \
  --memory-db experiments/memory/memory_v11v2_180_api_GPT-5.5.jsonl \
  --memory-update on \
  --embedding-model bge-small-en-v1.5 \
  --top-k 3 \
  --sim-threshold 0.99 \
  --image-source local \
  --tools all \
  --memory-case-ratio 1 \
  --tool-backend api \
  --tool-api-base-url http://127.0.0.1:8010 \
  --log-tag 180_GPT-5.5

echo "Done."
