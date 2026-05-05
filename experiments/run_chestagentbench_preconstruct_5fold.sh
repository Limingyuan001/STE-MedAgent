#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FOLD_DIR="experiments/memory/pre-constructed scheme five-verified_10"

for FOLD in 1 2 3 4 5; do
  echo "[pre-constructed fold ${FOLD}/5] v11v2_API backend, read-only memory, all tools, 32B"
  python experiments/benchmark_medrax_v11v2_API.py \
    --mode inference \
    --metadata-path "${FOLD_DIR}/metadata_val${FOLD}.jsonl" \
    --memory-db "${FOLD_DIR}/memory_84_${FOLD}.jsonl" \
    --memory-update off \
    --embedding-model bge-small-en-v1.5 \
    --top-k 3 \
    --sim-threshold 0.99 \
    --image-source local \
    --tools all \
    --tool-backend api \
    --tool-api-base-url http://127.0.0.1:8010 \
    --log-tag "preconstruct_fold${FOLD}_32B"
done

echo "All 5 folds done."
