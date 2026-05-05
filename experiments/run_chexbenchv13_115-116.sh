#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[115] medrax_chexbench_v13 使用 CheXBench 专用 tag ontology 做 retrieve，0.99 阈值，CheXbench memory 模式 — 全量记忆写入到 memory_chexbench_115.jsonl"
python experiments/benchmark_medrax_chexbench_v13.py \
  --mode memory \
  --data-path "/data/lmy/datasets/chexbench/chexbench_data.json" \
  --image-base "/data/lmy/datasets" \
  --memory-db "experiments/memory/memory_chexbench_115.jsonl" \
  --memory-update on \
  --embedding-model bge-small-en-v1.5 \
  --top-k 3 \
  --sim-threshold 0.99 \
  --image-source local \
  --tools all \
  --log-tag 115

echo "[115] Done."

echo "[116] medrax_chexbench_v13 使用 CheXBench 专用 tag ontology 做 retrieve，0.95 阈值，CheXbench memory 模式 — 全量记忆写入到 memory_chexbench_116.jsonl"
python experiments/benchmark_medrax_chexbench_v13.py \
  --mode memory \
  --data-path "/data/lmy/datasets/chexbench/chexbench_data.json" \
  --image-base "/data/lmy/datasets" \
  --memory-db "experiments/memory/memory_chexbench_116.jsonl" \
  --memory-update on \
  --embedding-model bge-small-en-v1.5 \
  --top-k 3 \
  --sim-threshold 0.95 \
  --image-source local \
  --tools all \
  --log-tag 116

echo "[116] Done."
echo ""
