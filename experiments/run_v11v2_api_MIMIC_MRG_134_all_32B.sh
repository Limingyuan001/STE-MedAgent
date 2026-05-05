#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_BASE="experiments/M4CXR/MIMIC_CXR/outputs"
OUT_134="${OUT_BASE}/134_all_32B_biovil_t"
MEMORY_DB="experiments/memory/memory_v11v2_mimic_134_all_32B_biovil_t.jsonl"

echo "[134_all_32B_biovil_t MIMIC_MRG_memory] v11v2_MIMIC_MRG_API concurrent backend, memory mode, BioViL-T image embedding, default MIMIC tools, default 32B LLM core, sample_concurrency=4"
python experiments/benchmark_medrax_v11v2_mimic_mrg_API_concurrent.py \
  --mode memory \
  --subset full \
  --memory-db "$MEMORY_DB" \
  --restore on \
  --memory-update on \
  --embedding-model biovil-t-image \
  --embedding-backend api \
  --embedding-api-base-url http://127.0.0.1:8011 \
  --top-k 3 \
  --sim-threshold 0.9 \
  --retrieval-enabled on \
  --tools all \
  --tool-backend api \
  --tool-api-base-url http://127.0.0.1:8010 \
  --sample-concurrency 4 \
  --eval-every 10 \
  --chexbert-backend api \
  --chexbert-api-base-url http://127.0.0.1:8011 \
  --output-dir "$OUT_134" \
  --log-tag 134_all_32B_biovil_t

python experiments/eval_mimic_mrg_f1.py \
  --predictions-path "${OUT_134}/predictions_eval.jsonl" \
  --output-json "${OUT_134}/mimic_mrg_f1_summary.json"

echo "Done."
