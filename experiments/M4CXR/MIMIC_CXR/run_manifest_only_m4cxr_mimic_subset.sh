#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/data/lmy/Projects/MedRAXlmy/worktrees/wt-b/.claude/worktrees/crazy-knuth"
SCRIPT_PATH="$ROOT_DIR/experiments/M4CXR/MIMIC_CXR/create_m4cxr_mimic_test_subset.py"

DATA_ROOT="/data/lmy/datasets/M4CXR_MIMIC"
SPLIT_CSV="$DATA_ROOT/mimic-cxr-2.0.0-split.csv.gz"
METADATA_CSV="$DATA_ROOT/mimic-cxr-2.0.0-metadata.csv.gz"
REPORTS_ROOT="$DATA_ROOT/mimic-cxr-reports"

python "$SCRIPT_PATH" \
  --split-csv "$SPLIT_CSV" \
  --metadata-csv "$METADATA_CSV" \
  --reports-root "$REPORTS_ROOT"
