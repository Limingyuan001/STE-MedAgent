#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/data/lmy/Projects/MedRAXlmy/worktrees/wt-b/.claude/worktrees/crazy-knuth"
SCRIPT_PATH="$ROOT_DIR/experiments/M4CXR/MIMIC_CXR/create_m4cxr_mimic_test_subset.py"

DATA_ROOT="/data/lmy/datasets/M4CXR_MIMIC"
SPLIT_CSV="$DATA_ROOT/mimic-cxr-2.0.0-split.csv.gz"
METADATA_CSV="$DATA_ROOT/mimic-cxr-2.0.0-metadata.csv.gz"
REPORTS_ROOT="$DATA_ROOT/mimic-cxr-reports"
MANIFEST_DIR="$DATA_ROOT/manifests"
JPG_RELPATHS="$MANIFEST_DIR/m4cxr_mimic_test_3858_jpg_relpaths.txt"
JPG_URLS="$MANIFEST_DIR/m4cxr_mimic_test_3858_jpg_urls.txt"
JPG_DEST="$DATA_ROOT/jpg_raw"
FAILED_LOG="$DATA_ROOT/logs/failed_jpg_parallel.txt"
DOWNLOAD_JOBS="${DOWNLOAD_JOBS:-16}"
PHYSIONET_USERNAME="${PHYSIONET_USERNAME:-}"
PHYSIONET_PASSWORD="${PHYSIONET_PASSWORD:-}"

echo "[1/2] Generate manifests only"
python "$SCRIPT_PATH" \
  --split-csv "$SPLIT_CSV" \
  --metadata-csv "$METADATA_CSV" \
  --reports-root "$REPORTS_ROOT"

echo "[2/2] Download the 3858 JPG images only"
if [[ -z "${PHYSIONET_USERNAME:-}" ]]; then
  echo "PHYSIONET_USERNAME is not set."
  echo "Example:"
  echo "  export PHYSIONET_USERNAME='your_physionet_username'"
  echo "  bash $0"
  exit 1
fi

if [[ -z "${PHYSIONET_PASSWORD:-}" ]]; then
  read -rsp "PhysioNet password: " PHYSIONET_PASSWORD
  echo
fi

mkdir -p "$JPG_DEST"
mkdir -p "$(dirname "$FAILED_LOG")"
: > "$FAILED_LOG"

WGET_AUTH_RC="$(mktemp)"
chmod 600 "$WGET_AUTH_RC"
cat > "$WGET_AUTH_RC" <<EOF
user = $PHYSIONET_USERNAME
password = $PHYSIONET_PASSWORD
continue = on
EOF
trap 'rm -f "$WGET_AUTH_RC"' EXIT

export JPG_DEST FAILED_LOG WGET_AUTH_RC

echo "Parallel jobs: $DOWNLOAD_JOBS"
echo "Downloads are resumable. Re-running this script will continue unfinished files."

paste "$JPG_RELPATHS" "$JPG_URLS" \
  | python -c 'import sys
for line in sys.stdin:
    rel, url = line.rstrip("\n").split("\t", 1)
    sys.stdout.write(rel + "\0" + url + "\0")' \
  | xargs -0 -n 2 -P "$DOWNLOAD_JOBS" bash -c '
      relpath="$1"
      url="$2"
      dest="$JPG_DEST/$relpath"
      mkdir -p "$(dirname "$dest")"
      if ! wget -q -c \
        --config="$WGET_AUTH_RC" \
        -O "$dest" \
        "$url"; then
        printf "%s\t%s\n" "$relpath" "$url" >> "$FAILED_LOG"
        exit 1
      fi
    ' _

if [[ -s "$FAILED_LOG" ]]; then
  echo "Some downloads failed. See: $FAILED_LOG"
  exit 1
fi

echo "Finished downloading JPG subset into $JPG_DEST"
