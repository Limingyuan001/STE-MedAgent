"""
Split metadata.jsonl into 5 folds by case (front-to-back),
then produce memory_84_{1..5}.jsonl by removing that fold's question IDs
from memory_v10.jsonl.
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repo root
METADATA = ROOT / "chestagentbench" / "metadata.jsonl"
MEMORY_SRC = ROOT / "experiments" / "memory" / "memory_v10_all_84.jsonl"
OUT_DIR = Path(__file__).parent

# ── 1. Load metadata, preserving insertion order of cases ──────────────────
case_to_records: dict[str, list[dict]] = {}
with METADATA.open() as f:
    for line in f:
        rec = json.loads(line)
        case_to_records.setdefault(rec["case_id"], []).append(rec)

case_ids = list(case_to_records.keys())          # ordered as they appear
n_cases  = len(case_ids)                          # 609
n_folds  = 5
base, extra = divmod(n_cases, n_folds)            # 121 r 4  →  folds 1-4: 122, fold 5: 121

# ── 2. Compute fold slices ─────────────────────────────────────────────────
fold_case_ids: list[list[str]] = []
start = 0
for i in range(n_folds):
    size = base + (1 if i < extra else 0)
    fold_case_ids.append(case_ids[start : start + size])
    start += size

# ── 3. Load memory source ──────────────────────────────────────────────────
with MEMORY_SRC.open() as f:
    memory_records = [json.loads(line) for line in f]

# ── 4. Generate files for each fold ───────────────────────────────────────
for fold_idx, fold_cases in enumerate(fold_case_ids, start=1):
    fold_qids: set[str] = set()
    fold_records: list[dict] = []
    for cid in fold_cases:
        for rec in case_to_records[cid]:
            fold_qids.add(rec["question_id"])
            fold_records.append(rec)

    # metadata_val{i}.jsonl
    meta_path = OUT_DIR / f"metadata_val{fold_idx}.jsonl"
    with meta_path.open("w") as f:
        for rec in fold_records:
            f.write(json.dumps(rec) + "\n")

    # memory_84_{i}.jsonl  — exclude this fold's question IDs
    mem_path = OUT_DIR / f"memory_84_{fold_idx}.jsonl"
    kept = [r for r in memory_records if r.get("question_id") not in fold_qids]
    with mem_path.open("w") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")

    print(
        f"Fold {fold_idx}: {len(fold_cases)} cases, "
        f"{len(fold_records)} questions, "
        f"{len(kept)}/{len(memory_records)} memory records kept"
    )

print("\nDone. Files written to:", OUT_DIR)
