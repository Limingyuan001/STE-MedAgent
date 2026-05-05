import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
import numpy as np
from sklearn.metrics import f1_score


TARGET_NAMES_14 = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]
OBS_5 = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]
OBS_5_INDICES = [TARGET_NAMES_14.index(name) for name in OBS_5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MIMIC-CXR findings generation predictions with CheXbert-derived F1 metrics."
    )
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--cache-jsonl", type=Path, default=None)
    parser.add_argument("--chexbert-backend", choices=["api", "local"], default="api")
    parser.add_argument("--chexbert-api-base-url", type=str, default="http://127.0.0.1:8011")
    parser.add_argument("--chexbert-api-timeout", type=float, default=300.0)
    parser.add_argument("--chexbert-device", type=str, default="cuda")
    return parser.parse_args()


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            cache_key = str(row.get("cache_key", "")).strip()
            if cache_key:
                cache[cache_key] = row
    return cache


def write_cache(path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key in sorted(cache.keys()):
            handle.write(json.dumps(cache[key], ensure_ascii=False) + "\n")


def get_row_instance_id(row: Dict[str, Any], index: int) -> str:
    candidates = [
        row.get("instance_id"),
        row.get("question_id"),
        (
            f"{row.get('study_id', '')}_{str(row.get('dicom_id', '')).replace('.jpg', '')}"
            if row.get("study_id") or row.get("dicom_id")
            else ""
        ),
    ]
    for value in candidates:
        text = str(value or "").strip()
        if text:
            return text
    return f"row_{index}"


def build_content_hash(reference: str, prediction: str) -> str:
    payload = json.dumps(
        {
            "reference_findings": str(reference or ""),
            "prediction_findings": str(prediction or ""),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cache_key(row: Dict[str, Any], index: int) -> str:
    instance_id = get_row_instance_id(row, index)
    content_hash = build_content_hash(
        str(row.get("reference_findings", "") or ""),
        str(row.get("prediction_findings", "") or ""),
    )
    return f"{instance_id}::{content_hash}"


def get_f1chexbert_scorer(device: str = "cuda"):
    try:
        from f1chexbert import F1CheXbert
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'f1chexbert'. Install it with `pip install f1chexbert` "
            "before running this evaluator."
        ) from exc
    return F1CheXbert(device=device)


def request_chexbert_labels_api(
    texts: List[str],
    api_base_url: str,
    api_timeout: float,
) -> List[List[int]]:
    client = httpx.Client(
        timeout=httpx.Timeout(api_timeout, connect=min(10.0, api_timeout))
    )
    url = api_base_url.rstrip("/") + "/tools/chexbert_labels"
    try:
        response = client.post(url, json={"texts": texts})
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"CheXbert API request failed with HTTP {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"CheXbert API request failed: {exc}") from exc
    finally:
        client.close()

    body = response.json()
    if not isinstance(body, dict) or "payload" not in body:
        raise ValueError("CheXbert API returned an invalid response")
    payload = body["payload"]
    if not isinstance(payload, dict) or "labels_14" not in payload:
        raise ValueError("CheXbert API payload is missing labels_14")
    labels_14 = payload["labels_14"]
    if not isinstance(labels_14, list):
        raise ValueError("CheXbert API labels_14 is not a list")
    return [[int(x) for x in row] for row in labels_14]


def label_texts(
    texts: List[str],
    chexbert_backend: str,
    chexbert_api_base_url: str,
    chexbert_api_timeout: float,
    chexbert_device: str,
) -> List[List[int]]:
    if not texts:
        return []
    if chexbert_backend == "api":
        return request_chexbert_labels_api(
            texts=texts,
            api_base_url=chexbert_api_base_url,
            api_timeout=chexbert_api_timeout,
        )
    scorer = get_f1chexbert_scorer(device=chexbert_device)
    return [list(map(int, scorer.get_label(str(text or "").strip()))) for text in texts]


def update_label_cache(
    rows: List[Dict[str, Any]],
    cache_path: Path,
    chexbert_backend: str = "api",
    chexbert_api_base_url: str = "http://127.0.0.1:8011",
    chexbert_api_timeout: float = 300.0,
    chexbert_device: str = "cuda",
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    cache = load_cache(cache_path)
    pending: List[Tuple[str, Dict[str, Any], int]] = []

    for index, row in enumerate(rows):
        cache_key = get_cache_key(row, index)
        if cache_key in cache:
            continue
        pending.append((cache_key, row, index))

    if not pending:
        return cache, 0

    prediction_texts = [str(row.get("prediction_findings", "") or "") for _, row, _ in pending]
    reference_texts = [str(row.get("reference_findings", "") or "") for _, row, _ in pending]
    prediction_labels = label_texts(
        prediction_texts,
        chexbert_backend=chexbert_backend,
        chexbert_api_base_url=chexbert_api_base_url,
        chexbert_api_timeout=chexbert_api_timeout,
        chexbert_device=chexbert_device,
    )
    reference_labels = label_texts(
        reference_texts,
        chexbert_backend=chexbert_backend,
        chexbert_api_base_url=chexbert_api_base_url,
        chexbert_api_timeout=chexbert_api_timeout,
        chexbert_device=chexbert_device,
    )

    for pending_index, (cache_key, row, index) in enumerate(pending):
        cache[cache_key] = {
            "cache_key": cache_key,
            "instance_id": get_row_instance_id(row, index),
            "prediction_findings": prediction_texts[pending_index],
            "reference_findings": reference_texts[pending_index],
            "prediction_labels_14": prediction_labels[pending_index],
            "reference_labels_14": reference_labels[pending_index],
        }

    write_cache(cache_path, cache)
    return cache, len(pending)


def build_label_arrays(
    rows: List[Dict[str, Any]],
    cache: Dict[str, Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    ref_labels: List[List[int]] = []
    pred_labels: List[List[int]] = []
    for index, row in enumerate(rows):
        cache_key = get_cache_key(row, index)
        cached = cache.get(cache_key)
        if cached is None:
            raise KeyError(f"Missing label cache entry for {cache_key}")
        ref_labels.append(list(map(int, cached["reference_labels_14"])))
        pred_labels.append(list(map(int, cached["prediction_labels_14"])))
    return np.asarray(ref_labels, dtype=np.int64), np.asarray(pred_labels, dtype=np.int64)


def compute_metrics_from_cache(
    rows: List[Dict[str, Any]],
    cache: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    if not rows:
        return {
            "mF1-14": 0.0,
            "MF1-14": 0.0,
            "mF1-5": 0.0,
            "MF1-5": 0.0,
        }

    ref_labels, pred_labels = build_label_arrays(rows, cache)
    ref_labels_5 = ref_labels[:, OBS_5_INDICES]
    pred_labels_5 = pred_labels[:, OBS_5_INDICES]
    return {
        "mF1-14": float(
            f1_score(ref_labels.reshape(-1), pred_labels.reshape(-1), average="micro", zero_division=0)
        ),
        "MF1-14": float(f1_score(ref_labels, pred_labels, average="macro", zero_division=0)),
        "mF1-5": float(
            f1_score(
                ref_labels_5.reshape(-1),
                pred_labels_5.reshape(-1),
                average="micro",
                zero_division=0,
            )
        ),
        "MF1-5": float(f1_score(ref_labels_5, pred_labels_5, average="macro", zero_division=0)),
    }


def summarize_labels() -> Dict[str, Any]:
    return {
        "fourteen_observations": TARGET_NAMES_14,
        "five_observations": OBS_5,
        "positive_only_binarization": True,
    }


def evaluate_predictions_incremental(
    rows: List[Dict[str, Any]],
    predictions_path: Path,
    output_json: Path,
    cache_jsonl: Path,
    chexbert_backend: str = "api",
    chexbert_api_base_url: str = "http://127.0.0.1:8011",
    chexbert_api_timeout: float = 300.0,
    chexbert_device: str = "cuda",
) -> Dict[str, Any]:
    cache, newly_evaluated = update_label_cache(
        rows=rows,
        cache_path=cache_jsonl,
        chexbert_backend=chexbert_backend,
        chexbert_api_base_url=chexbert_api_base_url,
        chexbert_api_timeout=chexbert_api_timeout,
        chexbert_device=chexbert_device,
    )
    metrics = compute_metrics_from_cache(rows=rows, cache=cache)
    summary = {
        "predictions_path": str(predictions_path),
        "num_samples": len(rows),
        "newly_evaluated_samples": newly_evaluated,
        "cache_jsonl": str(cache_jsonl),
        "chexbert_backend": chexbert_backend,
        "chexbert_api_base_url": chexbert_api_base_url if chexbert_backend == "api" else "",
        "chexbert_device": chexbert_device if chexbert_backend == "local" else "api_managed",
        "metrics": metrics,
        "labeling": summarize_labels(),
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    rows = load_predictions(args.predictions_path)
    output_json = args.output_json or args.predictions_path.with_name("mimic_mrg_f1_summary.json")
    cache_jsonl = args.cache_jsonl or args.predictions_path.with_name("mimic_mrg_f1_cache.jsonl")
    summary = evaluate_predictions_incremental(
        rows=rows,
        predictions_path=args.predictions_path,
        output_json=output_json,
        cache_jsonl=cache_jsonl,
        chexbert_backend=args.chexbert_backend,
        chexbert_api_base_url=args.chexbert_api_base_url,
        chexbert_api_timeout=args.chexbert_api_timeout,
        chexbert_device=args.chexbert_device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary to: {output_json}")


if __name__ == "__main__":
    main()
