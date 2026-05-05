import argparse
import base64
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

if TYPE_CHECKING:
    import openai


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_JSON = SCRIPT_DIR / "slake_test_en_closed_xray_114.json"
DEFAULT_IMAGE_ROOT = Path("/data/lmy/datasets/slake/imgs")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "local_api_slake_vqa"
DEFAULT_SYSTEM_PROMPT = (
    "You are a medical imaging expert. Answer the question using a short phrase only. "
    "Base your answer only on the provided X-ray image. "
    "Do not explain your reasoning or add extra words."
)


class SimpleChatCompletions:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def create(self, **payload: Any) -> dict[str, Any]:
        endpoint = f"{self.base_url}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from local API: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to reach local API at {endpoint}: {exc}") from exc


class SimpleChat:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.completions = SimpleChatCompletions(base_url=base_url, api_key=api_key)


class SimpleOpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.chat = SimpleChat(base_url=base_url, api_key=api_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local OpenAI-compatible multimodal model on SLAKE closed X-ray VQA."
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=DEFAULT_DATASET_JSON,
        help="Path to the SLAKE JSON subset.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help="Root directory containing SLAKE images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where predictions.jsonl and summary.json are written.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the local API.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Maximum completion tokens per sample.",
    )
    parser.add_argument(
        "--resume-log",
        type=Path,
        default=None,
        help="Optional JSONL file with prior results to reuse for skip-processed mode.",
    )
    parser.add_argument(
        "--skip-processed",
        action="store_true",
        help="Skip qids already present in the resume log or predictions.jsonl.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override OPENAI_MODEL for this run. This must match the model id exposed by <base-url>/models.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Override OPENAI_API_KEY for this run.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override OPENAI_BASE_URL for this run, for example http://127.0.0.1:8000/v1 or http://127.0.0.1:8001/v1.",
    )
    return parser.parse_args()


def load_slake_samples(dataset_json: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    with dataset_json.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    if max_samples is not None:
        return samples[:max_samples]
    return samples


def build_image_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime_type = mime_type or "image/jpeg"

    with image_path.open("rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def normalize_text(text: str | None) -> str:
    if not text:
        return ""

    normalized = text.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[.,!?;:]+$", "", normalized)
    return normalized.strip()


def make_phrase_pattern(answer: str) -> re.Pattern[str]:
    escaped = re.escape(answer)
    return re.compile(rf"(?<!\w){escaped}(?!\w)")


def extract_closed_answer(
    prediction: str | None, valid_answers: list[str] | tuple[str, ...]
) -> str | None:
    normalized = normalize_text(prediction)
    if not normalized:
        return None

    answers = [normalize_text(answer) for answer in valid_answers]
    answers = sorted(dict.fromkeys(answers), key=len, reverse=True)
    answer_set = set(answers)

    if normalized in answer_set:
        return normalized

    for answer in answers:
        pattern = make_phrase_pattern(answer)

        prompt_like_patterns = [
            re.compile(
                rf"(?:^|\b)(?:final answer|answer)\s*(?:is|:)?\s*(?P<ans>{re.escape(answer)})(?!\w)"
            ),
            re.compile(rf"^(?:it is|it's|its)\s+(?P<ans>{re.escape(answer)})(?!\w)"),
            re.compile(rf"^(?P<ans>{re.escape(answer)})(?:\b|$)"),
        ]

        for candidate_pattern in prompt_like_patterns:
            match = candidate_pattern.search(normalized)
            if match:
                return normalize_text(match.group("ans"))

        if pattern.search(normalized):
            return answer

    return None


def compute_sample_recall(ground_truth: str, raw_prediction: str | None) -> float:
    gt_words = normalize_text(ground_truth).split()
    if not gt_words:
        return 0.0

    pred_words = set(normalize_text(raw_prediction).split())
    matched = sum(1 for word in gt_words if word in pred_words)
    return matched / len(gt_words)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_jsonl_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for record in records:
        qid = str(record.get("qid"))
        if qid:
            deduped[qid] = record
    return list(deduped.values())


def resolve_image_path(image_root: Path, sample: dict[str, Any]) -> Path:
    return image_root / sample["img_name"]


def extract_message_text(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content.strip()

    if isinstance(message_content, list):
        text_parts: list[str] = []
        for item in message_content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return "\n".join(part for part in text_parts if part).strip()

    return str(message_content).strip()


def extract_usage(response: Any) -> dict[str, Any] | None:
    if isinstance(response, dict):
        usage = response.get("usage")
        if isinstance(usage, dict):
            return {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        return None

    if getattr(response, "usage", None) is not None:
        return {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        }
    return None


def extract_raw_prediction(response: Any) -> str:
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return extract_message_text(message.get("content"))

    return extract_message_text(response.choices[0].message.content)


def create_multimodal_request(
    client: Any,
    model_name: str,
    question: str,
    image_data_url: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 3,
) -> tuple[str, float, dict[str, Any] | None]:
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Answer the following medical VQA question using a short phrase only.\n"
                        f"Question: {question}"
                    ),
                },
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            duration = time.time() - start_time

            raw_prediction = extract_raw_prediction(response)
            usage = extract_usage(response)

            return raw_prediction, duration, usage
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(min(2**attempt, 10))

    assert last_error is not None
    raise last_error


def evaluate_predictions(
    records: list[dict[str, Any]],
    total_samples: int,
    model_name: str,
    dataset_json: Path,
    image_root: Path,
) -> dict[str, Any]:
    processed_records = dedupe_records(records)
    processed_count = len(processed_records)
    valid_predictions = sum(1 for record in processed_records if record.get("predicted_answer"))
    correct_predictions = sum(1 for record in processed_records if record.get("is_correct") == 1)
    recall_sum = sum(float(record.get("sample_recall", 0.0)) for record in processed_records)

    accuracy = correct_predictions / total_samples if total_samples else 0.0
    recall = recall_sum / total_samples if total_samples else 0.0

    return {
        "timestamp": datetime.now().isoformat(),
        "total_samples": total_samples,
        "processed_samples": processed_count,
        "valid_predictions": valid_predictions,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
        "recall": recall,
        "model": model_name,
        "dataset_json": str(dataset_json),
        "image_root": str(image_root),
    }


def build_processed_qids(
    predictions_path: Path, resume_log: Path | None, skip_processed: bool
) -> tuple[set[str], list[dict[str, Any]]]:
    existing_records: list[dict[str, Any]] = []

    if predictions_path.exists():
        existing_records.extend(load_jsonl_records(predictions_path))

    if resume_log and resume_log != predictions_path and resume_log.exists():
        existing_records.extend(load_jsonl_records(resume_log))

    deduped_records = dedupe_records(existing_records)
    if not skip_processed:
        return set(), deduped_records

    qids = {str(record.get("qid")) for record in deduped_records if record.get("qid") is not None}
    return qids, deduped_records


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()

    dataset_json = args.dataset_json.resolve()
    image_root = args.image_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    if not dataset_json.exists():
        raise FileNotFoundError(f"Dataset JSON not found: {dataset_json}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    model_name = args.model or os.getenv("OPENAI_MODEL")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")

    if not model_name:
        raise ValueError("Model name is not set. Pass --model or set OPENAI_MODEL.")
    if not api_key:
        raise ValueError("API key is not set. Pass --api-key or set OPENAI_API_KEY.")
    if not base_url:
        raise ValueError("Base URL is not set. Pass --base-url or set OPENAI_BASE_URL.")

    samples = load_slake_samples(dataset_json, args.max_samples)
    valid_answers = sorted({normalize_text(sample["answer"]) for sample in samples}, key=len, reverse=True)

    processed_qids, existing_records = build_processed_qids(
        predictions_path=predictions_path,
        resume_log=args.resume_log.resolve() if args.resume_log else None,
        skip_processed=args.skip_processed,
    )
    all_records = {str(record.get("qid")): record for record in existing_records if record.get("qid") is not None}

    try:
        import openai

        client: Any = openai.OpenAI(api_key=api_key, base_url=base_url)
        client_mode = "openai-sdk"
    except ImportError:
        client = SimpleOpenAICompatibleClient(api_key=api_key, base_url=base_url)
        client_mode = "stdlib-fallback"

    print(f"Dataset: {dataset_json}")
    print(f"Image root: {image_root}")
    print(f"Output dir: {output_dir}")
    print(f"Model: {model_name}")
    print(f"Client mode: {client_mode}")
    print(f"Loaded samples: {len(samples)}")
    if args.skip_processed:
        print(f"Previously processed qids: {len(processed_qids)}")

    for index, sample in enumerate(samples, start=1):
        qid = str(sample["qid"])
        if args.skip_processed and qid in processed_qids:
            print(f"[{index}/{len(samples)}] Skipping qid={qid} (already processed)")
            continue

        image_path = resolve_image_path(image_root, sample)
        base_record = {
            "timestamp": datetime.now().isoformat(),
            "qid": sample["qid"],
            "img_id": sample.get("img_id"),
            "img_name": sample.get("img_name"),
            "question": sample.get("question"),
            "ground_truth": sample.get("answer"),
            "normalized_ground_truth": normalize_text(sample.get("answer")),
            "raw_prediction": None,
            "normalized_prediction": None,
            "predicted_answer": None,
            "is_correct": 0,
            "sample_recall": 0.0,
            "duration": 0.0,
            "status": "pending",
            "error": None,
            "model": model_name,
            "image_path": str(image_path),
            "usage": None,
        }

        if not image_path.exists():
            record = {
                **base_record,
                "status": "skipped",
                "error": f"Image file not found: {image_path}",
            }
            append_jsonl(predictions_path, record)
            all_records[qid] = record
            print(f"[{index}/{len(samples)}] qid={qid} skipped: missing image")
            continue

        try:
            image_data_url = build_image_data_url(image_path)
            raw_prediction, duration, usage = create_multimodal_request(
                client=client,
                model_name=model_name,
                question=sample["question"],
                image_data_url=image_data_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            normalized_prediction = normalize_text(raw_prediction)
            predicted_answer = extract_closed_answer(raw_prediction, valid_answers)
            sample_recall = compute_sample_recall(sample["answer"], raw_prediction)
            is_correct = int(predicted_answer == normalize_text(sample["answer"]))

            record = {
                **base_record,
                "timestamp": datetime.now().isoformat(),
                "raw_prediction": raw_prediction,
                "normalized_prediction": normalized_prediction,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "sample_recall": sample_recall,
                "duration": round(duration, 4),
                "status": "completed",
                "usage": usage,
            }
            append_jsonl(predictions_path, record)
            all_records[qid] = record

            print(
                f"[{index}/{len(samples)}] qid={qid} "
                f"pred={predicted_answer!r} gt={normalize_text(sample['answer'])!r} "
                f"correct={is_correct}"
            )
        except Exception as exc:
            record = {
                **base_record,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(exc),
            }
            append_jsonl(predictions_path, record)
            all_records[qid] = record
            print(f"[{index}/{len(samples)}] qid={qid} error: {exc}")

    summary = evaluate_predictions(
        records=list(all_records.values()),
        total_samples=len(samples),
        model_name=model_name,
        dataset_json=dataset_json,
        image_root=image_root,
    )

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\nFinal summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
