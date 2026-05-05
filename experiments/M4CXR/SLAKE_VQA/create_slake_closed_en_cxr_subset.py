#!/usr/bin/env python3
"""Extract the 114-sample SLAKE VQA subset used by M4CXR.

Filter rule:
- split: test.json
- q_lang == "en"
- answer_type == "CLOSED"
- modality == "X-Ray"

By default this script writes the filtered subset to:
  experiments/M4CXR/SLAKE_VQA/slake_test_en_closed_xray_114.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("/data/lmy/datasets/slake/test.json")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "slake_test_en_closed_xray_114.json"
EXPECTED_COUNT = 114


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the 114 English closed-ended X-Ray questions from SLAKE test.json.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to SLAKE test.json. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=EXPECTED_COUNT,
        help=f"Expected subset size. Default: {EXPECTED_COUNT}",
    )
    return parser.parse_args()


def is_target_sample(sample: dict[str, Any]) -> bool:
    return (
        str(sample.get("q_lang", "")).strip().lower() == "en"
        and str(sample.get("answer_type", "")).strip().upper() == "CLOSED"
        and str(sample.get("modality", "")).strip() == "X-Ray"
    )


def main() -> None:
    args = parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {args.input}, got {type(data).__name__}")

    subset = [sample for sample in data if isinstance(sample, dict) and is_target_sample(sample)]

    if len(subset) != args.expected_count:
        raise ValueError(
            f"Filtered subset size mismatch: got {len(subset)}, expected {args.expected_count}. "
            "Please verify the SLAKE source data or filtering rule."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Saved {len(subset)} samples.")


if __name__ == "__main__":
    main()
