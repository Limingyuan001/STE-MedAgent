import os
import sys
from pathlib import Path
from typing import Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments import benchmark_medrax_v11v2_API as base
from langchain_openai import ChatOpenAI as RealChatOpenAI


FLEX_MODEL = "gpt-5.5"
FLEX_SERVICE_TIER = "flex"
REQUEST_TIMEOUT_SECONDS = 600


def build_parser():
    parser = base.build_parser()
    parser.description = "MedRAX v11v2 API benchmark with GPT-5.5 flex."
    return parser


def parse_args():
    return build_parser().parse_args()


def apply_flex_env(args) -> Dict[str, str | None]:
    tracked_keys = ["OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_SERVICE_TIER"]
    original_env = {key: os.environ.get(key) for key in tracked_keys}

    if args.llm_core == "2b":
        print("Flex override skipped because --llm-core 2b was selected.")
        return original_env

    # GPT-5.5 flex should hit the default OpenAI endpoint rather than any
    # previously configured OpenAI-compatible proxy/base URL.
    os.environ.pop("OPENAI_BASE_URL", None)
    os.environ["OPENAI_MODEL"] = FLEX_MODEL
    os.environ["OPENAI_SERVICE_TIER"] = FLEX_SERVICE_TIER
    print(
        "Flex mode enabled: "
        f"model={FLEX_MODEL}, service_tier={FLEX_SERVICE_TIER}, "
        f"base_url=default_openai, timeout={REQUEST_TIMEOUT_SECONDS}s"
    )
    return original_env


def restore_flex_env(original_env: Dict[str, str | None]) -> None:
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def run_benchmark(args) -> None:
    original_env = apply_flex_env(args)
    original_chat_openai = base.base.ChatOpenAI

    def flex_chat_openai(*chat_args, **chat_kwargs):
        chat_kwargs.setdefault("timeout", REQUEST_TIMEOUT_SECONDS)
        if args.llm_core != "2b":
            # GPT-5.5 rejects top_p on this endpoint; keep the rest of the
            # benchmark config unchanged and strip only the unsupported field.
            chat_kwargs.pop("top_p", None)
            model_kwargs = dict(chat_kwargs.get("model_kwargs") or {})
            model_kwargs.setdefault("service_tier", FLEX_SERVICE_TIER)
            chat_kwargs["model_kwargs"] = model_kwargs
        return RealChatOpenAI(*chat_args, **chat_kwargs)

    base.base.ChatOpenAI = flex_chat_openai
    try:
        base.run_benchmark(args)
    finally:
        base.base.ChatOpenAI = original_chat_openai
        restore_flex_env(original_env)


if __name__ == "__main__":
    run_benchmark(parse_args())
