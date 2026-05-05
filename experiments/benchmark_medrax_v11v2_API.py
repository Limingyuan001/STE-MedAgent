import argparse
import os
import sys
from pathlib import Path
from typing import Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments import benchmark_medrax_v11v2 as base
from medrax.tools.remote import get_remote_tool_factories


LOCAL_TOOL_FACTORY_GETTER = base.get_tool_factories


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser(
        description="MedRAX v11v2 benchmark with optional API-backed neural tools.",
        default_restore="off",
    )
    parser.add_argument(
        "--llm-core",
        choices=["32b", "2b"],
        default="32b",
        help="Choose which local OpenAI-compatible LLM core to use for the benchmark.",
    )
    parser.add_argument(
        "--llm-2b-base-url",
        type=str,
        default="http://127.0.0.1:8001/v1",
        help="Base URL for the local 2B LLM when --llm-core 2b is selected.",
    )
    parser.add_argument(
        "--llm-2b-model",
        type=str,
        default="qwen3-vl-2b-instruct-fp8",
        help="Model name for the local 2B LLM when --llm-core 2b is selected.",
    )
    parser.add_argument(
        "--llm-2b-api-key",
        type=str,
        default="",
        help="Optional API key override for the local 2B LLM. Defaults to the current OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--tool-backend",
        choices=["local", "api"],
        default="local",
        help="Choose whether neural tools are loaded in-process or called via the tool server.",
    )
    parser.add_argument(
        "--tool-api-base-url",
        type=str,
        default="http://127.0.0.1:8010",
        help="Base URL for the neural tool server when --tool-backend api is selected.",
    )
    parser.add_argument(
        "--tool-api-timeout",
        type=float,
        default=300.0,
        help="Per-call timeout in seconds for tool server requests when --tool-backend api is selected.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def get_tool_factories(device: str, args: argparse.Namespace) -> Dict[str, object]:
    if args.tool_backend == "api":
        return get_remote_tool_factories(
            api_base_url=args.tool_api_base_url,
            api_timeout=args.tool_api_timeout,
        )
    return LOCAL_TOOL_FACTORY_GETTER(device)


def apply_llm_core_env(args: argparse.Namespace) -> Dict[str, str | None]:
    tracked_keys = ["OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_API_KEY"]
    original_env = {key: os.environ.get(key) for key in tracked_keys}

    if args.llm_core == "2b":
        os.environ["OPENAI_BASE_URL"] = args.llm_2b_base_url
        os.environ["OPENAI_MODEL"] = args.llm_2b_model
        if args.llm_2b_api_key:
            os.environ["OPENAI_API_KEY"] = args.llm_2b_api_key

        effective_api_key = os.environ.get("OPENAI_API_KEY", "")
        print(
            "LLM core: 2b "
            f"(base_url={args.llm_2b_base_url}, model={args.llm_2b_model}, "
            f"api_key={'set' if effective_api_key else 'missing'})"
        )
    else:
        print(
            "LLM core: 32b/default "
            f"(base_url={os.environ.get('OPENAI_BASE_URL', '')}, "
            f"model={os.environ.get('OPENAI_MODEL', '')})"
        )

    return original_env


def restore_llm_core_env(original_env: Dict[str, str | None]) -> None:
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def run_benchmark(args: argparse.Namespace) -> None:
    original_factory_getter = base.get_tool_factories
    original_llm_env = apply_llm_core_env(args)

    def _patched_factory_getter(device: str):
        return get_tool_factories(device, args)

    base.get_tool_factories = _patched_factory_getter
    try:
        base.run_benchmark(args)
    finally:
        base.get_tool_factories = original_factory_getter
        restore_llm_core_env(original_llm_env)


if __name__ == "__main__":
    run_benchmark(parse_args())
