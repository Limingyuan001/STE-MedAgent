import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments import benchmark_medrax_v11v2 as v11base
from experiments import benchmark_medrax_v11v2_mimic_mrg_concurrent as base
from medrax.tools.remote import get_remote_tool_factories


LOCAL_TOOL_FACTORY_GETTER = v11base.get_tool_factories


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MedRAX v11v2-derived MIMIC MRG benchmark with optional API-backed neural tools."
    )
    parser.add_argument("--mode", choices=["inference", "memory"], default="inference")
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--subset", choices=["full", "frontal"], default="full")
    parser.add_argument("--mimic-image-root", type=Path, default=base.DEFAULT_MIMIC_IMAGE_ROOT)
    parser.add_argument("--memory-db", type=str, default=base.DEFAULT_MEMORY_DB)
    parser.add_argument("--restore", choices=["on", "off"], default="off")
    parser.add_argument("--restore-log", type=str, default="")
    parser.add_argument(
        "--embedding-model",
        type=str,
        choices=base.IMAGE_EMBEDDING_MODEL_CHOICES,
        default=base.DEFAULT_IMAGE_EMBEDDING_MODEL,
    )
    parser.add_argument("--embedding-backend", choices=["local", "api"], default="api")
    parser.add_argument(
        "--embedding-api-base-url",
        type=str,
        default="http://127.0.0.1:8011",
        help="Base URL for the dedicated MIMIC image-embedding server when --embedding-backend api is selected.",
    )
    parser.add_argument(
        "--embedding-api-timeout",
        type=float,
        default=300.0,
        help="Per-call timeout in seconds for image embedding server requests.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--sim-threshold", type=float, default=0.3)
    parser.add_argument("--retrieval-enabled", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--memory-update", choices=["on", "off"], default="on")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tools", type=str, default=",".join(base.DEFAULT_TOOL_NAMES))
    parser.add_argument("--list-tools", action="store_true")
    parser.add_argument("--max-reasoning-steps", type=int, default=7)
    parser.add_argument("--log-tag", type=str, default="")
    parser.add_argument("--log-file", type=str, default="")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument(
        "--sample-concurrency",
        type=int,
        default=1,
        help="Maximum number of MIMIC samples allowed in flight concurrently.",
    )
    parser.add_argument("--chexbert-backend", choices=["api", "local"], default="api")
    parser.add_argument("--chexbert-api-base-url", type=str, default="http://127.0.0.1:8011")
    parser.add_argument("--chexbert-api-timeout", type=float, default=300.0)
    parser.add_argument("--chexbert-device", type=str, default="cuda")
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
        help="Optional API key override for the local 2B LLM.",
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
        os.environ["OPENAI_API_KEY"] = args.llm_2b_api_key or os.environ.get(
            "OPENAI_API_KEY", "mingyuan"
        )
        effective_api_key = os.environ.get("OPENAI_API_KEY", "")
        print(
            "LLM core: 2b "
            f"(base_url={args.llm_2b_base_url}, "
            f"model={args.llm_2b_model}, api_key={'set' if effective_api_key else 'missing'})"
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
    original_v11_factory_getter = v11base.get_tool_factories
    original_base_factory_getter = base.get_tool_factories
    original_base_get_tools = base.get_tools
    original_llm_env = apply_llm_core_env(args)
    original_embedding_backend = getattr(base, "DEFAULT_IMAGE_EMBEDDING_API", "http://127.0.0.1:8011")

    alias_v11_modules = []
    alias_base_modules = []
    for module_name in ("benchmark_medrax_v11v2", "experiments.benchmark_medrax_v11v2"):
        try:
            alias_v11_modules.append(importlib.import_module(module_name))
        except Exception:
            continue
    for module_name in (
        "benchmark_medrax_v11v2_mimic_mrg_concurrent",
        "experiments.benchmark_medrax_v11v2_mimic_mrg_concurrent",
    ):
        try:
            alias_base_modules.append(importlib.import_module(module_name))
        except Exception:
            continue

    def _patched_factory_getter(device: str):
        return get_tool_factories(device, args)

    def _patched_get_tools(tool_names, device):
        factories = _patched_factory_getter(device)
        unknown = [name for name in tool_names if name not in factories]
        if unknown:
            available = ", ".join(sorted(factories.keys()))
            raise ValueError(f"Unknown tools: {unknown}. Available tools: {available}")
        tools = [factories[name]() for name in tool_names]
        print(f"Using tools ({len(tool_names)}): {', '.join(tool_names)}")
        return tools

    v11base.get_tool_factories = _patched_factory_getter
    base.get_tool_factories = _patched_factory_getter
    base.get_tools = _patched_get_tools
    for module in alias_v11_modules:
        setattr(module, "get_tool_factories", _patched_factory_getter)
    for module in alias_base_modules:
        setattr(module, "get_tool_factories", _patched_factory_getter)
        setattr(module, "get_tools", _patched_get_tools)
    try:
        base.DEFAULT_IMAGE_EMBEDDING_API = args.embedding_api_base_url
        base.run_benchmark(args)
    finally:
        v11base.get_tool_factories = original_v11_factory_getter
        base.get_tool_factories = original_base_factory_getter
        base.get_tools = original_base_get_tools
        base.DEFAULT_IMAGE_EMBEDDING_API = original_embedding_backend
        restore_llm_core_env(original_llm_env)


if __name__ == "__main__":
    run_benchmark(parse_args())
