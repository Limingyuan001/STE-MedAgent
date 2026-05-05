import argparse
import json
import logging
import os
import re
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from medrax.agent import AgentV13
from medrax.agent.agent_v13 import (
    ALLOWED_PATHOLOGICAL_FINDINGS,
    ALLOWED_QUESTION_TYPES,
    DEFAULT_EXTRACT_PROMPT,
    PATHOLOGICAL_FINDING_ALIAS_TO_CANONICAL,
)
from medrax.utils import load_prompts_from_file

try:
    from benchmark_medrax_chexbench import (
        DEFAULT_CHEXBENCH_PATH,
        DEFAULT_IMAGE_BASE,
        DEFAULT_TOOL_NAMES,
        PROMPT_FILE,
        ROOT,
        SUPPORTED_TASKS,
        TagEmbeddingService,
        add_usage,
        apply_restore_filter,
        build_instance_meta,
        build_local_image_data_url,
        calculate_cost,
        configure_logging,
        cosine_similarity,
        extract_choice_letter,
        get_tool_factories,
        get_tools,
        load_chexbench_entries,
        load_completed_question_ids,
        normalize_usage_from_callback,
        parse_tool_names,
        run_agent_once,
        run_choice_extraction,
        valid_embedding,
        write_restore_marker,
    )
except ImportError:
    from experiments.benchmark_medrax_chexbench import (
        DEFAULT_CHEXBENCH_PATH,
        DEFAULT_IMAGE_BASE,
        DEFAULT_TOOL_NAMES,
        PROMPT_FILE,
        ROOT,
        SUPPORTED_TASKS,
        TagEmbeddingService,
        add_usage,
        apply_restore_filter,
        build_instance_meta,
        build_local_image_data_url,
        calculate_cost,
        configure_logging,
        cosine_similarity,
        extract_choice_letter,
        get_tool_factories,
        get_tools,
        load_chexbench_entries,
        load_completed_question_ids,
        normalize_usage_from_callback,
        parse_tool_names,
        run_agent_once,
        run_choice_extraction,
        valid_embedding,
        write_restore_marker,
    )

warnings.filterwarnings("ignore")
_ = load_dotenv()

DEFAULT_MEMORY_DB = f"{ROOT}/experiments/memory/memory_chexbench_v13.jsonl"
ALLOWED_PATHOLOGICAL_FINDINGS_SET = set(ALLOWED_PATHOLOGICAL_FINDINGS)


def _dedupe_keep_order(items: List[str], max_items: Optional[int] = None) -> List[str]:
    seen = set()
    results: List[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        results.append(text)
        if max_items is not None and len(results) >= max_items:
            break
    return results


def canonicalize_pathological_finding(term: str) -> Optional[str]:
    cleaned = re.sub(r"\s+", " ", re.sub(r"[_/\-]+", " ", str(term).strip().lower()))
    if not cleaned:
        return None

    direct = PATHOLOGICAL_FINDING_ALIAS_TO_CANONICAL.get(cleaned)
    if direct and direct in ALLOWED_PATHOLOGICAL_FINDINGS_SET:
        return direct

    alias_items = sorted(
        PATHOLOGICAL_FINDING_ALIAS_TO_CANONICAL.items(),
        key=lambda kv: len(kv[0]),
        reverse=True,
    )
    for alias, canonical in alias_items:
        if canonical not in ALLOWED_PATHOLOGICAL_FINDINGS_SET:
            continue
        if re.search(rf"\b{re.escape(alias)}\b", cleaned):
            return canonical
    return None


def normalize_tags_for_embedding(tags: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(tags, dict):
        return {
            "question_type": [],
            "symptoms": [],
            "demographics": [],
            "risk_factors": [],
            "pathological_findings": [],
        }

    def _as_list(value: Any) -> List[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    allowed_type_map = {x.lower(): x for x in ALLOWED_QUESTION_TYPES}
    question_type_raw = _as_list(tags.get("question_type", []))
    cleaned_qt = []
    for item in question_type_raw:
        mapped = allowed_type_map.get(item.lower())
        if mapped:
            cleaned_qt.append(mapped)
    qt = _dedupe_keep_order(cleaned_qt, max_items=len(ALLOWED_QUESTION_TYPES))

    symptoms = _dedupe_keep_order(
        [x for x in _as_list(tags.get("symptoms", [])) if x.lower() not in {"yes", "no"}],
        max_items=30,
    )
    demographics = _dedupe_keep_order(
        [x for x in _as_list(tags.get("demographics", [])) if x.lower() not in {"yes", "no"}],
        max_items=20,
    )
    risk_factors = _dedupe_keep_order(
        [x for x in _as_list(tags.get("risk_factors", [])) if x.lower() not in {"yes", "no"}],
        max_items=20,
    )

    findings = tags.get("pathological_findings", [])
    if isinstance(findings, str):
        findings = [findings]
    findings_clean = []
    if isinstance(findings, list):
        canonical_findings = []
        for item in findings:
            canonical = canonicalize_pathological_finding(str(item))
            if canonical:
                canonical_findings.append(canonical)
        findings_clean = _dedupe_keep_order(canonical_findings, max_items=len(ALLOWED_PATHOLOGICAL_FINDINGS))

    return {
        "question_type": qt,
        "symptoms": symptoms,
        "demographics": demographics,
        "risk_factors": risk_factors,
        "pathological_findings": findings_clean,
    }


class CheXBenchTagEmbeddingService(TagEmbeddingService):
    @staticmethod
    def tags_to_text(tags: Dict[str, Any]) -> str:
        normalized = normalize_tags_for_embedding(tags)
        question_type = normalized.get("question_type", [])
        symptoms = normalized.get("symptoms", [])
        demographics = normalized.get("demographics", [])
        risk_factors = normalized.get("risk_factors", [])
        findings = normalized.get("pathological_findings", [])

        q_text = ", ".join(str(x) for x in question_type if str(x).strip())
        s_text = ", ".join(str(x) for x in symptoms if str(x).strip())
        d_text = ", ".join(str(x) for x in demographics if str(x).strip())
        r_text = ", ".join(str(x) for x in risk_factors if str(x).strip())
        f_text = ", ".join(str(x) for x in findings if str(x).strip())
        return (
            f"question_type: {q_text}\n"
            f"symptoms: {s_text}\n"
            f"demographics: {d_text}\n"
            f"risk_factors: {r_text}\n"
            f"pathological_findings: {f_text}"
        ).strip()

    def embed_tags(self, tags: Dict[str, Any]) -> List[float]:
        return self.embed_text(self.tags_to_text(tags))


class JsonlLongTermMemory:
    def __init__(self, db_path: str, embedder: CheXBenchTagEmbeddingService):
        self.db_path = db_path
        self.embedder = embedder
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.records: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.db_path):
            return
        with open(self.db_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                tags = record.get("tags", {})
                if isinstance(tags, dict):
                    record["tags"] = normalize_tags_for_embedding(tags)
                if not valid_embedding(record.get("embedding")):
                    tags = record.get("tags", {})
                    if isinstance(tags, dict):
                        record["embedding"] = self.embedder.embed_tags(tags)
                self.records.append(record)

    def append_record(self, record: Dict[str, Any]) -> None:
        tags = record.get("tags", {})
        if isinstance(tags, dict):
            record["tags"] = normalize_tags_for_embedding(tags)
        if not valid_embedding(record.get("embedding")):
            tags = record.get("tags", {})
            if isinstance(tags, dict):
                record["embedding"] = self.embedder.embed_tags(tags)
        with open(self.db_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records.append(record)

    def _build_summary(self, record: Dict[str, Any]) -> str:
        tags = record.get("tags", {})
        qt = tags.get("question_type", []) if isinstance(tags, dict) else []
        findings = tags.get("pathological_findings", []) if isinstance(tags, dict) else []
        tools = record.get("tools", []) or []
        top_tools = []
        for tool in tools[:3]:
            if isinstance(tool, dict):
                top_tools.append(
                    f"{tool.get('tool_name', 'unknown')}:{tool.get('score', 'NA')}"
                )
        return (
            f"question_type={qt}; findings={findings[:3]}; "
            f"tool_scores={top_tools}"
        )

    def retrieve(
        self,
        tags: Dict[str, Any],
        embedding: Optional[List[float]],
        top_k: int,
        similarity_threshold: float,
        state: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not self.records:
            return []

        query_embedding = embedding
        if not valid_embedding(query_embedding):
            query_embedding = self.embedder.embed_tags(tags)
        if not valid_embedding(query_embedding):
            return []

        import numpy as np

        query_vec = np.asarray(query_embedding, dtype=np.float32)
        meta = state.get("instance_meta", {}) if isinstance(state, dict) else {}
        current_case = str(meta.get("case_id", ""))
        current_instance = str(meta.get("instance_id", meta.get("question_id", "")))

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for record in self.records:
            record_case = str(record.get("case_id", ""))
            record_instance = str(record.get("instance_id", record.get("question_id", "")))
            if current_case and current_instance and record_case == current_case and record_instance == current_instance:
                continue

            rec_embedding = record.get("embedding")
            if not valid_embedding(rec_embedding):
                continue
            score = cosine_similarity(query_vec, np.asarray(rec_embedding, dtype=np.float32))
            if score < similarity_threshold:
                continue
            scored.append((score, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected: List[Dict[str, Any]] = []
        for score, record in scored[: max(0, top_k)]:
            selected.append(
                {
                    "case_id": record.get("case_id"),
                    "instance_id": record.get("instance_id", record.get("question_id")),
                    "tools": record.get("tools", []),
                    "summary": self._build_summary(record),
                    "retrieval_score": round(float(score), 4),
                }
            )
        return selected


def build_memory_record(
    entry: Dict[str, Any],
    state_snapshot: Dict[str, Any],
    answer_letter: str,
    tool_evaluation: Dict[str, Any],
    embedding_model: str,
) -> Dict[str, Any]:
    tags = state_snapshot.get("tags", {}) if isinstance(state_snapshot.get("tags", {}), dict) else {}
    tags = normalize_tags_for_embedding(tags)
    return {
        "case_id": entry["entry_id"],
        "instance_id": entry["entry_id"],
        "question_id": entry["entry_id"],
        "embedding_model": embedding_model,
        "embedding": state_snapshot.get("tag_embedding"),
        "tags": tags,
        "tools": tool_evaluation.get("tools", []) if isinstance(tool_evaluation, dict) else [],
        "question": entry["question_stem"],
        "options": entry["options"],
        "agent_answer": answer_letter,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MedRAX CheXbench benchmark v13 — evaluate AgentV13 on CheXbench with a dataset-specific tag ontology."
    )
    parser.add_argument("--mode", choices=["inference", "memory"], default="inference")
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_CHEXBENCH_PATH,
        help=f"Path to CheXbench JSON file. Default: {DEFAULT_CHEXBENCH_PATH}",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(sorted(SUPPORTED_TASKS)),
        help="Comma-separated task names. Default: all supported tasks.",
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=0,
        help="Maximum entries per task type (0 = unlimited).",
    )
    parser.add_argument(
        "--image-base",
        type=str,
        default=DEFAULT_IMAGE_BASE,
        help=f"Base directory for image datasets. Default: {DEFAULT_IMAGE_BASE}",
    )
    parser.add_argument("--memory-db", type=str, default=DEFAULT_MEMORY_DB)
    parser.add_argument(
        "--restore",
        choices=["on", "off"],
        default="off",
        help="Skip entries whose ID already exists in --memory-db.",
    )
    parser.add_argument("--restore-log", type=str, default="")
    parser.add_argument("--embedding-model", type=str, default="bge-small-en-v1.5")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--sim-threshold", type=float, default=0.3)
    parser.add_argument("--memory-update", choices=["on", "off"], default="on")
    parser.add_argument(
        "--shuffle-id",
        type=int,
        default=None,
        help="Shuffle entries within each task using this random seed.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-source", choices=["auto", "remote", "local"], default="local")
    parser.add_argument("--tools", type=str, default=",".join(DEFAULT_TOOL_NAMES))
    parser.add_argument("--list-tools", action="store_true")
    parser.add_argument(
        "--max-reasoning-steps",
        type=int,
        default=7,
        help="Maximum reasoning steps per question. Default: 7.",
    )
    parser.add_argument("--log-tag", type=str, default="")
    parser.add_argument("--log-file", type=str, default="")
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args()


def run_benchmark(args: argparse.Namespace) -> None:
    tool_names = parse_tool_names(args.tools, args.device)
    if args.list_tools:
        print("Available tools:")
        for name in sorted(get_tool_factories(args.device).keys()):
            print(f"- {name}")
        return

    selected_tasks = set()
    for t in args.tasks.split(","):
        t = t.strip()
        if t:
            selected_tasks.add(t)
    unsupported = selected_tasks - SUPPORTED_TASKS
    if unsupported:
        print(f"Warning: unsupported tasks will be skipped: {unsupported}")
    selected_tasks = selected_tasks & SUPPORTED_TASKS

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    log_path = configure_logging(args.log_tag, args.log_file, model_name, args.restore_log)

    print(f"Mode: {args.mode}")
    print(f"Data: {args.data_path}")
    print(f"Tasks: {', '.join(sorted(selected_tasks))}")
    print(f"Image base: {args.image_base}")
    print(f"Memory DB: {args.memory_db}")
    if args.shuffle_id is None:
        print("Shuffle: disabled")
    else:
        print(f"Shuffle: enabled (shuffle_id={args.shuffle_id})")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Retrieval: top_k={args.top_k}, sim_threshold={args.sim_threshold}")

    all_entries = load_chexbench_entries(
        data_path=args.data_path,
        tasks=selected_tasks,
        max_per_task=args.max_per_task,
        shuffle_seed=args.shuffle_id,
        image_base=args.image_base,
    )
    loaded_total = len(all_entries)

    task_counts: Dict[str, int] = defaultdict(int)
    for entry in all_entries:
        key = f"{entry['task_type']} / {entry['data_source']}"
        task_counts[key] += 1
    print(f"Loaded {loaded_total} entries:")
    for key in sorted(task_counts):
        print(f"  {key}: {task_counts[key]}")

    restore_mode = str(getattr(args, "restore", "off")).lower() == "on"
    entries = all_entries
    if restore_mode:
        completed_ids = load_completed_question_ids(args.memory_db)
        entries, restore_summary = apply_restore_filter(all_entries, completed_ids)
        print(
            "Restore: enabled "
            f"(completed={restore_summary['completed_count']}, "
            f"remaining={restore_summary['remaining_count']})"
        )
        if args.restore_log:
            write_restore_marker(
                log_path=log_path,
                data_path=args.data_path,
                memory_db_path=args.memory_db,
                shuffle_id=args.shuffle_id,
                restore_summary=restore_summary,
            )
        if restore_summary["remaining_count"] == 0:
            print("All entries already processed. Nothing left to run.")
            return
    else:
        print("Restore: disabled")

    total_questions = len(entries)
    print(f"Entries scheduled this run: {total_questions}")

    tools = get_tools(tool_names, args.device)
    prompts = load_prompts_from_file(PROMPT_FILE)
    system_prompt = prompts.get("MEDICAL_ASSISTANT", "")

    embedder = CheXBenchTagEmbeddingService(args.embedding_model)
    memory_store = JsonlLongTermMemory(args.memory_db, embedder)

    checkpointer = MemorySaver()
    model = ChatOpenAI(
        model=model_name,
        temperature=args.temperature,
        top_p=0.95,
    )
    agent = AgentV13(
        model=model,
        tools=tools,
        checkpointer=checkpointer,
        system_prompt=system_prompt,
        extract_prompt=DEFAULT_EXTRACT_PROMPT,
        tag_embedder=lambda tags, state: embedder.embed_tags(tags),
        memory_retriever=memory_store.retrieve,
        retrieval_top_k=args.top_k,
        similarity_threshold=args.sim_threshold,
        log_tools=True,
        log_dir="logs",
        max_reasoning_steps=args.max_reasoning_steps,
    )

    processed = 0
    correct = 0
    skipped = 0
    start_all = time.time()
    task_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0, "skipped": 0})

    for entry in entries:
        processed += 1
        entry_id = entry["entry_id"]
        task_type = entry["task_type"]
        data_source = entry["data_source"]
        ground_truth = entry["answer_letter"]
        question = entry["question"]
        image_paths = entry["image_paths"]
        task_key = f"{task_type} / {data_source}"

        task_stats[task_key]["total"] += 1

        if not image_paths:
            skipped += 1
            task_stats[task_key]["skipped"] += 1
            print(
                f"[{processed}/{total_questions}] id={entry_id} "
                f"SKIPPED (no images found) raw={entry.get('raw_image_path')}"
            )
            continue

        instance_meta = build_instance_meta(entry)

        image_urls: List[str] = []
        for image_path in image_paths:
            data_url = build_local_image_data_url(image_path)
            if data_url:
                image_urls.append(data_url)

        if not image_urls:
            skipped += 1
            task_stats[task_key]["skipped"] += 1
            print(
                f"[{processed}/{total_questions}] id={entry_id} "
                "SKIPPED (failed to encode images)"
            )
            continue

        os.environ["MEDRAX_IMAGE_PATHS"] = json.dumps(image_paths)

        prompt = (
            "Answer this question correctly using chain of thought reasoning and "
            "carefully evaluating choices. Solve using our own vision and reasoning and then "
            "use tools to complement your reasoning. Trust your own judgement over any tools.\n"
            f"{question}"
        )
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend([{"type": "image_url", "image_url": {"url": url}} for url in image_urls])
        messages = [HumanMessage(content=content)]
        init_state = AgentV13.build_initial_state(messages, instance_meta=instance_meta, mode=args.mode)
        thread = {"configurable": {"thread_id": entry_id}}

        status = "ok"
        error_message = ""
        tool_evaluation: Dict[str, Any] = {}
        memory_write_status = "skipped"
        raw_choice_response = ""
        duration = 0.0
        usage_main = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "total_cost": 0.0}
        usage_choice = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "total_cost": 0.0}
        usage_eval = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "total_cost": 0.0}
        final_response = ""
        model_answer = ""
        state_snapshot: Dict[str, Any] = {}

        try:
            start = time.time()

            final_response, state_snapshot, usage_main = run_agent_once(agent, thread, init_state)
            model_answer, raw_choice_response, usage_choice = run_choice_extraction(
                agent.extract_model, final_response
            )
            if not model_answer:
                model_answer = extract_choice_letter(final_response)
            state_snapshot["final_answer"] = model_answer

            if args.mode == "memory":
                with get_openai_callback() as cb:
                    tool_evaluation = agent.score_tool_contributions(
                        state_snapshot=state_snapshot,
                    )
                usage_eval = normalize_usage_from_callback(cb)

                if args.memory_update == "on":
                    memory_record = build_memory_record(
                        entry=entry,
                        state_snapshot=state_snapshot,
                        answer_letter=model_answer,
                        tool_evaluation=tool_evaluation,
                        embedding_model=args.embedding_model,
                    )
                    memory_store.append_record(memory_record)
                    memory_write_status = "written"
                else:
                    memory_write_status = "disabled"
            else:
                memory_write_status = "inference_mode"

            duration = time.time() - start
        except Exception as exc:
            status = "error"
            error_message = str(exc)
            duration = time.time() - start if "start" in locals() else 0.0

        is_correct = model_answer == ground_truth if model_answer else False
        if is_correct:
            correct += 1
            task_stats[task_key]["correct"] += 1
        if status != "ok":
            skipped += 1
            task_stats[task_key]["skipped"] += 1

        usage_total = add_usage(add_usage(usage_main, usage_choice), usage_eval)
        total_cost = usage_total["total_cost"]
        if total_cost <= 0:
            total_cost = calculate_cost(usage_total["prompt_tokens"], usage_total["completion_tokens"])

        retrieved = state_snapshot.get("retrieved_memories", []) if isinstance(state_snapshot, dict) else []
        retrieval_case_ids = [item.get("case_id") for item in retrieved if isinstance(item, dict)]
        retrieval_scores = [item.get("retrieval_score") for item in retrieved if isinstance(item, dict)]

        log_entry = {
            "entry_id": entry_id,
            "task_type": task_type,
            "data_source": data_source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "error": error_message,
            "mode": args.mode,
            "max_reasoning_steps": args.max_reasoning_steps,
            "shuffle_id": args.shuffle_id,
            "model": model_name,
            "temperature": args.temperature,
            "duration": round(duration, 3),
            "usage": usage_total,
            "cost": round(float(total_cost), 6),
            "model_answer": model_answer,
            "correct_answer": ground_truth,
            "is_correct": is_correct,
            "raw_response": final_response,
            "raw_choice_response": raw_choice_response,
            "tags": state_snapshot.get("tags", {}) if isinstance(state_snapshot, dict) else {},
            "tag_embedding_model": args.embedding_model,
            "retrieved_case_ids": retrieval_case_ids,
            "retrieval_scores": retrieval_scores,
            "tool_traces": state_snapshot.get("tool_traces", []) if isinstance(state_snapshot, dict) else [],
            "tool_evaluation": tool_evaluation,
            "memory_write_status": memory_write_status,
            "input": {
                "question": question,
                "question_stem": entry["question_stem"],
                "options": entry["options"],
                "image_paths": image_paths,
                "slice": entry.get("slice"),
            },
        }
        logging.info(json.dumps(log_entry, ensure_ascii=False))

        acc = (correct / (processed - skipped)) if (processed - skipped) > 0 else 0.0
        print(
            f"[{processed}/{total_questions}] task={task_type} src={data_source} "
            f"pred={model_answer or '-'} gt={ground_truth or '-'} "
            f"ok={is_correct} status={status} acc={acc:.3f}"
        )

    elapsed = time.time() - start_all
    answered = processed - skipped
    final_acc = (correct / answered) if answered else 0.0

    print("\n" + "=" * 80)
    print("CheXbench Summary")
    print("=" * 80)
    print(f"{'Task':<30} | {'Source':<14} | {'Total':>5} | {'Run':>4} | {'Skip':>4} | {'Acc':>8}")
    print("-" * 80)

    overall_correct = 0
    overall_run = 0
    overall_skip = 0
    for key in sorted(task_stats):
        stats = task_stats[key]
        parts = key.split(" / ")
        task = parts[0] if parts else key
        source = parts[1] if len(parts) > 1 else ""
        run = stats["total"] - stats["skipped"]
        acc_str = f"{stats['correct'] / run:.3f}" if run > 0 else "N/A"
        print(f"{task:<30} | {source:<14} | {stats['total']:>5} | {run:>4} | {stats['skipped']:>4} | {acc_str:>8}")
        overall_correct += stats["correct"]
        overall_run += run
        overall_skip += stats["skipped"]

    overall_total = overall_run + overall_skip
    overall_acc_str = f"{overall_correct / overall_run:.3f}" if overall_run > 0 else "N/A"
    print("-" * 80)
    print(f"{'Overall':<30} | {'':<14} | {overall_total:>5} | {overall_run:>4} | {overall_skip:>4} | {overall_acc_str:>8}")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Final accuracy: {final_acc:.3f}" if answered else "Final accuracy: N/A")
    print(f"Elapsed: {elapsed:.2f}s")


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
