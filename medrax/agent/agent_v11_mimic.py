import json
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from .agent_v11 import AgentStateV11, AgentV11


DEFAULT_EXTRACT_PROMPT_MIMIC = """You are preparing metadata tags for a single-image chest X-ray findings generation task.
Return JSON only with this exact schema:
{
  "task_type": ["single_image_findings_generation"],
  "view_position": ["..."],
  "output_format": ["findings_only"]
}
"""

DEFAULT_PROCESS_POLICY_MIMIC = """The retrieved references below are similar prior cases and provide tool reliability guidance only.
They are not ground-truth findings for the current image.
When tools conflict, prefer tools with higher scores, but use your own judgment on the current image.
Your final answer must be findings text only for the current chest X-ray image."""


class AgentV11Mimic(AgentV11):
    def __init__(
        self,
        model: BaseLanguageModel,
        tools: List[BaseTool],
        *,
        checkpointer: Any = None,
        system_prompt: str = "",
        extract_prompt: str = DEFAULT_EXTRACT_PROMPT_MIMIC,
        process_policy_prompt: str = DEFAULT_PROCESS_POLICY_MIMIC,
        tag_embedder=None,
        memory_retriever=None,
        retrieval_top_k: int = 3,
        similarity_threshold: float = 0.3,
        log_tools: bool = True,
        log_dir: Optional[str] = "logs",
        max_tool_output_chars: int = 2500,
        max_reasoning_steps: int = 7,
    ):
        super().__init__(
            model=model,
            tools=tools,
            checkpointer=checkpointer,
            system_prompt=system_prompt,
            extract_prompt=extract_prompt,
            process_policy_prompt=process_policy_prompt,
            tag_embedder=tag_embedder,
            memory_retriever=memory_retriever,
            retrieval_top_k=retrieval_top_k,
            similarity_threshold=similarity_threshold,
            log_tools=log_tools,
            log_dir=log_dir,
            max_tool_output_chars=max_tool_output_chars,
            max_reasoning_steps=max_reasoning_steps,
        )

    def extract_tags(self, state: AgentStateV11) -> Dict[str, Any]:
        meta = state.get("instance_meta", {}) or {}
        view_position = str(meta.get("view_position", "")).strip()
        tags = {
            "task_type": ["single_image_findings_generation"],
            "view_position": [view_position] if view_position else [],
            "output_format": ["findings_only"],
        }
        return {"tags": tags}

    def process_request(self, state: AgentStateV11) -> Dict[str, Any]:
        reasoning_steps = state.get("reasoning_steps", 0) + 1
        messages = list(state.get("messages", []))
        prefix_messages: List[Any] = []
        if self.system_prompt:
            prefix_messages.append(SystemMessage(content=self.system_prompt))

        memory_block = self._format_memory_block(state.get("retrieved_memories", []))
        latest_message = messages[-1] if messages else None
        should_attach_memory = isinstance(latest_message, HumanMessage)
        if memory_block and should_attach_memory:
            if self.process_policy_prompt:
                prefix_messages.append(HumanMessage(content=self.process_policy_prompt))
            prefix_messages.append(HumanMessage(content=memory_block))

        is_final_step = reasoning_steps >= self.max_reasoning_steps
        if is_final_step:
            force_msg = HumanMessage(
                content=(
                    f"[Reasoning step limit reached ({self.max_reasoning_steps})] "
                    "You must now provide the final findings text only for this single chest X-ray image. "
                    "Do not call more tools. Do not output an impression, diagnosis list, bullets, or markdown. "
                    "An optional 'FINDINGS:' header is allowed."
                )
            )
            response = self.extract_model.invoke(prefix_messages + messages + [force_msg])
        else:
            response = self.process_model.invoke(prefix_messages + messages)

        update: Dict[str, Any] = {"messages": [response], "reasoning_steps": reasoning_steps}
        if isinstance(response, AIMessage) and not response.tool_calls:
            update["final_answer"] = str(response.content or "").strip()
        return update

    def score_tool_contributions(
        self,
        *,
        state_snapshot: AgentStateV11,
        ground_truth: str = "",
        scorer_model: Optional[BaseLanguageModel] = None,
    ) -> Dict[str, Any]:
        del ground_truth
        tool_traces = state_snapshot.get("tool_traces", [])
        instance_meta = state_snapshot.get("instance_meta", {})

        final_answer = state_snapshot.get("final_answer", "")
        if not final_answer and state_snapshot.get("messages"):
            last_msg = state_snapshot["messages"][-1]
            if isinstance(last_msg, AIMessage):
                final_answer = str(last_msg.content or "")

        task_prompt = self._extract_question_text(state_snapshot)
        reasoning_trace = self._extract_reasoning_trace(state_snapshot)
        model = scorer_model or self.extract_model

        if not tool_traces:
            return self._zero_score_fallback(tool_traces, instance_meta)

        try:
            scored_tools = self._score_all_tools(
                tool_traces=tool_traces,
                task_prompt=task_prompt,
                reasoning_trace=reasoning_trace,
                final_report_text=str(final_answer or "").strip(),
                instance_meta=instance_meta,
                model=model,
            )
        except Exception:
            return self._zero_score_fallback(tool_traces, instance_meta)

        return {
            "case_id": str(instance_meta.get("case_id", "")),
            "instance_id": str(
                instance_meta.get("instance_id", instance_meta.get("question_id", ""))
            ),
            "tools": scored_tools,
        }

    def _score_all_tools(
        self,
        *,
        tool_traces: List[Dict[str, Any]],
        task_prompt: str,
        reasoning_trace: str,
        final_report_text: str,
        instance_meta: Dict[str, Any],
        model: BaseLanguageModel,
    ) -> list:
        del instance_meta
        n_tools = max(len(tool_traces), 1)
        per_tool = max(150, min(600, 1800 // n_tools))

        traces_text_parts = []
        for trace in tool_traces:
            tool_output_snippet = self._truncate(str(trace.get("normalized_output", "")), per_tool)
            traces_text_parts.append(
                f"[tool_id={trace.get('tool_id')}]\n"
                f"tool_name: {trace.get('tool_name', '')}\n"
                f"args: {self._safe_dump(trace.get('args', {}))}\n"
                f"tool_output: {tool_output_snippet}"
            )
        traces_text = "\n\n".join(traces_text_parts)
        tool_ids = [str(t.get("tool_id")) for t in tool_traces]
        reasoning_snippet = self._head_tail_truncate(reasoning_trace, 1800, 1200)
        final_report_snippet = self._truncate(final_report_text, 1500)

        prompt = (
            "You are evaluating all tool calls made by a medical AI agent on a chest X-ray findings generation task.\n"
            "Do NOT determine ground-truth findings. Evaluate only whether each tool output was useful for generating the final findings text.\n\n"
            "=== Task ===\n"
            f"{task_prompt}\n\n"
            "=== Tool Calls ===\n"
            f"{traces_text}\n\n"
            "=== Agent Reasoning Trace ===\n"
            f"{reasoning_snippet}\n\n"
            "=== Agent Final Findings Text ===\n"
            f"{final_report_snippet}\n\n"
            "For each tool call, assign FIVE independent sub-scores, each in [-1, 1].\n"
            "Score each sub-criterion independently before moving to the next.\n\n"
            "--- MOQ: Medical Output Quality (3 sub-scores) ---\n\n"
            "moq_relevance — Is the tool output medically relevant to chest X-ray findings generation for this image?\n"
            "moq_increment — Does the output add concrete information beyond the generic task prompt?\n"
            "moq_diagnostic — Does the output help identify, support, or exclude clinically meaningful findings?\n\n"
            "--- RCC: Reasoning Chain Coherence (2 sub-scores) ---\n\n"
            "rcc_citation — Does the agent explicitly use or paraphrase this tool output in its reasoning or final findings?\n"
            "rcc_consistency — Is the tool output logically consistent with the final findings text?\n\n"
            "Guidance:\n"
            "  1.0 = strongly positive contribution\n"
            "  0.5 = partially useful\n"
            "  0.0 = neutral or unused\n"
            " -0.5 = somewhat misleading\n"
            " -1.0 = strongly misleading or failed\n\n"
            f"Return JSON only, no markdown, with exactly these tool_ids: {tool_ids}:\n"
            "{\"tools\": [{"
            "\"tool_id\": <int>, "
            "\"moq_relevance\": <float -1 to 1>, "
            "\"moq_increment\": <float -1 to 1>, "
            "\"moq_diagnostic\": <float -1 to 1>, "
            "\"rcc_citation\": <float -1 to 1>, "
            "\"rcc_consistency\": <float -1 to 1>, "
            "\"rationale\": \"<brief justification>\""
            "}]}"
        )

        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a strict evaluator for medical-agent tool reliability. "
                        "Return JSON only."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        parsed = self._safe_parse_json(response.content)
        raw_tools = parsed.get("tools", []) if isinstance(parsed, dict) else []

        def _clamp(v: Any) -> float:
            try:
                return max(-1.0, min(1.0, float(v)))
            except Exception:
                return 0.0

        score_by_id = {}
        for item in raw_tools:
            if not isinstance(item, dict):
                continue
            try:
                tid = int(item.get("tool_id", -1))
            except Exception:
                continue
            moq_relevance = _clamp(item.get("moq_relevance", 0.0))
            moq_increment = _clamp(item.get("moq_increment", 0.0))
            moq_diagnostic = _clamp(item.get("moq_diagnostic", 0.0))
            rcc_citation = _clamp(item.get("rcc_citation", 0.0))
            rcc_consistency = _clamp(item.get("rcc_consistency", 0.0))
            moq_score = round((moq_relevance + moq_increment + moq_diagnostic) / 3, 4)
            rcc_score = round((rcc_citation + rcc_consistency) / 2, 4)
            score = round((moq_score + rcc_score) / 2, 4)
            rationale = str(item.get("rationale", ""))[:600]
            score_by_id[tid] = (
                score,
                moq_score,
                rcc_score,
                moq_relevance,
                moq_increment,
                moq_diagnostic,
                rcc_citation,
                rcc_consistency,
                rationale,
            )

        scored_tools = []
        for trace in tool_traces:
            tid = trace["tool_id"]
            if tid in score_by_id:
                (
                    score,
                    moq_score,
                    rcc_score,
                    moq_relevance,
                    moq_increment,
                    moq_diagnostic,
                    rcc_citation,
                    rcc_consistency,
                    rationale,
                ) = score_by_id[tid]
            else:
                (
                    score,
                    moq_score,
                    rcc_score,
                    moq_relevance,
                    moq_increment,
                    moq_diagnostic,
                    rcc_citation,
                    rcc_consistency,
                    rationale,
                ) = (0.0,) * 8 + ("fallback_zero_score",)
            scored_tools.append(
                {
                    "tool_id": tid,
                    "tool_name": trace["tool_name"],
                    "args": trace.get("args", {}),
                    "moq_relevance": moq_relevance,
                    "moq_increment": moq_increment,
                    "moq_diagnostic": moq_diagnostic,
                    "moq_score": moq_score,
                    "rcc_citation": rcc_citation,
                    "rcc_consistency": rcc_consistency,
                    "rcc_score": rcc_score,
                    "score": score,
                    "rationale": rationale,
                }
            )
        return scored_tools

    def _format_memory_block(self, memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return ""
        lines = [
            "Retrieved similar prior cases for tool reliability reference only (not ground truth findings):",
        ]
        for idx, memory in enumerate(memories, start=1):
            case_id = memory.get("case_id", "unknown_case")
            instance_id = memory.get("instance_id", memory.get("question_id", "unknown_instance"))
            tools = memory.get("tools", [])
            lines.append(f"[Ref-{idx}] case={case_id}, instance={instance_id}")
            if isinstance(tools, list) and tools:
                for tool in tools:
                    tool_name = tool.get("tool_name", "unknown_tool")
                    score = tool.get("score", "NA")
                    args = self._safe_dump(self._filter_args_for_display(tool.get("args", {})))
                    lines.append(f"  - tool={tool_name}, args={args}, score={score}")
        return "\n".join(lines)

    def _extract_answer_letter(self, text: Any) -> Optional[str]:
        del text
        return None
