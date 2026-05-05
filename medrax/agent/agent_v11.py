import json
import operator
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    TypedDict,
)

from dotenv import load_dotenv
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph

_ = load_dotenv()

ALLOWED_QUESTION_TYPES = [
    "Detection",
    "Classification",
    "Localization",
    "Comparison",
    "Relationship",
    "Diagnosis",
    "Characterization",
]

ALLOWED_PATHOLOGICAL_FINDINGS = [
    "Mass",
    "Effusion",
    "Pleural Effusion",
    "Consolidation",
    "Nodule",
    "Calcification",
    "Pneumothorax",
    "Lymphadenopathy",
    "Pneumonia",
    "Emphysema",
    "Interstitial findings",
    "Bronchiectasis",
    "Atelectasis",
    "Fibrosis",
    "Edema",
    "Cavitation",
    "Fracture",
    "Tuberculosis",
    "Metastasis",
    "Cardiomegaly",
]
ALLOWED_PATHOLOGICAL_FINDINGS_SET = set(ALLOWED_PATHOLOGICAL_FINDINGS)

PATHOLOGICAL_FINDING_ALIAS_TO_CANONICAL = {
    "mass": "Mass",
    "effusion": "Effusion",
    "pleural effusion": "Pleural Effusion",
    "consolidation": "Consolidation",
    "nodule": "Nodule",
    "calcification": "Calcification",
    "pneumothorax": "Pneumothorax",
    "lymphadenopathy": "Lymphadenopathy",
    "pneumonia": "Pneumonia",
    "emphysema": "Emphysema",
    "interstitial findings": "Interstitial findings",
    "interstitial finding": "Interstitial findings",
    "interstitial opacity": "Interstitial findings",
    "interstitial opacities": "Interstitial findings",
    "interstitial infiltrate": "Interstitial findings",
    "interstitial infiltrates": "Interstitial findings",
    "bronchiectasis": "Bronchiectasis",
    "atelectasis": "Atelectasis",
    "fibrosis": "Fibrosis",
    "edema": "Edema",
    "oedema": "Edema",
    "cavitation": "Cavitation",
    "cavity": "Cavitation",
    "cavitary lesion": "Cavitation",
    "fracture": "Fracture",
    "fractures": "Fracture",
    "tuberculosis": "Tuberculosis",
    "tb": "Tuberculosis",
    "metastasis": "Metastasis",
    "metastases": "Metastasis",
    "cardiomegaly": "Cardiomegaly",
    "cardiac enlargement": "Cardiomegaly",
    "enlarged cardiac silhouette": "Cardiomegaly",
}
DEFAULT_EXTRACT_PROMPT = """You are a medical tagging assistant for chest imaging MCQ questions.
Extract tags from the question stem and options as JSON with this exact schema:
{
  "question_type": ["..."],
  "symptoms": ["..."],
  "demographics": ["..."],
  "risk_factors": ["..."],
  "pathological_findings": ["..."],
  "denied_information": {
    "symptoms": ["..."],
    "pathological_findings": ["..."],
    "risk_factors": ["..."]
  }
}

question_type must be a subset of exactly these 7 labels (use these exact names):
- Detection: Identifying specific findings. (e.g., "Is there a nodule present in the right upper lobe?")
- Classification: Classifying specific findings. (e.g., "Is this mass benign or malignant in appearance?")
- Localization: Precise positioning of findings. (e.g., "In which bronchopulmonary segment is the mass located?")
- Comparison: Analyzing relative sizes and positions. (e.g., "How has the pleural effusion volume changed compared to prior imaging?")
- Relationship: Understanding relationship of findings. (e.g., "Does the mediastinal lymphadenopathy correlate with the lung mass?")
- Diagnosis: Interpreting findings for clinical decisions. (e.g., "Given the CXR, what is the likely diagnosis?")
- Characterization: Describing specific finding attributes. (e.g., "What are the margins of the nodule - smooth, spiculated, or irregular?")

Extraction rules:
1) pathological_findings must be selected only from this closed set of 20 labels:
   ["Mass","Effusion","Pleural Effusion","Consolidation","Nodule","Calcification","Pneumothorax","Lymphadenopathy","Pneumonia","Emphysema","Interstitial findings","Bronchiectasis","Atelectasis","Fibrosis","Edema","Cavitation","Fracture","Tuberculosis","Metastasis","Cardiomegaly"].
2) Extract pathological_findings from question stem and options only (text evidence only; no image inference).
3) If synonym/variant appears, map to the closest canonical label from the 20 labels above.
4) Do not output any finding outside this 20-label set; deduplicate output.
5) Symptoms are subjective, patient-reported experiences indicating illness or injury, and are not directly observable by others (e.g., pain, nausea, fatigue, dizziness, headache, cough, fever, chest pain, dyspnea).
6) Demographics are statistical attributes describing people, such as age, sex/gender, pregnancy status, ethnicity, occupation, education, or income, when explicitly stated and clinically relevant.
7) Risk factors are characteristics, conditions, exposures, or behaviors that increase adverse outcome likelihood (e.g., smoking, TB exposure, immunosuppression, alcohol misuse, occupational dust exposure, family history).
8) denied_information must contain only explicitly negated items from question stem for symptoms, pathological_findings, and risk_factors (e.g., "no history of smoking", "afebrile", "no chest pain", "without pleural effusion"). Never infer negation.
9) denied_information.pathological_findings must use the same 20 canonical pathological finding labels.
10) Keep all lists concise and deduplicated. If no evidence is present for a field, return an empty list.

Return JSON only. No markdown.
"""

DEFAULT_PROCESS_POLICY = """Medical tools are not perfectly reliable.
The retrieved references below come from similar past questions and provide tool reliability guidance.
Each reference is for reliability guidance on the latest HumanMessage in this turn, not ground-truth answers.
Tool entries include usage arguments (`args`) and contribution scores (`score`):
- `score > 0`: this tool was found helpful in similar past questions.
- `score < 0`: this tool was found misleading or low-quality in similar past questions.
- Higher `score`: generally more reliable under the current question context.
Not every question will retrieve suitable references; continue reasoning normally when references are missing.
When there is a conflict between the results of different tools, give priority to the ones with higher scores.
"""

class ToolTraceV10(TypedDict):
    tool_id: int
    timestamp: str
    tool_call_id: str
    tool_name: str
    args: Any
    raw_output: str
    normalized_output: str

class AgentStateV11(TypedDict, total=False):
    """Agent 在图节点间共享的可选状态字典。"""
    messages: Annotated[List[AnyMessage], operator.add]
    instance_meta: Dict[str, Any]
    tags: Dict[str, Any]
    tag_embedding: Optional[List[float]]
    retrieved_memories: List[Dict[str, Any]]
    tool_traces: Annotated[List[ToolTraceV10], operator.add]
    final_answer: str
    mode: str
    reasoning_steps: int  # counts process_request entries per question; reset per question


class TagEmbedder(Protocol):
    def __call__(self, tags: Dict[str, Any], state: AgentStateV11) -> Optional[List[float]]:
        ...


class MemoryRetriever(Protocol):
    def __call__(
        self,
        tags: Dict[str, Any],
        embedding: Optional[List[float]],
        top_k: int,
        similarity_threshold: float,
        state: AgentStateV11,
    ) -> List[Dict[str, Any]]:
        ...


class AgentV11:
    """
    MedRAX v11 agent — extends v10 with a per-question reasoning step limit.

    Changes vs AgentV10:
    - AgentStateV11 adds `reasoning_steps: int` field (non-annotated, so it is
      replaced — not appended — on each state update, enabling per-question reset).
    - __init__ adds `max_reasoning_steps: int = 7`.
    - process_request increments the counter and, on the final step, invokes
      extract_model (no tools bound) with a forcing prompt so the model must
      produce a text-only final answer.  has_tool_calls() then naturally routes
      to END without any extra logic.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: List[BaseTool],
        *,
        checkpointer: Any = None,
        system_prompt: str = "",
        extract_prompt: str = DEFAULT_EXTRACT_PROMPT,
        process_policy_prompt: str = DEFAULT_PROCESS_POLICY,
        tag_embedder: Optional[TagEmbedder] = None,
        memory_retriever: Optional[MemoryRetriever] = None,
        retrieval_top_k: int = 3,
        similarity_threshold: float = 0.3,
        log_tools: bool = True,
        log_dir: Optional[str] = "logs",
        max_tool_output_chars: int = 2500,
        max_reasoning_steps: int = 7,
    ):
        self.system_prompt = system_prompt
        self.extract_prompt = extract_prompt
        self.process_policy_prompt = process_policy_prompt
        self.retrieval_top_k = retrieval_top_k
        self.similarity_threshold = similarity_threshold
        self.tag_embedder = tag_embedder
        self.memory_retriever = memory_retriever
        self.max_tool_output_chars = max_tool_output_chars
        self.log_tools = log_tools
        self.max_reasoning_steps = max_reasoning_steps

        self.tools = {tool.name: tool for tool in tools}
        self.process_model = model.bind_tools(tools)
        self.extract_model = model

        if self.log_tools:
            self.log_path = Path(log_dir or "logs")
            self.log_path.mkdir(exist_ok=True)

        workflow = StateGraph(AgentStateV11)
        workflow.add_node("extract", self.extract_tags)
        workflow.add_node("retrieve", self.retrieve_memory)
        workflow.add_node("process", self.process_request)
        workflow.add_node("execute", self.execute_tools)
        workflow.add_edge("extract", "retrieve")
        workflow.add_edge("retrieve", "process")
        workflow.add_conditional_edges(
            "process",
            self.has_tool_calls,
            {
                True: "execute",
                False: END,
            },
        )
        workflow.add_edge("execute", "process")
        workflow.set_entry_point("extract")
        if checkpointer is None:
            self.workflow = workflow.compile()
        else:
            self.workflow = workflow.compile(checkpointer=checkpointer)

    @staticmethod
    def build_initial_state(
        messages: List[AnyMessage],
        *,
        instance_meta: Optional[Dict[str, Any]] = None,
        mode: str = "inference",
    ) -> AgentStateV11:
        return {
            "messages": messages,
            "instance_meta": instance_meta or {},
            "mode": mode,
            "tool_traces": [],
            "retrieved_memories": [],
            "reasoning_steps": 0,  # reset per question (non-annotated field → replaced, not appended)
        }

    def extract_tags(self, state: AgentStateV11) -> Dict[str, Any]:
        question_text = self._extract_question_text(state)
        options_text = self._extract_options_text(state)
        user_prompt = (
            "Question:\n"
            f"{question_text}\n\n"
            "Options:\n"
            f"{options_text}\n\n"
            "Return JSON only."
        )
        response = self.extract_model.invoke(
            [
                SystemMessage(content=self.extract_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        tags = self._normalize_tags(
            self._safe_parse_json(response.content),
            question_stem=question_text,
        )
        return {"tags": tags}

    def retrieve_memory(self, state: AgentStateV11) -> Dict[str, Any]:
        tags = state.get("tags", {})
        embedding: Optional[List[float]] = None
        if self.tag_embedder is not None:
            try:
                embedding = self.tag_embedder(tags, state)
            except Exception:
                embedding = None

        retrieved_memories: List[Dict[str, Any]] = []
        if self.memory_retriever is not None:
            try:
                retrieved_memories = self.memory_retriever(
                    tags=tags,
                    embedding=embedding,
                    top_k=self.retrieval_top_k,
                    similarity_threshold=self.similarity_threshold,
                    state=state,
                )
            except Exception:
                retrieved_memories = []

        return {
            "tag_embedding": embedding,
            "retrieved_memories": retrieved_memories,
        }

    def process_request(self, state: AgentStateV11) -> Dict[str, Any]:
        reasoning_steps = state.get("reasoning_steps", 0) + 1
        messages = list(state.get("messages", []))
        prefix_messages: List[AnyMessage] = []
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
            # Force a text-only final answer by using extract_model (no tools bound).
            # The forcing prompt is included in the invoke context but not stored in state.
            force_msg = HumanMessage(
                content=(
                    f"[Reasoning step limit reached ({self.max_reasoning_steps})] "
                    "You must now provide your final answer without calling any more tools. "
                    "Choose the best option based on your reasoning so far."
                )
            )
            response = self.extract_model.invoke(prefix_messages + messages + [force_msg])
        else:
            response = self.process_model.invoke(prefix_messages + messages)

        update: Dict[str, Any] = {"messages": [response], "reasoning_steps": reasoning_steps}

        if isinstance(response, AIMessage) and not response.tool_calls:
            update["final_answer"] = self._extract_answer_letter(response.content) or str(
                response.content
            ).strip()
        return update

    def has_tool_calls(self, state: AgentStateV11) -> bool:
        if not state.get("messages"):
            return False
        response = state["messages"][-1]
        return bool(getattr(response, "tool_calls", None))

    def execute_tools(self, state: AgentStateV11) -> Dict[str, Any]:
        latest = state["messages"][-1]
        tool_calls = getattr(latest, "tool_calls", []) or []

        tool_messages: List[ToolMessage] = []
        traces: List[ToolTraceV10] = []
        trace_base = len(state.get("tool_traces", []))

        for idx, call in enumerate(tool_calls):
            tool_name = call.get("name")
            tool_call_id = call.get("id", f"tool_call_{trace_base + idx + 1}")
            call_args = self._rewrite_tool_args(call.get("args"))

            if tool_name not in self.tools:
                raw_result = "invalid tool, please retry"
                normalized = raw_result
            else:
                raw_result = self.tools[tool_name].invoke(call_args)
                normalized = self._format_tool_output(raw_result, tool_name)

            tool_messages.append(
                ToolMessage(
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    args=call_args,
                    content=normalized,
                )
            )

            traces.append(
                ToolTraceV10(
                    tool_id=trace_base + idx + 1,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    tool_call_id=tool_call_id,
                    tool_name=tool_name or "",
                    args=call_args,
                    raw_output=self._safe_dump(raw_result),
                    normalized_output=self._truncate(normalized),
                )
            )

        self._save_tool_calls(tool_messages)
        return {"messages": tool_messages, "tool_traces": traces}

    def score_tool_contributions(
        self,
        *,
        state_snapshot: AgentStateV11,
        ground_truth: str = "",  # accepted for API compatibility, intentionally unused
        scorer_model: Optional[BaseLanguageModel] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate per-call tool contribution in [-1, 1] using a process-based LLM judge.

        Unlike v6, this method does NOT use ground_truth. Each tool call is
        scored on two independent dimensions by a single LLM call:
          1. Medical Output Quality (MOQ): relevance, information increment,
             diagnostic directionality of the tool output itself.
          2. Reasoning Chain Coherence (RCC): whether the agent's reasoning
             explicitly cited the tool output, and whether the output is
             logically consistent with the agent's chosen answer.
        Final score = avg(moq_score, rcc_score). is_correct is not used.
        """
        tool_traces = state_snapshot.get("tool_traces", [])
        instance_meta = state_snapshot.get("instance_meta", {})

        final_answer = state_snapshot.get("final_answer", "")
        if not final_answer and state_snapshot.get("messages"):
            last_msg = state_snapshot["messages"][-1]
            if isinstance(last_msg, AIMessage):
                final_answer = str(last_msg.content or "")
        final_letter = self._extract_answer_letter(final_answer) or ""

        question_text = self._extract_question_text(state_snapshot)
        options_text = self._extract_options_text(state_snapshot)
        reasoning_trace = self._extract_reasoning_trace(state_snapshot)

        model = scorer_model or self.extract_model

        if not tool_traces:
            return self._zero_score_fallback(tool_traces, instance_meta)

        try:
            scored_tools = self._score_all_tools(
                tool_traces=tool_traces,
                question_text=question_text,
                options_text=options_text,
                reasoning_trace=reasoning_trace,
                final_letter=final_letter,
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_all_tools(
        self,
        *,
        tool_traces: List[ToolTraceV10],
        question_text: str,
        options_text: str,
        reasoning_trace: str,
        final_letter: str,
        instance_meta: Dict[str, Any],
        model: BaseLanguageModel,
    ) -> list:
        """Single LLM call to score all tool calls at once. Returns scored_tools list."""
        final_label = final_letter or "(unknown)"

        n_tools = max(len(tool_traces), 1)
        per_tool = max(150, min(600, 1800 // n_tools))

        reasoning_snippet = reasoning_trace

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
        prompt = (
            "You are evaluating all tool calls made by a medical AI agent on a "
            "chest X-ray multiple-choice question.\n"
            "Do NOT attempt to determine the correct answer. Evaluate only tool "
            "output quality and how each was used in the agent's reasoning.\n\n"
            "=== Question ===\n"
            f"{question_text}\n\n"
            "=== Options ===\n"
            f"{options_text}\n\n"
            "=== Tool Calls ===\n"
            f"{traces_text}\n\n"
            "=== Agent Reasoning Trace ===\n"
            f"{reasoning_snippet}\n\n"
            f"=== Agent's Final Answer: {final_label} ===\n\n"
            "For each tool call, assign FIVE independent sub-scores, each in [-1, 1].\n"
            "Score each sub-criterion on its own merits before moving to the next.\n\n"
            "--- MOQ: Medical Output Quality (3 sub-scores) ---\n\n"
            "moq_relevance — Is the tool output medically relevant to the question?\n"
            "  1.0 : directly targets the pathology or task type asked about\n"
            "  0.5 : related to the general clinical domain but only partially on-topic\n"
            "  0.0 : generic or tangential output with no specific relevance\n"
            " -0.5 : off-topic output that may distract from the actual question\n"
            " -1.0 : output addresses a completely different condition or task\n\n"
            "moq_increment — Does the output add information beyond the question text?\n"
            "  1.0 : provides concrete new findings (e.g. measurements, classifications, regions) not in the stem\n"
            "  0.5 : adds some new detail but largely overlaps with what the question already states\n"
            "  0.0 : restates question content without adding anything new\n"
            " -0.5 : introduces information that contradicts or confuses what the stem states\n"
            " -1.0 : output is empty, garbled, or a clear tool failure\n\n"
            "moq_diagnostic — Does the output help narrow the differential diagnosis?\n"
            "  1.0 : clearly supports or excludes one or more plausible diagnoses or options on clinical grounds\n"
            "  0.5 : suggestive toward a diagnostic direction but not conclusive\n"
            "  0.0 : neutral — does not distinguish between competing diagnoses or options\n"
            " -0.5 : contains a clinically incorrect finding that could push reasoning toward a wrong diagnosis\n"
            " -1.0 : asserts a finding directly contradicted by image evidence or clinical context\n\n"
            "--- RCC: Reasoning Chain Coherence (2 sub-scores) ---\n\n"
            "rcc_citation — Does the agent's reasoning explicitly reference this tool output?\n"
            "  1.0 : agent directly quotes or paraphrases specific values/findings from this output\n"
            "  0.5 : agent mentions the tool or its result in passing without detailed use\n"
            "  0.0 : agent does not reference this tool output at all\n"
            " -0.5 : agent acknowledges the tool but explicitly dismisses or contradicts it\n"
            " -1.0 : agent misrepresents the tool output (cites wrong values or inverted findings)\n\n"
            f"rcc_consistency — Is the tool output logically consistent with the agent's answer {final_label}?\n"
            f"  (Do NOT judge whether {final_label} is correct — only judge logical alignment.)\n"
            f"  1.0 : tool output directly supports the clinical reasoning that leads to {final_label}\n"
            f"  0.5 : tool output is compatible with {final_label} but does not strongly support it\n"
            f"  0.0 : tool output is neutral — consistent with multiple options equally\n"
            f" -0.5 : tool output is more consistent with a different option than {final_label}\n"
            f" -1.0 : tool output directly contradicts the clinical basis of {final_label}\n\n"
            f"Return JSON only, no markdown, with exactly these tool_ids: {tool_ids}:\n"
            "{\"tools\": [{"
            "\"tool_id\": <int>, "
            "\"moq_relevance\": <float -1 to 1>, "
            "\"moq_increment\": <float -1 to 1>, "
            "\"moq_diagnostic\": <float -1 to 1>, "
            "\"rcc_citation\": <float -1 to 1>, "
            "\"rcc_consistency\": <float -1 to 1>, "
            "\"rationale\": \"<one sentence per sub-score, five total>\""
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
            moq_relevance  = _clamp(item.get("moq_relevance",  0.0))
            moq_increment  = _clamp(item.get("moq_increment",  0.0))
            moq_diagnostic = _clamp(item.get("moq_diagnostic", 0.0))
            rcc_citation   = _clamp(item.get("rcc_citation",   0.0))
            rcc_consistency= _clamp(item.get("rcc_consistency",0.0))
            moq_score = round((moq_relevance + moq_increment + moq_diagnostic) / 3, 4)
            rcc_score = round((rcc_citation + rcc_consistency) / 2, 4)
            score     = round((moq_score + rcc_score) / 2, 4)
            rationale = str(item.get("rationale", ""))[:600]
            score_by_id[tid] = (score, moq_score, rcc_score,
                                moq_relevance, moq_increment, moq_diagnostic,
                                rcc_citation, rcc_consistency, rationale)

        scored_tools = []
        for trace in tool_traces:
            tid = trace["tool_id"]
            if tid in score_by_id:
                (score, moq_score, rcc_score,
                 moq_relevance, moq_increment, moq_diagnostic,
                 rcc_citation, rcc_consistency, rationale) = score_by_id[tid]
            else:
                (score, moq_score, rcc_score,
                 moq_relevance, moq_increment, moq_diagnostic,
                 rcc_citation, rcc_consistency, rationale) = (0.0,) * 8 + ("fallback_zero_score",)
            scored_tools.append(
                {
                    "tool_id": tid,
                    "tool_name": trace["tool_name"],
                    "args": trace.get("args", {}),
                    "moq_relevance": moq_relevance,
                    "moq_increment": moq_increment,
                    "moq_diagnostic": moq_diagnostic,
                    "moq_score": moq_score,       # avg(moq_relevance, moq_increment, moq_diagnostic)
                    "rcc_citation": rcc_citation,
                    "rcc_consistency": rcc_consistency,
                    "rcc_score": rcc_score,        # avg(rcc_citation, rcc_consistency)
                    "score": score,                # avg(moq_score, rcc_score), used for retrieval ranking
                    "rationale": rationale,
                }
            )

        return scored_tools

    def _extract_reasoning_trace(self, state_snapshot: AgentStateV11) -> str:
        """Concatenate text content from all AIMessage objects in the message history."""
        messages = state_snapshot.get("messages", []) or []
        parts = []
        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = str(item.get("text", "")).strip()
                        if text:
                            parts.append(text)
        return "\n\n".join(parts)

    def _extract_question_text(self, state: AgentStateV11) -> str:
        meta = state.get("instance_meta", {})
        if isinstance(meta.get("question_text"), str) and meta["question_text"].strip():
            return meta["question_text"].strip()
        return self._extract_text_from_messages(state.get("messages", []))

    def _extract_options_text(self, state: AgentStateV11) -> str:
        meta = state.get("instance_meta", {})
        options = meta.get("options")
        if isinstance(options, dict):
            lines = []
            for key, value in options.items():
                lines.append(f"{key}: {value}")
            return "\n".join(lines)
        if isinstance(options, list):
            lines = []
            for idx, value in enumerate(options):
                letter = chr(ord("A") + idx)
                lines.append(f"{letter}: {value}")
            return "\n".join(lines)
        return ""

    def _extract_text_from_messages(self, messages: List[AnyMessage]) -> str:
        for message in reversed(messages):
            if not isinstance(message, HumanMessage):
                continue
            content = getattr(message, "content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(str(part.get("text", "")))
                if text_parts:
                    return "\n".join(text_parts)
        return ""

    @staticmethod
    def _dedupe_keep_order(values: List[str], max_items: Optional[int] = None) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for raw in values:
            text = str(raw).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(text)
            if max_items is not None and len(deduped) >= max_items:
                break
        return deduped

    @staticmethod
    def _term_is_explicitly_negated(term: str, question_stem: str) -> bool:
        if not term or not question_stem:
            return False
        lowered_stem = question_stem.lower()
        lowered_term = term.lower().strip()
        if not lowered_term:
            return False

        escaped_term = re.escape(lowered_term)
        neg_cues = [
            "no",
            "without",
            "absence of",
            "absent",
            "negative for",
            "free of",
            "lack of",
        ]
        for cue in neg_cues:
            patterns = [
                rf"{re.escape(cue)}\s+{escaped_term}",
                rf"{escaped_term}\s+is\s+{re.escape(cue)}",
                rf"{escaped_term}\s*:\s*{re.escape(cue)}",
            ]
            for pattern in patterns:
                if re.search(pattern, lowered_stem):
                    return True
        return False

    @staticmethod
    def _canonicalize_pathological_finding(term: str) -> Optional[str]:
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

    def _normalize_tags(self, data: Any, question_stem: str = "") -> Dict[str, Any]:
        normalized: Dict[str, Any] = {
            "question_type": [],
            "symptoms": [],
            "demographics": [],
            "risk_factors": [],
            "pathological_findings": [],
            "denied_information": {
                "symptoms": [],
                "pathological_findings": [],
                "risk_factors": [],
            },
        }
        if not isinstance(data, dict):
            return normalized

        allowed_type_map = {x.lower(): x for x in ALLOWED_QUESTION_TYPES}
        question_type = data.get("question_type", [])
        if isinstance(question_type, str):
            question_type = [question_type]
        if isinstance(question_type, list):
            cleaned_qt = []
            for item in question_type:
                text = str(item).strip()
                if not text:
                    continue
                mapped = allowed_type_map.get(text.lower())
                if mapped:
                    cleaned_qt.append(mapped)
            normalized["question_type"] = self._dedupe_keep_order(cleaned_qt, max_items=7)

        def _as_list(value: Any) -> List[str]:
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                return []
            return [str(item).strip() for item in value if str(item).strip()]

        auto_denied_symptoms: List[str] = []
        for symptom in _as_list(data.get("symptoms", [])):
            if self._term_is_explicitly_negated(symptom, question_stem):
                auto_denied_symptoms.append(symptom)
            else:
                normalized["symptoms"].append(symptom)
        normalized["symptoms"] = self._dedupe_keep_order(normalized["symptoms"], max_items=30)

        normalized["demographics"] = self._dedupe_keep_order(
            _as_list(data.get("demographics", [])),
            max_items=20,
        )

        auto_denied_risk_factors: List[str] = []
        for risk_factor in _as_list(data.get("risk_factors", [])):
            if self._term_is_explicitly_negated(risk_factor, question_stem):
                auto_denied_risk_factors.append(risk_factor)
            else:
                normalized["risk_factors"].append(risk_factor)
        normalized["risk_factors"] = self._dedupe_keep_order(normalized["risk_factors"], max_items=20)

        findings = data.get("pathological_findings", [])
        if isinstance(findings, str):
            findings = [findings]
        if isinstance(findings, list):
            canonical_findings: List[str] = []
            auto_denied_findings: List[str] = []
            for item in findings:
                text = str(item).strip()
                canonical = self._canonicalize_pathological_finding(text)
                if canonical:
                    if self._term_is_explicitly_negated(text, question_stem) or self._term_is_explicitly_negated(
                        canonical, question_stem
                    ):
                        auto_denied_findings.append(canonical)
                    else:
                        canonical_findings.append(canonical)
            normalized["pathological_findings"] = self._dedupe_keep_order(
                canonical_findings,
                max_items=20,
            )
        else:
            auto_denied_findings = []

        denied_information = data.get("denied_information", {})
        if not isinstance(denied_information, dict):
            denied_information = {}

        denied_symptoms = []
        for item in auto_denied_symptoms + _as_list(denied_information.get("symptoms", [])):
            if self._term_is_explicitly_negated(item, question_stem):
                denied_symptoms.append(item)
        normalized["denied_information"]["symptoms"] = self._dedupe_keep_order(
            denied_symptoms,
            max_items=30,
        )

        denied_risk_factors = []
        for item in auto_denied_risk_factors + _as_list(denied_information.get("risk_factors", [])):
            if self._term_is_explicitly_negated(item, question_stem):
                denied_risk_factors.append(item)
        normalized["denied_information"]["risk_factors"] = self._dedupe_keep_order(
            denied_risk_factors,
            max_items=20,
        )

        denied_pathological_findings: List[str] = []
        denied_findings_raw = _as_list(denied_information.get("pathological_findings", []))
        for item in auto_denied_findings + denied_findings_raw:
            canonical = self._canonicalize_pathological_finding(item)
            if not canonical:
                continue
            if self._term_is_explicitly_negated(item, question_stem) or self._term_is_explicitly_negated(
                canonical, question_stem
            ):
                denied_pathological_findings.append(canonical)
        denied_pathological_findings = self._dedupe_keep_order(
            denied_pathological_findings,
            max_items=20,
        )
        normalized["denied_information"]["pathological_findings"] = denied_pathological_findings
        if denied_pathological_findings:
            denied_set = set(denied_pathological_findings)
            normalized["pathological_findings"] = [
                finding for finding in normalized["pathological_findings"] if finding not in denied_set
            ]
        return normalized

    def _filter_args_for_display(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Remove image-like fields (base64 data or long paths) from args, keep text params."""
        filtered = {}
        for k, v in args.items():
            if not isinstance(v, str):
                filtered[k] = v
                continue
            # Skip base64 blobs (data URIs or raw base64 strings)
            if v.startswith("data:") or len(v) > 300:
                filtered[k] = "<image_data>"
            else:
                filtered[k] = v
        return filtered

    def _format_memory_block(self, memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return ""
        lines = [
            "Retrieved memory references for tool reliability (not ground truth):",
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

    def _zero_score_fallback(
        self, traces: List[ToolTraceV10], instance_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "case_id": str(instance_meta.get("case_id", "")),
            "instance_id": str(instance_meta.get("instance_id", instance_meta.get("question_id", ""))),
            "tools": [
                {
                    "tool_id": trace["tool_id"],
                    "tool_name": trace["tool_name"],
                    "args": trace.get("args", {}),
                    "moq_relevance": 0.0,
                    "moq_increment": 0.0,
                    "moq_diagnostic": 0.0,
                    "moq_score": 0.0,
                    "rcc_citation": 0.0,
                    "rcc_consistency": 0.0,
                    "rcc_score": 0.0,
                    "score": 0.0,
                    "rationale": "fallback_zero_score",
                }
                for trace in traces
            ],
        }

    def _rewrite_tool_args(self, args: Any) -> Any:
        if not isinstance(args, dict):
            return args
        updated = dict(args)
        for key, value in list(updated.items()):
            if isinstance(value, str):
                stripped = value.strip()
                if stripped.startswith(("[", "{")):
                    try:
                        updated[key] = json.loads(stripped)
                    except Exception:
                        updated[key] = value

        if isinstance(updated.get("image_path"), str):
            updated["image_path"] = self._resolve_image_path(updated["image_path"])

        image_paths = updated.get("image_paths")
        if isinstance(image_paths, list):
            updated["image_paths"] = [
                self._resolve_image_path(path) if isinstance(path, str) else path
                for path in image_paths
            ]
        return updated

    def _resolve_image_path(self, image_path: str) -> str:
        if not image_path:
            return image_path
        if os.path.isabs(image_path) and os.path.exists(image_path):
            return image_path

        image_paths_json = os.getenv("MEDRAX_IMAGE_PATHS")
        if image_paths_json:
            try:
                available_images = json.loads(image_paths_json)
                if isinstance(available_images, list):
                    for img_path in available_images:
                        if not isinstance(img_path, str) or not os.path.exists(img_path):
                            continue
                        if os.path.basename(img_path).lower() in image_path.lower():
                            return img_path
                    if available_images and isinstance(available_images[0], str):
                        fallback = available_images[0]
                        if os.path.exists(fallback):
                            return fallback
            except Exception:
                pass

        figures_dir = os.getenv("MEDRAX_FIGURES_DIR")
        case_id = os.getenv("MEDRAX_CASE_ID")
        if figures_dir and case_id:
            candidate = os.path.join(figures_dir, case_id, image_path)
            if os.path.exists(candidate):
                return candidate
        return image_path

    def _format_tool_output(self, output: Any, tool_name: str) -> str:
        if isinstance(output, tuple) and len(output) == 2:
            maybe_payload = output[0]
            if isinstance(maybe_payload, dict):
                return self._format_tool_output(maybe_payload, tool_name)
            return self._truncate(str(maybe_payload))

        if isinstance(output, dict):
            if "error" in output:
                return f"Error: {output['error']}"
            if "response" in output:
                return self._truncate(str(output["response"]))
            if "metrics" in output and "segmentation_image_path" in output:
                return self._truncate(
                    json.dumps(
                        {
                            "segmentation_image_path": output.get("segmentation_image_path"),
                            "metrics": output.get("metrics"),
                        },
                        ensure_ascii=False,
                    )
                )
            return self._truncate(json.dumps(output, ensure_ascii=False, default=str))
        return self._truncate(str(output))

    def _extract_answer_letter(self, text: Any) -> Optional[str]:
        if not isinstance(text, str):
            return None
        stripped = text.strip().upper()
        if stripped in {"A", "B", "C", "D", "E", "F"}:
            return stripped
        match = re.search(r"\b([A-F])\b", stripped)
        if match:
            return match.group(1)
        return None

    def _safe_parse_json(self, content: Any) -> Any:
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        text = str(content or "").strip()
        if not text:
            return {}
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
            text = re.sub(r"```\s*$", "", text).strip()
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            text = match.group(0)
        try:
            return json.loads(text)
        except Exception:
            return {}

    def _save_tool_calls(self, tool_messages: List[ToolMessage]) -> None:
        if not self.log_tools:
            return
        ts = datetime.now().strftime("%Y%m%d")
        log_file = self.log_path / f"tool_calls_{ts}.json"
        entries = []
        for msg in tool_messages:
            entries.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "tool_call_id": msg.tool_call_id,
                    "name": msg.name,
                    "args": msg.args,
                    "content": self._truncate(msg.content),
                }
            )
        with open(log_file, "a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _safe_dump(self, value: Any) -> str:
        try:
            return self._truncate(json.dumps(value, ensure_ascii=False, default=str))
        except Exception:
            return self._truncate(str(value))

    def _truncate(self, text: str, max_chars: Optional[int] = None) -> str:
        limit = max_chars if max_chars is not None else self.max_tool_output_chars
        if len(text) <= limit:
            return text
        return text[:limit] + "...(truncated)"

    def _head_tail_truncate(self, text: str, head: int, tail: int) -> str:
        """Keep the first `head` and last `tail` characters, joining with an ellipsis.
        Used for reasoning traces where initial context and final conclusions both matter.
        """
        if len(text) <= head + tail:
            return text
        return text[:head] + "\n...(middle omitted)...\n" + text[-tail:]
