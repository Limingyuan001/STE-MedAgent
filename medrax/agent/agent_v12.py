import operator
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Protocol, TypedDict

from dotenv import load_dotenv
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph

from .agent_v11 import AgentV11, DEFAULT_PROCESS_POLICY, ToolTraceV10

_ = load_dotenv()


class AgentStateV12(TypedDict, total=False):
    """Agent state for question-embedding retrieval."""

    messages: Annotated[List[AnyMessage], operator.add]
    instance_meta: Dict[str, Any]
    query_text: str
    query_embedding: Optional[List[float]]
    retrieved_memories: List[Dict[str, Any]]
    tool_traces: Annotated[List[ToolTraceV10], operator.add]
    final_answer: str
    mode: str
    reasoning_steps: int


class QueryEmbedder(Protocol):
    def __call__(self, query_text: str, state: AgentStateV12) -> Optional[List[float]]:
        ...


class MemoryRetriever(Protocol):
    def __call__(
        self,
        query_text: str,
        embedding: Optional[List[float]],
        top_k: int,
        similarity_threshold: float,
        state: AgentStateV12,
    ) -> List[Dict[str, Any]]:
        ...


class AgentV12(AgentV11):
    """Question-embedding variant of AgentV11 without tag extraction."""

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: List[BaseTool],
        *,
        checkpointer: Any = None,
        system_prompt: str = "",
        extract_prompt: str = "",
        process_policy_prompt: str = DEFAULT_PROCESS_POLICY,
        query_embedder: Optional[QueryEmbedder] = None,
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
        self.query_embedder = query_embedder
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

        workflow = StateGraph(AgentStateV12)
        workflow.add_node("retrieve", self.retrieve_memory)
        workflow.add_node("process", self.process_request)
        workflow.add_node("execute", self.execute_tools)
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
        workflow.set_entry_point("retrieve")
        if checkpointer is None:
            self.workflow = workflow.compile()
        else:
            self.workflow = workflow.compile(checkpointer=checkpointer)

    @staticmethod
    def _build_query_text(instance_meta: Optional[Dict[str, Any]] = None) -> str:
        meta = instance_meta or {}
        question_text = str(meta.get("question_text", "") or "").strip()
        options = meta.get("options")

        option_lines: List[str] = []
        if isinstance(options, dict):
            for key, value in options.items():
                option_lines.append(f"{key}: {value}")
        elif isinstance(options, list):
            for idx, value in enumerate(options):
                letter = chr(ord("A") + idx)
                option_lines.append(f"{letter}: {value}")

        if question_text and option_lines:
            return f"Question:\n{question_text}\n\nOptions:\n" + "\n".join(option_lines)
        if question_text:
            return question_text
        return "\n".join(option_lines)

    @staticmethod
    def build_initial_state(
        messages: List[AnyMessage],
        *,
        instance_meta: Optional[Dict[str, Any]] = None,
        mode: str = "inference",
    ) -> AgentStateV12:
        meta = instance_meta or {}
        return {
            "messages": messages,
            "instance_meta": meta,
            "mode": mode,
            "query_text": AgentV12._build_query_text(meta),
            "tool_traces": [],
            "retrieved_memories": [],
            "reasoning_steps": 0,
        }

    def retrieve_memory(self, state: AgentStateV12) -> Dict[str, Any]:
        query_text = str(state.get("query_text", "") or "").strip()
        if not query_text:
            query_text = self._build_query_text(state.get("instance_meta", {}))

        embedding: Optional[List[float]] = None
        if self.query_embedder is not None:
            try:
                embedding = self.query_embedder(query_text, state)
            except Exception:
                embedding = None

        retrieved_memories: List[Dict[str, Any]] = []
        if self.memory_retriever is not None:
            try:
                retrieved_memories = self.memory_retriever(
                    query_text=query_text,
                    embedding=embedding,
                    top_k=self.retrieval_top_k,
                    similarity_threshold=self.similarity_threshold,
                    state=state,
                )
            except Exception:
                retrieved_memories = []

        return {
            "query_text": query_text,
            "query_embedding": embedding,
            "retrieved_memories": retrieved_memories,
        }
