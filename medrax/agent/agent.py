import json
import operator
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, TypedDict, Annotated, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

_ = load_dotenv()


class ToolCallLog(TypedDict):
    """
    A TypedDict representing a log entry for a tool call.

    Attributes:
        timestamp (str): The timestamp of when the tool call was made.
        tool_call_id (str): The unique identifier for the tool call.
        name (str): The name of the tool that was called.
        args (Any): The arguments passed to the tool.
        content (str): The content or result of the tool call.
    """

    timestamp: str
    tool_call_id: str
    name: str
    args: Any
    content: str


class AgentState(TypedDict):
    """
    A TypedDict representing the state of an agent.

    Attributes:
        messages (Annotated[List[AnyMessage], operator.add]): A list of messages
            representing the conversation history. The operator.add annotation
            indicates that new messages should be appended to this list.
    """

    messages: Annotated[List[AnyMessage], operator.add]


class Agent:
    """
    A class representing an agent that processes requests and executes tools based on
    language model responses.

    Attributes:
        model (BaseLanguageModel): The language model used for processing.
        tools (Dict[str, BaseTool]): A dictionary of available tools.
        checkpointer (Any): Manages and persists the agent's state.
        system_prompt (str): The system instructions for the agent.
        workflow (StateGraph): The compiled workflow for the agent's processing.
        log_tools (bool): Whether to log tool calls.
        log_path (Path): Path to save tool call logs.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: List[BaseTool],
        checkpointer: Any = None,  # todo 这个不知道怎么使用，似乎和state的管理有关。
        system_prompt: str = "",
        log_tools: bool = True,
        log_dir: Optional[str] = "logs",
    ):
        """
        Initialize the Agent.

        Args:
            model (BaseLanguageModel): The language model to use.
            tools (List[BaseTool]): A list of available tools.
            checkpointer (Any, optional): State persistence manager. Defaults to None.
            system_prompt (str, optional): System instructions. Defaults to "".
            log_tools (bool, optional): Whether to log tool calls. Defaults to True.
            log_dir (str, optional): Directory to save logs. Defaults to 'logs'.
        """
        self.system_prompt = system_prompt
        self.log_tools = log_tools

        if self.log_tools:
            self.log_path = Path(log_dir or "logs")
            self.log_path.mkdir(exist_ok=True)

        # Define the agent workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("process", self.process_request)
        workflow.add_node("execute", self.execute_tools)
        workflow.add_conditional_edges(
            "process", self.has_tool_calls, {True: "execute", False: END}  # todo 如果需要加复杂的条件逻辑就是从这个地方加，可以搞多情况的，然后还能用子图当节点？？？？。
        )
        workflow.add_edge("execute", "process")
        workflow.set_entry_point("process")

        self.workflow = workflow.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)  # 它是一个 绑定了 tools 的 Runnable/LLM 包装器。在 LangChain 生态里，这类 invoke(messages) 通常会返回：AIMessage，内含有tool_calls信息：一个字典，包含tools名字（如chest_xray_expert），args（图像路径与prompt（要用工具干啥）或许不同工具args不一样吧，这里是一），id。和类型（就叫'tool_call'）。

    def process_request(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        """
        Process the request using the language model.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[AnyMessage]]: A dictionary containing the model's response.
        """
        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages  # 把系统设定的prompt用SystemMessage函数处理后存在messages的最前面
        response = self.model.invoke(messages)  # 这个时用来分析是否使用工具，不用的话结果是啥，用的话。# 由于self.model = model.bind_tools(tools)，所以这返回结果的AIMessage中一定有
        # todo HumanMessage中在此前加入了一句指令要求回答问题的固定的prompt：Answer this question correctly using chain of thought reasoning and carefully evaluating choices.(这个提示词其实会导致模型一定要分析然后再调用，可能会导致大量token浪费，规划了五六步最后就是调用一个报告生成专家) Solve using our own vision and reasoning and thenuse tools to complement your reasoning. Trust your own judgement over any tools.
        return {"messages": [response]}

    def has_tool_calls(self, state: AgentState) -> bool:
        """
        Check if the response contains any tool calls.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            bool: True if tool calls exist, False otherwise.
        """
        response = state["messages"][-1]
        return len(response.tool_calls) > 0  # todo 这个tool_calls似乎时只有AiMessage才能调用？用于核实调用啥工具？？？

    def execute_tools(self, state: AgentState) -> Dict[str, List[ToolMessage]]:
        """
        Execute tool calls from the model's response.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[ToolMessage]]: A dictionary containing tool execution results.
        """
        tool_calls = state["messages"][-1].tool_calls
        results = []

        for call in tool_calls:
            print(f"Executing tool: {call}")
            if call["name"] not in self.tools:
                print("\n....invalid tool....")
                result = "invalid tool, please retry"
                call_args = call.get("args")  #就是这个地方，有可能出现解析错误的情况无法调用工具的情况。
            else:
                call_args = self._rewrite_tool_args(call.get("args"))
                result = self.tools[call["name"]].invoke(call_args)
            
            # Format tool result for VLM
            # Most tools return (output, metadata) tuple, extract just the output
            if isinstance(result, tuple) and len(result) == 2:
                output, metadata = result
                # Convert output dict to readable format
                if isinstance(output, dict):
                    formatted_result = self._format_tool_output(output, call["name"])
                else:
                    formatted_result = str(output)
            else:
                formatted_result = str(result)

            results.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    name=call["name"],
                    args=call_args,
                    content=formatted_result,
                )
            )

        self._save_tool_calls(results)
        print("Returning to model processing!")

        return {"messages": results}

    def _rewrite_tool_args(self, args: Any) -> Any:
        if not isinstance(args, dict):
            return args

        updated = dict(args)
        
        # Auto-parse JSON strings to actual data structures
        # VLMs sometimes encode lists/dicts as JSON strings
        for key, value in list(updated.items()):
            if isinstance(value, str) and value.strip().startswith(('[', '{')):
                try:
                    import json
                    updated[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    # If parsing fails, keep original string
                    pass
        
        # Resolve image paths
        if isinstance(updated.get("image_path"), str):
            updated["image_path"] = self._resolve_image_path(updated["image_path"])

        image_paths = updated.get("image_paths")
        if isinstance(image_paths, list):
            updated["image_paths"] = [
                self._resolve_image_path(path) if isinstance(path, str) else path
                for path in image_paths
            ]

        return updated

    def _format_tool_output(self, output: dict, tool_name: str) -> str:
        """Format tool output into a readable string for VLM."""
        # Handle error cases
        if "error" in output:
            return f"Error: {output['error']}"
        
        # Handle specific tool outputs
        if "response" in output:
            # For VQA/QA tools, return just the response text
            return output["response"]
        
        if "predictions" in output:
            # For grounding tools
            import json
            return json.dumps(output, indent=2)
        
        if "metrics" in output and "segmentation_image_path" in output:
            # For segmentation tools
            import json
            return (f"Segmentation completed successfully.\n"
                   f"Visualization saved to: {output['segmentation_image_path']}\n"
                   f"Metrics:\n{json.dumps(output['metrics'], indent=2)}")
        
        # For classification/report tools or unknown format
        import json
        return json.dumps(output, indent=2, default=str)
    
    def _resolve_image_path(self, image_path: str) -> str:
        if not image_path:
            return image_path

        if os.path.isabs(image_path) and os.path.exists(image_path):
            return image_path

        image_paths_json = os.getenv("MEDRAX_IMAGE_PATHS")
        if image_paths_json:
            try:
                import json
                available_images = json.loads(image_paths_json)
                if isinstance(available_images, list):
                    for img_path in available_images:
                        if os.path.exists(img_path):
                            if os.path.basename(img_path).lower() in image_path.lower():
                                return img_path
                    if available_images and os.path.exists(available_images[0]):
                        return available_images[0]
            except Exception:
                pass

        figures_dir = os.getenv("MEDRAX_FIGURES_DIR")
        case_id = os.getenv("MEDRAX_CASE_ID")
        candidates: List[str] = []

        if figures_dir:
            if image_path.startswith(("figures/", "figures\\")):
                rel = image_path.split("/", 1)[1] if "/" in image_path else image_path.split("\\", 1)[1]
                candidates.append(os.path.join(figures_dir, rel))

            base_name = os.path.basename(image_path)
            base_lower = base_name.lower()
            if case_id:
                candidates.append(os.path.join(figures_dir, str(case_id), base_name))
                if base_lower != base_name:
                    candidates.append(os.path.join(figures_dir, str(case_id), base_lower))
                if base_lower.endswith(".png"):
                    candidates.append(
                        os.path.join(figures_dir, str(case_id), base_lower[:-4] + ".jpg")
                    )
                elif base_lower.endswith(".jpg"):
                    candidates.append(
                        os.path.join(figures_dir, str(case_id), base_lower[:-4] + ".png")
                    )

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate

        return image_path

    def _save_tool_calls(self, tool_calls: List[ToolMessage]) -> None:
        """
        Save tool calls to a JSON file with timestamp-based naming.

        Args:
            tool_calls (List[ToolMessage]): List of tool calls to save.
        """
        if not self.log_tools:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_path / f"tool_calls_{timestamp}.json"

        logs: List[ToolCallLog] = []
        for call in tool_calls:
            log_entry = {
                "tool_call_id": call.tool_call_id,
                "name": call.name,
                "args": call.args,
                "content": call.content,
                "timestamp": datetime.now().isoformat(),
            }
            logs.append(log_entry)

        with open(filename, "w") as f:
            json.dump(logs, f, indent=4)
