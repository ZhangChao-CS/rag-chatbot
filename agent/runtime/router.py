from typing import Optional

from agent.plan_utils import format_prior_outputs
from agent.planning.schema import Task, TaskType
from agent.runtime.calculator_router import CalculatorRouter
from agent.runtime.routing_utils import build_routing_guidance
from agent.runtime.types import RouteResult
from agent.state import AgentState
from rag.llm import ask_llm
from rag.utils import parse_llm_json
from tools.registry import ToolRegistry

INFORMATION_TOOLS = ("retrieval", "web_search")

TOOL_SELECT_PROMPT = """
你是 Tool Router。Planner 已给出任务目标，请你选择最合适的 Tool 并构造参数。

系统环境：
{system_context}

用户问题：{question}
任务目标：#{task_id} {task_description}

可用 Tool：
{tool_descriptions}

前置 Task 结构化 output：
{prior_results}

选型指引：
{routing_guidance}

{tool_hint}

输出 JSON：
{{
  "thought": "选择理由（需说明为何该信息应来自本地文档或外部网络）",
  "tool": "retrieval 或 web_search",
  "arguments": {{"query": "完整查询词"}}
}}

规则：
1. 判断依据是「信息来源类型」，不是主题是否专业或是否听起来像新技术
2. 本地知识库已就绪时：理解/查询/对比文档内的概念、方法、术语、论文 → retrieval
3. 实时价格、最新新闻、明确需要联网的公开数据 → web_search
4. 两者都可能时，本地知识库已就绪则优先 retrieval
5. query 应覆盖任务目标的全部信息需求，避免多次检索
6. 禁止输出 calculator
"""


class Router:
    """
    Task → Tool 动态路由。
    - information_collection: LLM 选择 retrieval / web_search
    - computation: CalculatorRouter 确定性生成 expression
    """

    MAX_ROUTE_ATTEMPTS = 3

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.calculator_router = CalculatorRouter()

    def route(self, task: Task, state: AgentState) -> RouteResult:
        if task.task_type == TaskType.COMPUTATION:
            return self._route_computation(task, state)

        return self._route_information(task, state)

    def _route_computation(self, task: Task, state: AgentState) -> RouteResult:
        tool_name = "calculator"
        args = self.calculator_router.build_arguments_or_raise(task, state.plan)

        err = self._validate_arguments(tool_name, args)
        if err:
            raise ValueError(err)

        print(f"[Router] Task #{task.id} → {tool_name} (deterministic)")
        print(f"[Router Arguments] {args}")

        return RouteResult(
            task_id=task.id,
            tool=tool_name,
            arguments=args,
            thought="CalculatorRouter: 从依赖 TaskResult 自动生成 expression",
        )

    def _route_information(self, task: Task, state: AgentState) -> RouteResult:
        available = [t for t in INFORMATION_TOOLS if t in self.registry.list_tools()]
        if not available:
            raise ValueError("未注册 retrieval 或 web_search 工具")

        tool_hint = ""
        preferred = state.tool_hints.get(task.id)
        if preferred and preferred in available:
            tool_hint = f"Reflection 建议使用 Tool: {preferred}"

        routing_guidance = build_routing_guidance(
            state.question,
            task.description,
            state.local_kb_available,
        )
        print(f"[Router Guidance] {routing_guidance}")

        tool_descriptions = self._format_tool_descriptions(available)
        prior = format_prior_outputs(task, state.plan)
        last_error = ""

        for attempt in range(1, self.MAX_ROUTE_ATTEMPTS + 1):
            prompt = TOOL_SELECT_PROMPT.format(
                system_context=state.system_context,
                question=state.question,
                task_id=task.id,
                task_description=task.description,
                tool_descriptions=tool_descriptions,
                prior_results=prior + (f"\n上次错误: {last_error}" if last_error else ""),
                routing_guidance=routing_guidance,
                tool_hint=tool_hint,
            )

            response = ask_llm(prompt, max_tokens=350)
            data = parse_llm_json(response, error_prefix="Router")

            tool_name = data.get("tool", available[0])
            if tool_name not in available:
                tool_name = available[0]

            arguments = data.get("arguments") or {}
            thought = data.get("thought", "")

            err = self._validate_arguments(tool_name, arguments)
            if err is None:
                print(f"[Router] Task #{task.id} → {tool_name} (attempt {attempt})")
                print(f"[Router Thought] {thought}")
                print(f"[Router Arguments] {arguments}")
                return RouteResult(
                    task_id=task.id,
                    tool=tool_name,
                    arguments=arguments,
                    thought=thought,
                )

            last_error = err
            print(f"[Router Retry {attempt}] {err}")

        raise ValueError(f"Router 无法构造有效参数: {last_error}")

    def _format_tool_descriptions(self, tools: list[str]) -> str:
        lines = []
        for schema in self.registry.get_tool_schema():
            if schema["name"] not in tools:
                continue
            params = schema.get("parameters", {})
            props = params.get("properties", {})
            param_names = ", ".join(props.keys()) if props else ""
            desc = schema["description"].strip().replace("\n", " ")[:300]
            lines.append(f"- {schema['name']}({param_names}): {desc}")
        return "\n".join(lines)

    def _validate_arguments(self, tool_name: str, arguments: dict) -> Optional[str]:
        if not arguments:
            return "arguments 为空"
        try:
            self.registry.get(tool_name).args_schema(**arguments)
            return None
        except Exception as e:  # noqa: BLE001
            return str(e)
