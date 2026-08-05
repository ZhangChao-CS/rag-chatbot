import json
import time

from agent.action import ToolAction
from agent.reflection.reflection import Reflection
from agent.state import AgentState
from rag.llm import ask_llm
from tools.registry import ToolRegistry


class SimpleAgent:
    def __init__(self, tools, memory):

        self.registry = ToolRegistry()
        self.reflection = Reflection()

        for tool in tools:
            self.registry.register(tool)

        self.memory = memory

    def build_tool_prompt(self):

        tools = self.registry.get_tool_schema()

        text = ""

        for tool in tools:
            text += f"""
            工具名称:
            {tool["name"]}

            功能:
            {tool["description"]}

            参数格式:
            {json.dumps(tool["parameters"], ensure_ascii=False, indent=2)}

            ------------------
            """

        return text

    def decide_tool(self, state: AgentState):
        prompt = f"""
        你是一个 ReAct Agent。

        你的任务是：
        根据当前信息决定下一步行动。

        当前用户问题：
        {state.question}

        历史对话：
        {state.history_text}

        当前已有观察信息：
        {state.get_observation_text()}

        当前Reflection结果:
        {state.reflection}

        当前执行步数：
        {state.step_count}

        可用工具：

        {self.build_tool_prompt()}

        请严格输出 JSON：

        {{
            "thought":"",
            "action":
            {{
                "tool":"",
                "arguments":{{}}
            }}
        }}

        action.tool 只能选择：

        1. 工具名称
        表示继续调用工具。
        不要输出final。

        2. 不允许选择final。
        Agent是否结束由 Reflection 判断。
        你的职责只有：
        选择下一步工具。

        注意：
        - 1.如果问题可能来自本地知识库、论文、PDF：优先选择 retrieval。

          2.只有以下情况选择 web_search：最新信息、实时数据、当前事件、本地知识库无法回答的问题。

          3.不要因为问题专业就选择 web_search。专业问题优先查询知识库。
        - 如果已有足够信息，选择 final。

        非常重要：
        arguments 必须是工具参数的直接值。

        例如：
        正确：
        {{
            "query":"MPNN是什么"
        }}

        错误：
        {{
            "query":{{
                "title":"MPNN是什么"
        }}
        }}

        错误：
        {{
            "query":{{
                "text":"MPNN是什么"
        }}
        }}

        不要自行增加字段。
        不要包装对象。
        不要嵌套 JSON。

        必须严格按照工具 schema 输出。
        """

        response = ask_llm(prompt)

        response = response.replace("```json", "").replace("```", "").strip()

        try:
            result = json.loads(response)

        except Exception as e:  # noqa: BLE001
            print("[Decision Parse Error]", e)

            return ("", ToolAction(tool="final", arguments={}))

        thought = result.get("thought", "")

        action_data = result.get("action", {})

        action = ToolAction(
            tool=action_data.get("tool", "final"),
            arguments=action_data.get("arguments", {}),
        )

        print("[Thought]", thought)
        print("[Action]", action.tool)
        print("[Arguments]", action.arguments)

        return thought, action

    def normalize_arguments(self, action):

        new_args = {}

        for k, v in action.arguments.items():
            # 如果字符串参数被包成对象
            if isinstance(v, dict) and len(v) == 1:
                v = next(iter(v.values()))

            new_args[k] = v

        action.arguments = new_args

        return action

    def validate_action(self, action):

        tool = self.registry.get(action.tool)

        schema = tool.args_schema

        try:
            validated = schema(**action.arguments)

        except Exception as e:
            print("[Schema Error]", e)

            print("[Arguments]", action.arguments)

            raise e  # noqa: TRY201

        return validated

    def run(self, question):

        # ================= Memory Retrieve =================

        history_text = self.memory.get_context()

        state = AgentState(question, history_text)

        state.trace.question = question

        state.trace.add_step(stage="memory_retrieve", context=history_text)

        self.memory.add_user_message(question)

        # ================= ReAct Loop =================

        while state.can_continue():
            # ================= Planner =================

            start = time.time()

            state.thought, state.action = self.decide_tool(state)

            planner_time = time.time() - start

            state.trace.add_step(
                stage="planner",
                duration=planner_time,
                thought=state.thought,
                tool=state.action.tool,
                arguments=state.action.arguments,
            )

            # ================= Tool =================

            try:
                tool = self.registry.get(state.action.tool)

            except Exception as e:  # noqa: BLE001
                state.trace.add_step(stage="error", error=str(e))

                state.answer = "工具不存在"

                break

            # ================= Validate =================

            try:
                state.action = self.normalize_arguments(state.action)
                args = self.validate_action(state.action)

            except Exception as e:  # noqa: BLE001
                state.trace.add_step(
                    stage="validation_error",
                    error=str(e),
                    arguments=state.action.arguments,
                )

                state.answer = "工具参数校验失败"

                break

            # ================= Tool Execute =================

            start = time.time()

            try:
                result = tool.run(**args.model_dump())

                tool_time = time.time() - start

                state.observation.append(result["observation"])

                state.trace.add_step(
                    stage="tool_execute",
                    duration=tool_time,
                    tool=state.action.tool,
                    input=args.model_dump(),
                    observation=state.trace.summarize_observation(
                        result["observation"]
                    ),
                )

            except Exception as e:  # noqa: BLE001
                state.trace.add_step(stage="tool_error", error=str(e))

                state.answer = "工具执行失败"

                break

            # ================= Reflection =================

            reflection = self.reflection.run(
                question=state.question, observation=state.get_observation_text()
            )

            state.reflection = reflection

            state.trace.add_step(
                stage="reflection",
                sufficient=reflection.sufficient,
                confidence=reflection.confidence,
                reason=reflection.reason,
                missing_information=reflection.missing_information,
                suggested_tool=reflection.suggested_tool,
            )

            if reflection.sufficient:
                state.answer = ask_llm(
                    f"""
                    用户问题：
                    {state.question}

                    已知信息：
                    {state.get_observation_text()}

                    请生成最终答案。
                    """
                )

                state.trace.add_step(stage="final_answer", answer=state.answer)

                state.finished = True

                break

            state.next_step()

        # ================= Memory Save =================

        self.memory.add_ai_message(state.answer)

        state.trace.add_step(stage="memory_save", message=state.answer)

        state.trace.answer = state.answer

        print(state.trace)

        return (state.answer, [])
