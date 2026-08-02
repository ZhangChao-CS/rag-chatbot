from rag.llm import ask_llm

from agent.state import AgentState
from agent.action import ToolAction

from tools.registry import ToolRegistry

import json


class SimpleAgent:

    def __init__(self, tools):

        self.registry = ToolRegistry()

        for tool in tools:

            self.registry.register(tool)

    def build_tool_prompt(self):

        tools = self.registry.get_tool_schema()

        text = ""

        for tool in tools:

            text += f"""
            工具名称:
            {tool['name']}

            功能:
            {tool['description']}

            参数格式:
            {json.dumps(
                tool['parameters'],
                ensure_ascii=False,
                indent=2
            )}

            ------------------
            """

        return text

    def decide_tool(self, question):

        prompt = f"""
        你是一个 AI Agent。

        可用工具：
        {self.build_tool_prompt()}

        请严格输出 JSON：
        {{
            "thought":"",
            "action":
            {{
                "tool":"",
                "arguments":{{"必须严格符合工具参数格式"}}
            }}
        }}

        注意：
        arguments必须是键值对。
        字符串参数必须直接填写字符串。
        不要额外嵌套对象。

        如果无需工具：
        action.tool = "direct"
        arguments为空。

        问题：
        {question}
        """

        response = ask_llm(prompt)

        response = (
            response
            .replace("```json","")
            .replace("```","")
            .strip()
        )

        try:

            result = json.loads(response)

        except Exception as e:

            print(
                "[Decision Parse Error]",
                e
            )

            return (
                "",
                ToolAction(
                    tool="direct",
                    arguments={}
                )
            )

        thought = result.get(
            "thought",
            ""
        )

        action_data = result.get(
            "action",
            {}
        )

        action = ToolAction(
            tool=action_data.get(
                "tool",
                "direct"
            ),

            arguments=
                action_data.get(
                    "arguments",
                    {}
                )
        )

        print("[Thought]", thought)
        print("[Action]", action.tool)
        print("[Arguments]", action.arguments)

        return thought, action

    def reason_after_observation(self, state: AgentState):

        prompt = f"""
        你是一个AI Agent。
        用户问题：
        {state.question}
        你的第一次思考：
        {state.thought}
        工具返回的结果：
        {state.observation}
        请思考：
        1. 工具返回的信息是否足够？
        2. 如果足够，请给出最终答案。
        3. 如果不足，请明确说明信息不足，不要编造。
        请严格按照JSON格式输出：
        {{
            "thought": "...",
            "answer": "..."
        }}
        """

        response = ask_llm(prompt)

        response = (
            response.replace("```json", "")
            .replace("```", "")
            .strip()
        )

        result = json.loads(response)

        state.thought=result.get(
            "thought",
            ""
        )

        state.answer=result.get(
            "answer",
            ""
        )

    def validate_action(self, action):

        tool = self.registry.get(
            action.tool
        )

        schema = tool.args_schema

        try:
            validated = schema(
                **action.arguments
            )

        except Exception as e:
            print(
                "[Schema Error]",
                e
            )

            print(
                "[Arguments]",
                action.arguments
            )

            raise e

        return validated

    def run(self, question, history_text):

        state = AgentState(question, history_text)

        state.thought, state.action = self.decide_tool(question)

        # direct
        if state.action.tool == "direct":

            state.answer = ask_llm(
                f"""
                历史：
                {history_text}

                问题：
                {question}
                """
            )

            return state.answer, []

        try:
            tool = self.registry.get(
                state.action.tool
            )

        except Exception:
            state.answer = ("工具不存在")
            return state.answer,[]

        args = self.validate_action(state.action)

        result = tool.run(**args.model_dump())

        state.observation = (result["observation"])

        self.reason_after_observation(state)

        return (
            state.answer,
            result["raw"]
        )
