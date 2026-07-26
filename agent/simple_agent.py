from rag.llm import ask_llm
from agent.state import AgentState

import json


class SimpleAgent:

    def __init__(self, tools):
        self.tools = {
            tool.name: tool
            for tool in tools
        }

    def decide_tool(self, question: str) -> str:
        prompt = f"""
        你是一个 AI Agent。

        你可以使用以下工具：
        1. retrieval
        适用于：
        - 查询上传的 PDF
        - 查询论文
        - 查询知识库
        - 查询文档内容

        2. calculator
        适用于：
        - 数学计算
        - 四则运算
        - 百分比
        - 幂运算等

        3. direct
        适用于：
        - 常识问题
        - 闲聊
        - 不需要调用工具即可回答的问题

        请严格按照下面 JSON 输出：
        {{
            "thought": "...",
            "tool": "retrieval | calculator | direct",
            "tool_input": "..."
        }}

        要求：
        1.tool 必须是 retrieval、calculator、direct 三者之一。
        2.retrieval 的 tool_input 一般就是用户问题。
        3.calculator 的 tool_input 必须是数学表达式，例如 "23*45+18"。
        4.direct 的 tool_input 留空即可。

        问题：
        {question}
        """

        response = ask_llm(prompt)
        print(f"[Decision Raw] {response}")

        # 去掉Markdown代码块
        response = response.replace("```json", "").replace("```", "").strip()

        try:
            result = json.loads(response)

            thought = result.get("thought", "")
            tool = result.get("tool", "retrieval")
            tool_input = result.get("tool_input", question)

        except Exception as e:
            print(f"[Decision Error] {e}")
            thought = ""
            tool = "retrieval"
            tool_input = ""

        print(f"[Thought] {thought}")
        print(f"[Action] {tool}")
        print(f"[Tool Input] {tool_input}")

        return thought, tool, tool_input

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

        state.thought = result["thought"]
        state.answer = result["answer"]

    def run(self, question: str, history_text: str, use_rerank: bool = True):
        state = AgentState(question, history_text)
        state.thought, state.action, state.tool_input = self.decide_tool(question)

        if state.action == "retrieval":
            tool = self.tools["retrieval"]

            docs = tool.run(state.tool_input, use_rerank=use_rerank)

            print(f"[Observation] Retrieved {len(docs)} documents")

            state.observation = "\n".join(
                doc.page_content for doc in docs
            )

            self.reason_after_observation(state)
            print(f"[Thought2] {state.thought}")

            return state.answer, docs
        # ---------------- calculator ----------------
        elif state.action == "calculator":

            tool = self.tools["calculator"]
            result = tool.run(state.tool_input)
            state.observation = str(result)
            self.reason_after_observation(state)

            print(f"[Thought2] {state.thought}")

            return state.answer, []

        # ---------------- direct ----------------
        else:
            prompt = f"""
            以下是历史对话：
            {history_text}
            当前问题：
            {question}
            请直接回答，控制在100字以内。
            """
            state.answer = ask_llm(prompt)

            return state.answer, []
