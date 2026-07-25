from rag.llm import ask_llm

import json


class SimpleAgent:

    def __init__(self, retrieval_tool):
        self.retrieval_tool = retrieval_tool

    def decide_tool(self, question: str) -> str:
        prompt = f"""
        你是一个Agent，请判断应该使用哪个工具。如果问题涉及上传文档、论文、资料、PDF、文件内容，一律选择 retrieval。
        可用工具：
        1. retrieval：需要查询上传文档时使用
        2. direct：可以直接回答时使用
        请严格按照下面JSON格式输出，不要输出任何解释：
        {{
            "tool": "retrieval"
        }}
        或者
        {{
            "tool": "direct"
        }}
        问题：
        {question}
        """

        response = ask_llm(prompt)
        print(f"[Decision Raw] {response}")

        # 去掉Markdown代码块
        response = response.replace("```json", "").replace("```", "").strip()

        try:
            result = json.loads(response)
            tool = result.get("tool", "retrieval")
        except Exception as e:
            print(f"[Decision Error] {e}")
            tool = "retrieval"

        print(f"[Agent Decision] {tool}")
        return tool

    def run(self, question: str, history_text: str):
        action = self.decide_tool(question)

        if action == "retrieval":
            print("[Tool] RetrievalTool")
            docs = self.retrieval_tool.run(question)

            print(f"[Retrieved Docs] {len(docs)}")
            context = "\n".join(doc.page_content for doc in docs)

            prompt = f"""
            以下是历史对话：
            {history_text}
            请根据以下资料回答问题：
            资料：
            {context}
            问题：
            {question}
            请在100字以内回答。
            """
            return ask_llm(prompt), docs
        else:
            print("[Tool] Direct LLM")
            prompt = f"""
            以下是历史对话：
            {history_text}
            当前问题：
            {question}
            请在100字以内回答。
            """
            return ask_llm(prompt), []
