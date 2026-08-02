from rag.llm import ask_llm


class QueryRewriteService:


    def run(self, query: str) -> str:


        if len(query) < 10:

            print("[No Rewrite]")

            return query


        prompt = f"""
        你是一名 Query Rewrite Agent。

        你的任务是将用户的问题改写成更适合文档检索的查询。

        要求：

        1. 保留原意，不要回答问题。
        2. 补充必要上下文，使查询更加明确。
        3. 不要编造信息。
        4. 输出仅包含改写后的查询。


        用户问题：

        {query}

        """


        rewritten_query = ask_llm(
            prompt
        ).strip()


        print("[Rewrite]")
        print("Original :", query)
        print("Rewritten:", rewritten_query)


        return rewritten_query
        