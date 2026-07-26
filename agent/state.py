class AgentState:
    def __init__(self, question: str, history: str):
        self.question = question
        self.history = history

        self.thought = ""
        self.action = ""

        self.tool_input = ""

        self.observation = ""

        self.answer = ""
