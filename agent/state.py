class AgentState:
    
    def __init__(
        self,
        question,
        history_text
    ):

        self.question = question
        self.history_text = history_text

        self.thought=""
        self.action=None

        self.observation=""
        self.answer=""
