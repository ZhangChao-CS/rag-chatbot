from memory.base_memory import BaseMemory
from memory.message import Message


class ConversationMemory(BaseMemory):
    def __init__(self, max_messages=10):

        self.messages = []

        self.max_messages = max_messages

    def add_user_message(self, content):

        self.messages.append(Message(role="user", content=content))

        self._trim()

    def add_ai_message(self, content):

        self.messages.append(Message(role="assistant", content=content))

        self._trim()

    def _trim(self):

        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def get_context(self):

        history = ""

        for msg in self.messages:
            history += f"{msg.role}: {msg.content}\n"

        return history
