from abc import ABC, abstractmethod


class BaseMemory(ABC):
    @abstractmethod
    def add_user_message(self, content: str):
        pass

    @abstractmethod
    def add_ai_message(self, content: str):
        pass

    @abstractmethod
    def get_context(self):
        pass
