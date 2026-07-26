from abc import ABC, abstractmethod

class BaseTool(ABC):

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def run(self, tool_input):
        pass
    