from abc import ABC, abstractmethod


class BaseTool(ABC):

    @property
    @abstractmethod
    def name(self):
        pass


    @property
    @abstractmethod
    def description(self):
        pass

    @staticmethod
    def create_result(observation, raw=None):

        return {
            "observation":
                observation,

            "raw":
                raw
        }

    @property
    @abstractmethod
    def args_schema(self):
        pass

    @abstractmethod
    def run(self, **kwargs):
        pass
