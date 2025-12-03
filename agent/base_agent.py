from abc import ABC, abstractmethod

# from src.utils.logger import get_logger


class BaseAgent(ABC):
    """An abstract base class for all agents in the pipeline."""

    def __init__(self):
        # self.logger = get_logger(self.__class__.__name__)
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        """The main entry point for the agent's execution."""
        pass
