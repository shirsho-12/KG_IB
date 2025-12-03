from abc import ABC, abstractmethod
from langchain_core.runnables import RunnableLambda

# from src.utils.logger import get_logger
# from src.config.config import BASE_URL, API_KEY, TEMPERATURE


class BaseModel(RunnableLambda, ABC):
    """An abstract base class for all models in the pipeline."""

    def __init__(self, api_url=None, api_key=None, temperature=None):
        # self.logger = get_logger(self.__class__.__name__)
        self.api_url = api_url  #  or BASE_URL
        self.api_key = api_key  # or API_KEY
        self.temperature = temperature  # or TEMPERATURE

    @abstractmethod
    def invoke(self, *args, **kwargs):
        """The main entry point for the model's execution."""
        pass
