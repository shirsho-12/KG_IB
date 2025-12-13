from abc import ABC, abstractmethod

import backoff
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable


class BaseAgent(ABC):
    """An abstract base class for all agents in the pipeline."""

    def __init__(self):
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        """The main entry point for the agent's execution."""
        pass


class Agent(BaseAgent):
    """Agent that'll be invoked."""

    def __init__(self, llm, prompt):
        super().__init__()
        self.llm = llm
        self.prompt_template = PromptTemplate.from_template(prompt)
        self.chain: Runnable = self.prompt_template | self.llm | StrOutputParser()

    def run(self, invoke_dct: dict) -> str:
        """Generates output based on the input dictionary."""
        return self.chain.invoke(invoke_dct)

    @staticmethod
    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def _run_with_retry(chain: Runnable, invoke_dct: dict) -> str:
        output = await chain.ainvoke(invoke_dct)
        if not output:
            raise ValueError("Empty response from LLM.")
        return output

    async def run_async(self, invoke_dct: dict) -> str:
        """Async variant of run with retries."""
        return await self._run_with_retry(self.chain, invoke_dct)
