from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable

from agent.base_agent import BaseAgent


class Agent(BaseAgent):
    """Agent that'll be invoked."""

    def __init__(self, llm, prompt):
        super().__init__()
        self.llm = llm
        self.prompt_template = PromptTemplate.from_template(prompt)
        self.chain: Runnable = self.prompt_template | self.llm | StrOutputParser()

    def run(self, invoke_dct: dict) -> str:
        """Generates output based on the input dictionary."""
        try:
            output = self.chain.invoke(invoke_dct)
            return output
        except Exception as e:
            raise e
