from .base import Agent
import json


class EvaluationAgent(Agent):
    """
    Wrapper around the generic Agent class for evaluation.
    """

    def __init__(self, llm):
        prompt = """
Evaluate the correctness of the following knowledge triple:

Triple:
({head}) --> ({tail})

Evidence sentences:
{sentences}

Question:
Is the triple a correct factual statement according to the evidence?
Answer "yes" or "no".
"""
        super().__init__(llm, prompt)
