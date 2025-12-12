from .base import Agent
import json
from pathlib import Path


class EntityTypingAgent:
    """
    Wrapper around the generic Agent class using the ontology.
    """

    def __init__(self, llm):
        prompt = Path(
            Path(__file__).parent.parent.parent / "prompts" / "tier_2_ontology.txt"
        ).read_text()
        self.agent = Agent(llm, prompt=prompt)

    def assign_type(self, entity: str):
        resp = self.agent.run({"entity": entity})
        try:
            obj = json.loads(resp)
            return obj["type"]
        except:
            return "OTHER"
