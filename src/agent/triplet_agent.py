from .base import Agent
import json


class TripletExtractionAgent:
    """
    Wraps the generic Agent class with a specific prompt
    for relation extraction.
    """

    def __init__(self, llm):
        prompt = (
            "Extract all relational triples from the text.\n"
            "Return a JSON array of objects with keys: head, relation, tail.\n\n"
            "Text: {{text}}\n"
        )
        self.agent = Agent(prompt, llm)

    def extract(self, text: str):
        """
        Returns a list of triples: [(head, relation, tail), ...]
        """
        resp = self.agent.run({"text": text})
        try:
            triples = json.loads(resp)
        except:
            # Fallback if the agent didn’t produce valid JSON
            return []
        return [(d["head"], d["relation"], d["tail"]) for d in triples]
