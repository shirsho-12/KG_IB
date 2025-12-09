from .base import Agent
import json


class TripletExtractionAgent:
    """
    Wraps the generic Agent class with a specific prompt
    for relation extraction.
    """

    def __init__(self, llm):
        prompt = """Extract all relational triples from the text.
            Return a JSON array of objects with keys: head, relation, tail.

            Text: {text}
        """
        self.agent = Agent(llm, prompt)

    def extract(self, text: str):
        """
        Returns a list of triples: [(head, relation, tail), ...]
        """
        resp = self.agent.run({"text": text})
        if "```json" in resp:
            resp = resp.split("```json")[1].split("```")[0].strip()
        try:
            triples = json.loads(resp)
        except:
            # Fallback if the agent didn’t produce valid JSON
            print(resp)
            return []
        return [(d["head"], d["relation"], d["tail"]) for d in triples]
