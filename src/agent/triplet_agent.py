from .base import Agent
import json
from pathlib import Path


class TripletExtractionAgent:
    """
    Wraps the generic Agent class with a specific prompt
    for relation extraction.
    """

    def __init__(self, llm):
        prompt = Path(
            Path(__file__).parent.parent.parent / "prompts" / "new_triplet_prompt.txt"
        ).read_text()

        self.agent = Agent(llm, prompt)

    def _parse_response(self, resp: str):
        if "```json" in resp:
            resp = resp.split("```json")[1].split("```")[0].strip()
        try:
            triples = json.loads(resp)
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            print(resp)
            return []
        cleaned = []
        if not isinstance(triples, list):
            return cleaned
        for item in triples:
            if not isinstance(item, dict):
                continue
            head = item.get("head")
            relation = item.get("relation")
            tail = item.get("tail")
            if head is None or relation is None or tail is None:
                continue
            cleaned.append((head, relation, tail))
        return cleaned

    def extract(self, text: str):
        """
        Returns a list of triples: [(head, relation, tail), ...]
        """
        resp = self.agent.run({"text": text})
        return self._parse_response(resp)

    async def extract_async(self, text: str):
        """
        Async variant of extract.
        """
        resp = await self.agent.run_async({"text": text})
        return self._parse_response(resp)
