from .base import Agent
import json


class EntityTypingAgent:
    """
    Wrapper around the generic Agent class using the Tier-1 ontology.
    """

    def __init__(self, llm):
        prompt = """
You are an entity classification system.

Given an entity string, assign exactly one semantic type from the following list:

- PERSON
- ORG
- GPE
- LOCATION
- DATE
- NUMBER
- WORK
- EVENT
- OTHER

Guidelines:
- PERSON: individual people or fictional characters.
- ORG: companies, institutions, corporations, non-profits.
- LOCATION: mountains, rivers, planets, regions.
- DATE: temporal expressions.
- NUMBER: numeric values not representing dates.
- WORK: tasks, publications, creative works.
- EVENT: wars, elections, conferences, historical events.
- OTHER: anything else.

Constraints:
- Do NOT infer facts not implied by the string.
- If ambiguous, choose the most common interpretation.
- Return ONLY JSON of the form:
{{
  "entity": "{{entity}}",
  "type": "TYPE_LABEL"
}}

Entity:
{entity}
"""
        self.agent = Agent(llm, prompt=prompt)

    def assign_type(self, entity: str):
        resp = self.agent.run({"entity": entity})
        try:
            obj = json.loads(resp)
            return obj["type"]
        except:
            return "OTHER"
