from .base import Agent
import json


class EntityTypingAgent:
    """
    Wrapper around the generic Agent class using the Tier-2 ontology.
    """

    def __init__(self, llm):
        prompt = """
You are an entity typing system based on the OntoNotes 5.0 / ACE coarse-grained ontology.

Given an entity string, assign exactly one semantic type from the following list:

- PERSON
- ORG
- GPE
- LOCATION
- FACILITY
- PRODUCT
- EVENT
- WORK_OF_ART
- LAW
- LANGUAGE
- DATE
- TIME
- QUANTITY
- PERCENT
- MONEY
- NORP
- OTHER

Guidelines:
- PERSON: individual people or fictional characters.
- ORG: companies, institutions, corporations, non-profits.
- GPE: countries, cities, states, nationalities used as geopolitical units.
- LOCATION: mountains, rivers, planets, regions.
- FACILITY: airports, highways, stadiums, bridges, buildings.
- PRODUCT: vehicles, software, consumer goods.
- EVENT: wars, elections, conferences, historical events.
- WORK_OF_ART: books, films, songs, artworks.
- LAW: legal documents or treaties.
- LANGUAGE: natural languages.
- DATE/TIME: temporal expressions.
- QUANTITY/PERCENT/MONEY: numeric expressions.
- NORP: nationalities, religions, political groups.
- OTHER: anything else.

Constraints:
- Do NOT infer facts not implied by the string.
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
