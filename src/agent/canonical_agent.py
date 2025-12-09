from .base import Agent


class CanonicalLabelAgent:
    """
    Wraps the generic Agent class with a specific prompt
    for canonicalizing relation labels.
    """

    def __init__(self, llm):
        prompt = """
You are a relation-normalization system.
Given several surface forms of a relation, produce a single canonical label
in snake_case, 2–4 words, expressing the shared meaning.

Examples:
- ["works_at", "is_employed_at"] → "employed_at"
- ["born_in", "was_born_in", "originates_from"] → "born_in"
- ["is_headquartered_in", "based_in", "operates_out_of"] → "headquartered_in"

Surface forms:
{forms}
"""
        self.agent = Agent(llm, prompt)

    def extract(self, forms: str):
        """
        Returns a canonical relation label.
        """
        resp = self.agent.run({"forms": forms})
        return resp.strip()
