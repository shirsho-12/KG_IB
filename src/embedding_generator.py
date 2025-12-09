from openai import OpenAI
import numpy as np
import torch


class EmbeddingGenerator:
    """
    Produces embeddings for relation clusters.
    Replace random embeddings with OpenAI API.
    """

    def __init__(self, client: OpenAI, dim=1024):
        self.dim = dim
        self.client = client

    def embedding_fn(self, relation: str, context: str) -> np.ndarray:
        """
        GPT-based embedding function.
        """
        text = relation + "\n" + context
        response = self.client.embeddings.create(
            input=[text], model="text-embedding-3-small"
        )
        embeddings = [
            torch.tensor(data_point.embedding) for data_point in response.data
        ]
        return torch.stack(embeddings).numpy()

    def __call__(self, relation: str, context: str) -> np.ndarray:
        return self.embedding_fn(relation, context)
