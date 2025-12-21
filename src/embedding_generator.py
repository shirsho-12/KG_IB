import asyncio

import backoff
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

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def embedding_fn(self, relation: str, context: str) -> np.ndarray:
        """
        GPT-based embedding function.
        """
        text = relation + "\n" + context
        response = self.client.embeddings.create(
            input=[text], model="text-embedding-3-small"
        )
        if not response.data:
            raise ValueError("No embedding data received.")
        embeddings = [
            torch.tensor(data_point.embedding) for data_point in response.data
        ]
        return torch.stack(embeddings).numpy()

    async def aembedding_fn(self, relation: str, context: str) -> np.ndarray:
        return await asyncio.to_thread(self.embedding_fn, relation, context)

    def __call__(self, relation: str, context: str) -> np.ndarray:
        return self.embedding_fn(relation, context)
