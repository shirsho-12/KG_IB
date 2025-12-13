import os
from model.openai_model import OpenAIModel
import openai
from dotenv import load_dotenv
from src.agent import TripletExtractionAgent, EntityTypingAgent
from src.embedding_generator import EmbeddingGenerator

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def process_line(text, extractor, embedder, typer):
    """
    Runs:
    1) triple extraction
    2) embedding generation
    3) type assignment
    Returns array of dict objects, one per triple.
    """
    triples = extractor.extract(text)
    s_dct = {"sentence": text, "triples": triples}
    processed = []
    for h, r, t in triples:

        # Embedding from relation + the full sentence
        emb = embedder(r, text)
        # Type pair
        type_h = typer.assign_type(h)
        type_t = typer.assign_type(t)

        processed.append(
            {
                "head": h,
                "relation": r,
                "tail": t,
                "embedding": emb,
                "type_pair": (type_h, type_t),
                "sentence": text,
            }
        )
    s_dct["data"] = processed
    return s_dct


def get_model():
    model = OpenAIModel(
        model_name="openai/gpt-4o-mini",
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        temperature=0.0,
    )
    client = openai.Client(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
    return model, client


def get_agents():
    model, client = get_model()
    extractor = TripletExtractionAgent(model)
    embedder = EmbeddingGenerator(client)
    typer = EntityTypingAgent(model)
    return extractor, embedder, typer
