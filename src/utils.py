import asyncio
import os
import random

from dotenv import load_dotenv
import openai
from tqdm.asyncio import tqdm as async_tqdm

from model.openai_model import OpenAIModel
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


async def process_line_async(text, extractor, embedder, typer):
    triples = await extractor.extract_async(text)
    s_dct = {"sentence": text, "triples": triples}
    if not triples:
        s_dct["data"] = []
        return s_dct

    async def handle_triple(triple):
        h, r, t = triple
        emb_task = asyncio.create_task(embedder.aembedding_fn(r, text))
        type_h_task = asyncio.create_task(typer.assign_type_async(h))
        type_t_task = asyncio.create_task(typer.assign_type_async(t))
        embedding = await emb_task
        type_h, type_t = await asyncio.gather(type_h_task, type_t_task)
        return {
            "head": h,
            "relation": r,
            "tail": t,
            "embedding": embedding,
            "type_pair": (type_h, type_t),
            "sentence": text,
        }

    processed = await asyncio.gather(*(handle_triple(triple) for triple in triples))
    s_dct["data"] = processed
    return s_dct


async def process_tasks_asynchronously(tasks, worker, concurrency_limit, desc):
    if not tasks:
        return []
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def process_with_semaphore(task, pbar):
        async with semaphore:
            await asyncio.sleep(random.uniform(0.1, 0.3))
            result = await worker(task)
            pbar.update(1)
            return result

    with async_tqdm(total=len(tasks), desc=desc) as pbar:
        async_tasks = [process_with_semaphore(task, pbar) for task in tasks]
        return await asyncio.gather(*async_tasks)


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
    llm = model.llm
    extractor = TripletExtractionAgent(llm)
    embedder = EmbeddingGenerator(client)
    typer = EntityTypingAgent(llm)
    return extractor, embedder, typer
