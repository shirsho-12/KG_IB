from pprint import pprint
from tqdm import tqdm
from collections import defaultdict, deque
from email.utils import parsedate_to_datetime
import pandas as pd
from pathlib import Path
from src.utils import get_model, process_tasks_asynchronously
from src.agent.base import Agent
import pickle
from src.kg import NXKnowledgeGraph
import asyncio
import time

from datasets import load_dataset

# ds = load_dataset("hotpotqa/hotpot_qa", "distractor", cache_dir="data/")

dataset_name = "musique"
ds = load_dataset("dgslibisey/MuSiQue", cache_dir="data/")

dset: pd.DataFrame
dset = ds["validation"]  # type: ignore
model, _ = get_model()

data_path = Path.cwd() / "output" / "musique" / "26-12-2025"
output_path = Path.cwd() / "output" / "musique"

data_files = list(data_path.glob("stage_1_results_*.pkl"))
kg_files = list(data_path.glob("kg_*.pkl"))

kgs = {}
for kg_file in tqdm(kg_files, desc="Loading KG files"):
    with open(kg_file, "rb") as f:
        kg_dct: dict[str, NXKnowledgeGraph] = pickle.load(f)
        kgs.update(kg_dct)
data = {}
for data_file in tqdm(data_files, desc="Loading data files"):
    with open(data_file, "rb") as f:
        data_dct: dict[str, dict] = pickle.load(f)
        data.update(data_dct)

d_dct = {}
# print(len(kgs))
for k, v in kgs.items():

    # print(k, len(v.G.nodes), len(v.G.edges))
    # print(dset[0])
    for d in dset:
        if d["id"] == k:  # type: ignore
            question = d["question"]
            answer = d["answer"]
            d_dct[k] = {
                "question": question,
                "answer": answer,
                "kg": v,
                "data": data[k],
            }
            break


def evaluate_kg_qa(kg, question: str):
    kg_str = ""
    for h, t, data in kg.G.edges(data=True):
        surfaces = data.get("surfaces")
        if surfaces is None:
            continue
        for surface in surfaces:
            kg_str += f"({h} -> {surface} -> {t})\n"

    kg_qa_prompt = f"""You are given a knowledge graph represented as a list of triples in the form (head -> relation -> tail).
    Your task is to answer the question based on the knowledge graph. Only use the information present in the knowledge graph to formulate your answer.
    Do not assume any external knowledge and only provide the answer. Think, then answer. Do not use external knowledge, but you are allowed to 
    perform multi-hop reasoning over the triples in the knowledge graph to arrive at the answer.
    Provide triples you used to arrive at the answer.
    Here is the knowledge graph:
    {kg_str}
    Question: {question}
    Answer:"""
    kg_qa_agent = Agent(llm=model, prompt=kg_qa_prompt)

    resp = kg_qa_agent.run({"question": question, "kg_str": kg_str})
    return resp


# Convert knowledge graph to string of triples
tasks = []
for k, v in tqdm(d_dct.items()):
    question = v["question"]
    answer = v["answer"]
    kg: NXKnowledgeGraph = v["kg"]
    tasks.append((k, question, answer, kg))


class RateLimiter:
    def __init__(self, max_calls: int, period_seconds: int):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.calls = deque()
        self.lock = asyncio.Lock()

    async def acquire(self):
        while True:
            async with self.lock:
                now = time.monotonic()
                while self.calls and now - self.calls[0] >= self.period_seconds:
                    self.calls.popleft()
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return
                wait_for = self.period_seconds - (now - self.calls[0])
            await asyncio.sleep(wait_for)


rate_limiter = RateLimiter(max_calls=20, period_seconds=60)


async def call_with_retries(kg, question, max_retries: int = 5):
    for _ in range(max_retries):
        await rate_limiter.acquire()
        try:
            # Run sync LLM call in a thread to avoid blocking the event loop.
            return await asyncio.to_thread(evaluate_kg_qa, kg, question)
        except Exception as exc:
            if _is_rate_limit_error(exc):
                delay = _get_retry_after_seconds(exc)
                await asyncio.sleep(delay if delay is not None else 60)
                continue
            raise
    raise RuntimeError("Exceeded retries due to rate limiting.")


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc)
    if "Rate limit" in msg or "429" in msg:
        return True
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    code = getattr(exc, "code", None)
    return code == 429


def _get_retry_after_seconds(exc: Exception) -> float | None:
    headers = _extract_headers(exc)
    if not headers:
        return None
    headers_lc = {str(k).lower(): v for k, v in headers.items()}
    for key in (
        "retry-after",
        "x-ratelimit-reset",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
    ):
        if key not in headers_lc:
            continue
        value = headers_lc[key]
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value is None:
            continue
        try:
            delay = float(value)
            if delay > 1e9:
                delay = max(0.0, delay - time.time())
            return delay + 1.0
        except (TypeError, ValueError):
            try:
                dt = parsedate_to_datetime(str(value))
                return max(0.0, dt.timestamp() - time.time()) + 1.0
            except (TypeError, ValueError):
                continue
    return None


def _extract_headers(exc: Exception) -> dict | None:
    for attr in ("headers", "response"):
        obj = getattr(exc, attr, None)
        if obj is None:
            continue
        if isinstance(obj, dict):
            if "headers" in obj and isinstance(obj["headers"], dict):
                return obj["headers"]
        headers = getattr(obj, "headers", None)
        if isinstance(headers, dict):
            return headers
    for attr in ("metadata", "body", "error", "args"):
        obj = getattr(exc, attr, None)
        if obj is None:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("headers"), dict):
            return obj["headers"]
        if isinstance(obj, tuple):
            for item in obj:
                if isinstance(item, dict) and isinstance(item.get("headers"), dict):
                    return item["headers"]
    return None


async def worker(task):
    k, question, answer, kg = task
    resp = await call_with_retries(kg, question)
    return k, {"response": resp, "answer": answer}


async def run_eval():
    return await process_tasks_asynchronously(
        tasks, worker, concurrency_limit=50, desc="Evaluating KG QA"
    )


processed_results = asyncio.run(run_eval())

results = {}
for k, res in processed_results:
    results[k] = res


def compute_accuracy_and_f1(outputs):
    y_true = []
    y_pred = []
    for k, v in outputs.items():
        y_true.append(v["answer"])
        y_pred.append(v["response"])

    accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true in pred) / len(
        y_true
    )
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true in pred)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true not in pred)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true not in pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(tp, fp, fn)
    print(precision, recall)
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return accuracy, f1


accuracy, f1 = compute_accuracy_and_f1(results)
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
# Save outputs
with open(output_path / "kg_qa_outputs.pkl", "wb") as f:
    pickle.dump(results, f)
