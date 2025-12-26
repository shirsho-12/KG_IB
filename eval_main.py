from pprint import pprint
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from pathlib import Path
from src.utils import get_model, process_tasks_asynchronously
from src.agent.base import Agent
import pickle
from src.kg import NXKnowledgeGraph
import asyncio

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


async def worker(task):
    k, question, answer, kg = task
    resp = evaluate_kg_qa(kg, question)
    return k, {"response": resp, "answer": answer}


loop = asyncio.get_event_loop()
processed_results = loop.run_until_complete(
    process_tasks_asynchronously(
        tasks, worker, concurrency_limit=4, desc="Evaluating KG QA"
    )
)

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
