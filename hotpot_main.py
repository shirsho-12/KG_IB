import argparse
import asyncio
import pickle
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from src.clusterer import OnlineRelationClusterer
from src.kg import NXKnowledgeGraph
from src.pragma import PragmaticEquivalenceLearner
from src.redundancy_filter import RedundancyFilter
from src.utils import get_agents, process_tasks_asynchronously


parser = argparse.ArgumentParser()
parser.add_argument(
    "--samples",
    type=int,
    default=3,
    help="Number of validation samples to process",
)
parser.add_argument(
    "--concurrency",
    type=int,
    default=5,
    help="Maximum number of concurrent Stage 1 requests",
)


async def process_line_async(text, extractor, embedder, typer):
    triples = await extractor.extract_async(text)
    s_dct = {"sentence": text, "triples": triples}
    filtered = []
    for triple in triples:
        if triple is None:
            continue
        h, r, t = triple
        if "Paragraph" in h or "Paragraph" in t:
            continue
        if "Title:" in h or "Title:" in t:
            continue
        filtered.append((h, r, t))

    if not filtered:
        s_dct["data"] = []
        return s_dct

    async def handle_triple(triple):
        if triple is None:
            return None
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

    processed = await asyncio.gather(*(handle_triple(triple) for triple in filtered))
    s_dct["data"] = [item for item in processed if item is not None]
    return s_dct


async def main():
    args = parser.parse_args()
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", cache_dir="data/")
    dset = ds["validation"]
    print(len(dset))

    dset_name = "hotpotqa"
    output_path = Path(f"output/{dset_name}/tier_2")
    output_file_path = output_path / "s_1_extracted.pkl"
    output_path.mkdir(parents=True, exist_ok=True)

    extractor, embedder, typer = get_agents()

    if output_file_path.exists():
        print(f"Stage 1 data already exists at {output_file_path}, skipping extraction.")
        with open(output_file_path, "rb") as f:
            s_1_data = pickle.load(f)
    else:
        sample_count = min(args.samples, len(dset))
        dataset_slice = [dset[i] for i in range(sample_count)]
        tasks = []
        for sample in dataset_slice:
            paragraphs = sample["context"]
            n = len(paragraphs["title"])
            for i in range(n):
                title = paragraphs["title"][i]
                para_sentences = paragraphs["sentences"][i]
                text = "Title: " + title + "\nParagraph: " + " ".join(para_sentences)
                tasks.append((sample["id"], text))

        async def worker(payload):
            sample_id, text = payload
            processed = await process_line_async(text, extractor, embedder, typer)
            return sample_id, processed

        results = await process_tasks_asynchronously(
            tasks, worker, args.concurrency, "Stage 1"
        )
        grouped = defaultdict(list)
        for sample_id, processed in results:
            grouped[sample_id].append(processed)

        s_1_data = {}
        for sample in dataset_slice:
            s_1_data[sample["id"]] = grouped.get(sample["id"], [])

        with open(output_file_path, "wb") as f:
            pickle.dump(s_1_data, f)
        print(f"Saved stage 1 data to {output_file_path}")

    log_text = ""
    graph_dct = {}
    for idx, data in tqdm(s_1_data.items()):
        log_text += f"Processing sample {idx}\n"
        clusterer = OnlineRelationClusterer()
        for sample in data:
            for triple in sample["data"]:
                clusterer.process_triple(triple)

        log_text += f"Formed {len(clusterer.clusters)} clusters.\n"

        learner = PragmaticEquivalenceLearner(mi_threshold=0.25, min_pairs=2)

        equiv_classes, inverse_map = learner(clusterer)
        log_text += f"Equivalence classes: {len(equiv_classes)}\n"
        log_text += f"Inverse map: {len(inverse_map)}\n"

        kg = NXKnowledgeGraph()
        redundancy_filter = RedundancyFilter(kg, equiv_classes, inverse_map)

        for h, r_surface, t, cid, sentence in clusterer.fact_list:
            added = redundancy_filter.add_if_novel(
                h, cid, t, surface=r_surface, sentence=sentence
            )
            if not added:
                log_text += f"REDUNDANT: {h}, {r_surface}, {t} → cluster {cid}\n"
        log_text += (
            f"Final KG has {len(kg.G.nodes())} nodes and {len(kg.G.edges())} edges.\n"
        )
        graph_dct[idx] = kg

    with open(output_path / "processing_log.txt", "w") as f:
        f.write(log_text)

    with open(output_path / "knowledge_graphs.pkl", "wb") as f:
        pickle.dump(graph_dct, f)


if __name__ == "__main__":
    asyncio.run(main())
