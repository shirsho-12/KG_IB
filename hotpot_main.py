from pathlib import Path
from src.kg import NXKnowledgeGraph
from src.clusterer import OnlineRelationClusterer
from src.pragma import PragmaticEquivalenceLearner
from src.redundancy_filter import RedundancyFilter
from src.utils import get_agents
from tqdm import tqdm
import pickle
from datasets import load_dataset

ds = load_dataset("hotpotqa/hotpot_qa", "distractor", cache_dir="data/")

dset_name = "hotpotqa"
input_path = Path(f"data/{dset_name}.txt")

dset = ds["validation"]
print(len(dset))
output_path = Path(f"output/{dset_name}/tier_2")
output_file_path = output_path / "s_1_extracted.pkl"
output_path.mkdir(parents=True, exist_ok=True)


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
        if "Paragraph" in h or "Paragraph" in t:
            continue
        if "Title:" in h or "Title:" in t:
            continue
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


extractor, embedder, typer = get_agents()

if output_file_path.exists():
    print(f"Stage 1 data already exists at {output_path}, skipping extraction.")
    s_1_data = pickle.load(open(output_file_path, "rb"))
else:
    s_1_data = {}
    for i in tqdm(range(3), desc="Stage 1", total=3):
        sample = dset[i]
        # for sample in tqdm(dset, desc="Stage 1", total=len(dset)):
        paragraphs = sample["context"]
        n = len(paragraphs["title"])
        idx_arr = []
        for i in range(n):
            title = paragraphs["title"][i]
            para_sentences = paragraphs["sentences"][i]
            text = "Title: " + title + "\nParagraph: " + " ".join(para_sentences)
            processed = process_line(text, extractor, embedder, typer)
            idx_arr.append(processed)
        s_1_data[sample["id"]] = idx_arr

    # save to disk
    pickle.dump(s_1_data, open(output_file_path, "wb"))
    print(f"Saved stage 1 data to {output_path}")

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
        # print only if redundant
        if not added:
            log_text += f"REDUNDANT: {h}, {r_surface}, {t} → cluster {cid}\n"
    log_text += (
        f"Final KG has {len(kg.G.nodes())} nodes and {len(kg.G.edges())} edges.\n"
    )
    graph_dct[idx] = kg

with open(output_path / "processing_log.txt", "w") as f:
    f.write(log_text)

pickle.dump(graph_dct, open(output_path / "knowledge_graphs.pkl", "wb"))
