from pprint import pprint
from dotenv import load_dotenv
from pathlib import Path
from src.kg import NXKnowledgeGraph
from src.clusterer import OnlineRelationClusterer
from src.pragma import PragmaticEquivalenceLearner
from src.redundancy_filter import RedundancyFilter
import os
from model.openai_model import OpenAIModel
import openai
from src.agent import TripletExtractionAgent, EntityTypingAgent
from src.embedding_generator import EmbeddingGenerator
import argparse
from tqdm import tqdm
import pickle


load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

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


extractor = TripletExtractionAgent(model)
embedder = EmbeddingGenerator(client)
typer = EntityTypingAgent(model)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dset",
    type=str,
    required=True,
    choices=["rebel", "webnlg", "wiki-nre"],
    help="Name of dataset",
)
args = parser.parse_args()
dset_name = args.dset
input_path = Path(f"data/{dset_name}.txt")

dset = input_path.read_text().splitlines()

output_path = Path(f"output/{dset_name}/s_1_extracted.pkl")
output_path.parent.mkdir(parents=True, exist_ok=True)
if output_path.exists():
    print(f"Stage 1 data already exists at {output_path}, skipping extraction.")
    s_1_data = pickle.load(open(output_path, "rb"))
else:
    s_1_data = []
    for line in tqdm(dset, desc="Stage 1", total=len(dset)):
        processed = process_line(line, extractor, embedder, typer)
        s_1_data.append(processed)

    # save to disk
    pickle.dump(s_1_data, open(output_path, "wb"))
    print(f"Saved stage 1 data to {output_path}")

clusterer = OnlineRelationClusterer()
for sample in tqdm(s_1_data, desc="Clustering", total=len(s_1_data)):
    for triple in sample["data"]:
        clusterer.process_triple(triple)

print(f"Formed {len(clusterer.clusters)} clusters.")

learner = PragmaticEquivalenceLearner(mi_threshold=0.25, min_pairs=2)

equiv_classes, inverse_map = learner(clusterer)
print("Equivalence classes:", len(equiv_classes))
print("Inverse map:", len(inverse_map))

kg = NXKnowledgeGraph()
redundancy_filter = RedundancyFilter(kg, equiv_classes, inverse_map)

for h, r_surface, t, cid, sentence in clusterer.fact_list:
    added = redundancy_filter.add_if_novel(
        h, cid, t, surface=r_surface, sentence=sentence
    )
    # print only if redundant
    if not added:
        print(["REDUNDANT", "ACCEPTED"][added], h, r_surface, t, "→ cluster", cid)
print(f"Final KG has {len(kg.G.nodes())} nodes and {len(kg.G.edges())} edges.")

kg.save(Path(f"output/{dset_name}/final_kg.graphml"))
print(f"Saved KG to output/{dset_name}/final_kg.graphml")
