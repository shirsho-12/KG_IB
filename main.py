from pprint import pprint

from pathlib import Path
from src.kg import NXKnowledgeGraph
from src.clusterer import OnlineRelationClusterer
from src.pragma import PragmaticEquivalenceLearner
from src.redundancy_filter import RedundancyFilter
from src.utils import process_line, get_agents

import argparse
from tqdm import tqdm
import pickle


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dset",
    type=str,
    required=True,
    choices=["rebel", "example", "webnlg", "wiki-nre"],
    help="Name of dataset",
)
args = parser.parse_args()
dset_name = args.dset
input_path = Path(f"data/{dset_name}.txt")

dset = input_path.read_text().splitlines()

output_path = Path(f"output/{dset_name}/tier_2")
output_file_path = output_path / "s_1_extracted.pkl"
output_path.mkdir(parents=True, exist_ok=True)
if output_file_path.exists():
    print(f"Stage 1 data already exists at {output_file_path}, skipping extraction.")
    s_1_data = pickle.load(open(output_file_path, "rb"))
output_path = Path(f"output/{dset_name}/s_1_extracted.pkl")
output_path.parent.mkdir(parents=True, exist_ok=True)
extractor, embedder, typer = get_agents()
if output_path.exists():
    print(f"Stage 1 data already exists at {output_path}, skipping extraction.")
    s_1_data = pickle.load(open(output_path, "rb"))
else:
    s_1_data = []
    for line in tqdm(dset, desc="Stage 1", total=len(dset)):
        processed = process_line(line, extractor, embedder, typer)
        s_1_data.append(processed)

    # save to disk
    pickle.dump(s_1_data, open(output_file_path, "wb"))
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

kg.save(output_path / "final_kg.graphml")
print(f"Saved KG to {output_path}/final_kg.graphml")
