from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
from pprint import pprint
from dotenv import load_dotenv
import os
from pathlib import Path

# import openai
from tqdm import tqdm
from collections import defaultdict, Counter

# load_dotenv()
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

from src.clusterer import OnlineRelationClusterer
from src.redundancy_filter import RedundancyFilter
from src.kg import NXKnowledgeGraph
from src.pragma import PragmaticEquivalenceLearner
import networkx as nx

from src.evaluate import (
    sample_kg_edges,
    attach_provenance,
    compute_precision,
    compute_semantic_coherence,
    compute_type_entropy,
    build_edge_sets,
    compute_inverse_alignment,
    cluster_level_reduction,
    degree_summary,
    get_top_hubs,
    component_entropy,
)


import pickle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="webnlg")
parser.add_argument("--mi_threshold", type=float, default=0.25)
parser.add_argument("--min_pairs", type=int, default=2)
parser.add_argument("--lambda_sem", type=float, default=0.5)
parser.add_argument("--lambda_type", type=float, default=0.5)
parser.add_argument("--lambda_new", type=float, default=1.0)


args = parser.parse_args()

if args.dataset == "webnlg":
    dataset_path = Path("output/webnlg/s_1_extracted.pkl")
elif args.dataset == "rebel":
    dataset_path = Path("output/rebel/s_1_extracted.pkl")
elif args.dataset == "wiki-nre":
    dataset_path = Path("output/wiki-nre/s_1_extracted.pkl")
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

with open(dataset_path, "rb") as f:
    step1_data = pickle.load(f)

# kg = NXKnowledgeGraph("output/output/webnlg/final_kg.graphml")

kg = NXKnowledgeGraph()

clusterer = OnlineRelationClusterer(
    lambda_sem=args.lambda_sem, lambda_type=args.lambda_type, lambda_new=args.lambda_new
)


for sample in tqdm(step1_data):
    for t in sample["data"]:
        t["sentence"] = sample["sentence"]
        clusterer.process_triple(t)
learner = PragmaticEquivalenceLearner(
    mi_threshold=args.mi_threshold, min_pairs=args.min_pairs
)
equiv_classes, inverse_map = learner(clusterer)
rf = RedundancyFilter(kg, equiv_classes, inverse_map)

fail_list = []
count = 0
for h, r_surface, t, cid, sentence in tqdm(
    clusterer.fact_list, desc="Filtering redundancy"
):
    added = rf.add_if_novel(h, cid, t, surface=r_surface, sentence=sentence)
    # print only if redundant
    if not added:
        fail_list.append((h, r_surface, t, cid, sentence))
        count += 1


print(f"Filtered out {count} redundant edges. Total edges now: {len(kg.G.edges)}")
save_path = Path.cwd() / "output" / args.dataset / "distortion_analysis"
save_path.mkdir(parents=True, exist_ok=True)
kg.save(
    save_path
    / f"final_kg_mi_{args.lambda_sem}_{args.lambda_type}_{args.lambda_new}.graphml"
)

embedding_map = {}

for entry in step1_data:
    for d in entry["data"]:
        embedding_map[(d["head"], d["relation"], d["tail"])] = d["embedding"]

semantic_coherence = compute_semantic_coherence(clusterer, embedding_map)
for k, v in semantic_coherence.items():
    if v > 0.0:
        print(f"Cluster {k}: Semantic Coherence = {v:.4f}")

type_entropy = compute_type_entropy(clusterer)
counts = {0: 0, 1: 0, 2: 0}
for k, v in type_entropy.items():
    counts[v] += 1
print("Type Entropy Distribution:")
for entropy_level, cnt in counts.items():
    print(f"Entropy {entropy_level}: {cnt} clusters")

import pandas as pd

df = pd.DataFrame(
    {
        "cluster_id": list(semantic_coherence.keys()),
        "semantic_coherence": list(semantic_coherence.values()),
        "type_entropy": [type_entropy[cid] for cid in semantic_coherence],
    }
)

df.sort_values("semantic_coherence", ascending=False)
print(df.head(10))
N_raw = 0
for entry in step1_data:
    N_raw += len(entry["triples"])

N_final = kg.G.number_of_edges()
reduction = 1 - (N_final / N_raw)
print({"raw_triples": N_raw, "final_edges": N_final, "reduction_rate": reduction})


# Degree (undirected sense)
degrees = [deg for _, deg in kg.G.degree()]

# Directed in/out degrees
in_degrees = [deg for _, deg in kg.G.in_degree()]
out_degrees = [deg for _, deg in kg.G.out_degree()]
summary_total = degree_summary(degrees)
summary_in = degree_summary(in_degrees)
summary_out = degree_summary(out_degrees)

print("Degree Summary (Total, In, Out):")
print(summary_total, summary_in, summary_out)
print("\nTop 10 Hubs:")
print(get_top_hubs(kg, n=10))


def get_type(ent):
    return entity_type_map.get(ent, "OTHER")


import matplotlib.pyplot as plt


def plot_degree_hist(data, title, ax):
    ax.hist(data, bins=40)
    ax.set_title(title)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Frequency")


def plot_loglog(data, title, ax):
    data = np.array(data)
    data = data[data > 0]  # log can't take zeros
    ax.scatter(
        np.log10(np.arange(1, len(data) + 1)), np.log10(np.sort(data)[::-1]), s=3
    )
    ax.set_title(title + " (Log-Log Plot)")
    ax.set_xlabel("log10(rank)")
    ax.set_ylabel("log10(degree)")


# Create figure with 6 subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Network Analysis", fontsize=16)

# Plot 1: Total Degree Distribution
plot_degree_hist(degrees, "Total Degree Distribution", axes[0, 0])

# Plot 2: In-Degree Distribution
plot_degree_hist(in_degrees, "In-Degree Distribution", axes[0, 1])

# Plot 3: Out-Degree Distribution
plot_degree_hist(out_degrees, "Out-Degree Distribution", axes[0, 2])

# Plot 4: Degree Distribution Log-Log
plot_loglog(degrees, "Degree Distribution", axes[1, 0])

# Plot 5: Component Size Distribution
components = list(nx.connected_components(kg.G.to_undirected()))
component_sizes = [len(c) for c in components]
plot_degree_hist(component_sizes, "Component Size Distribution", axes[1, 1])

# Plot 6: Component Size vs Entropy
entity_type_counts = defaultdict(Counter)
for entry in step1_data:
    for d in entry["data"]:
        h = d["head"]
        t = d["tail"]
        th, tt = d["type_pair"]
        entity_type_counts[h][th] += 1
        entity_type_counts[t][tt] += 1

entity_type_map = {}
for ent, counter in entity_type_counts.items():
    entity_type_map[ent] = counter.most_common(1)[0][0]

G_undirected = kg.G.to_undirected()
components = list(nx.connected_components(G_undirected))
components_sorted = sorted(components, key=lambda c: len(c), reverse=True)

component_stats = []
for i, comp in enumerate(components_sorted):
    ent, counts = component_entropy(comp, lambda e: entity_type_map.get(e, "OTHER"))
    component_stats.append({"component_id": i, "size": len(comp), "entropy_bits": ent})

sizes = [c["size"] for c in component_stats]
entropies = [c["entropy_bits"] for c in component_stats]
axes[1, 2].scatter(sizes, entropies)
axes[1, 2].set_xlabel("Component Size")
axes[1, 2].set_ylabel("Type Entropy (bits)")
axes[1, 2].set_title("Component Size vs Entropy")

plt.tight_layout()
plt.savefig(
    f"{save_path}/network_analysis_{args.lambda_sem}_{args.lambda_type}_{args.lambda_new}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
