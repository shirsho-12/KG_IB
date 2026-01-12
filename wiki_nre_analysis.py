from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
from pprint import pprint
from dotenv import load_dotenv
import os
from pathlib import Path
import openai
from tqdm import tqdm
from collections import defaultdict

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

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

triplet_path = Path.cwd() / "output" / "wiki-nre" / "tier_2" / "triplets.txt"
data_path = Path.cwd() / "data" / "wiki-nre.txt"

triplets_text = triplet_path.read_text().splitlines()

import ast

all_triplets = [ast.literal_eval(line) for line in triplets_text]

from model.openai_model import OpenAIModel
from src.agent import EntityTypingAgent

model = OpenAIModel(
    model_name="openai/gpt-4o-mini",
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
    temperature=0.0,
)
type_function_agent = EntityTypingAgent(
    llm=model,
)


with open("output/wiki-nre/tier_2/s_1_extracted.pkl", "rb") as f:
    step1_data = pickle.load(f)

# kg = NXKnowledgeGraph("output/output/webnlg/final_kg.graphml")

kg = NXKnowledgeGraph()

clusterer = OnlineRelationClusterer()


for sample in tqdm(step1_data):
    for t in sample["data"]:
        t["sentence"] = sample["sentence"]
        clusterer.process_triple(t)
learner = PragmaticEquivalenceLearner(mi_threshold=0.25, min_pairs=2)
equiv_classes, inverse_map = learner(clusterer)
rf = RedundancyFilter(kg, equiv_classes, inverse_map)

for h, r_surface, t, cid, sentence in clusterer.fact_list:
    added = rf.add_if_novel(h, cid, t, surface=r_surface, sentence=sentence)
    # print only if redundant
    if not added:
        print(["REDUNDANT", "ACCEPTED"][added], h, r_surface, t, "→ cluster", cid)
kg.save("output/wiki-nre/final_kg_2.graphml")

import ast

lines = (Path.cwd() / "data" / "wiki-nre.txt").read_text().splitlines()
id_dct = {line: i for i, line in enumerate(lines)}
triplet_arr = [[] for _ in range(1165)]
for u, v, d in tqdm(kg.G.edges(data=True)):
    surfaces = d["surfaces"]
    # if d["sentences"] != []:
    #     print(d["sentences"])
    # print(d)
    # break
    if len(surfaces) > 1:
        print(u, v, surfaces, len(surfaces))
    for sent in d["sentences"]:
        for rel in surfaces:
            triplet_arr[id_dct[sent]].append([u, rel, v])
for i in range(len(triplet_arr)):
    triplet_arr[i].sort()

with open("output/wiki-nre/final_triplets.txt", "w") as f:
    for triplets in triplet_arr:
        f.write(str(triplets) + "\n")

from collections import defaultdict

provenance = defaultdict(list)

for entry in step1_data:
    sentence = entry["sentence"]
    for h, r, t in entry["triples"]:
        provenance[(h, r, t)].append(sentence)


sampled = sample_kg_edges(kg.G, k=200)
sampled[:3]

evaluation_set = attach_provenance(sampled, provenance)
evaluation_set[:3]

from src.agent import EvaluationAgent

eval_agent = EvaluationAgent(model)
responses = []
for obj in tqdm(evaluation_set):
    resp = eval_agent.run(
        {
            "head": obj["head"],
            "tail": obj["tail"],
            "sentences": " ".join(obj["evidence_sentences"][:10]),
        }
    )
    responses.append(resp)


for obj, resp in zip(evaluation_set, responses):
    if not resp.lower().startswith("yes"):
        print(
            f"Failed: {obj['head']} - {obj['surface_forms']} - {obj['tail']} \n {obj['evidence_sentences'][0]}"
        )


metrics = compute_precision(responses)
print(metrics)


step1_flat = []

for entry in step1_data:
    for d in entry["data"]:
        step1_flat.append(d)

print("Total triple instances:", len(step1_flat))
step1_flat[0]

clusterer = OnlineRelationClusterer()

for triple in step1_flat:
    clusterer.process_triple(triple)

print("Induced clusters:", len(clusterer.clusters))

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
print(counts)

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


peq = PragmaticEquivalenceLearner(mi_threshold=0.25, min_pairs=2)
equiv_classes, inverse_map = peq.compute(clusterer)

print(inverse_map)


E = build_edge_sets(clusterer)
inverse_alignment_scores = compute_inverse_alignment(E, inverse_map)
print(inverse_alignment_scores)

N_raw = 0
for entry in step1_data:
    N_raw += len(entry["triples"])

N_final = kg.G.number_of_edges()
reduction = 1 - (N_final / N_raw)
print(f"Raw triples: {N_raw}, Final edges: {N_final}, Reduction rate: {reduction:.4f}")


cluster_reduction = cluster_level_reduction(clusterer, kg.G)
for k, v in list(cluster_reduction.items())[:5]:
    print(f"Cluster {k}: {v}")

    # Degree (undirected sense)
degrees = [deg for _, deg in kg.G.degree()]

# Directed in/out degrees
in_degrees = [deg for _, deg in kg.G.in_degree()]
out_degrees = [deg for _, deg in kg.G.out_degree()]

import matplotlib.pyplot as plt


def plot_degree_hist(data, title):
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=40)
    plt.title(title)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")


plot_degree_hist(degrees, "Total Degree Distribution")
plot_degree_hist(in_degrees, "In-Degree Distribution")
plot_degree_hist(out_degrees, "Out-Degree Distribution")


def plot_loglog(data, title):
    data = np.array(data)
    data = data[data > 0]  # log can't take zeros

    plt.figure(figsize=(6, 4))
    plt.scatter(
        np.log10(np.arange(1, len(data) + 1)), np.log10(np.sort(data)[::-1]), s=3
    )
    plt.title(title + " (Log-Log Plot)")
    plt.xlabel("log10(rank)")
    plt.ylabel("log10(degree)")
    plt.show()
    plt.savefig(f"{title.replace(' ', '_').lower()}_loglog.png")


plot_loglog(degrees, "Degree Distribution")

summary_total = degree_summary(degrees)
summary_in = degree_summary(in_degrees)
summary_out = degree_summary(out_degrees)

print("Degree Summary (Total, In, Out):")
print(summary_total, summary_in, summary_out)
print("\nTop 10 Hubs:")
print(get_top_hubs(kg, n=20))


components = list(nx.connected_components(kg.G.to_undirected()))
component_sizes = [len(c) for c in components]

plot_degree_hist(component_sizes, "Component Size Distribution")

from collections import Counter

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

print(len(entity_type_map))


def get_type(ent):
    return entity_type_map.get(ent, "OTHER")


G_undirected = kg.G.to_undirected()
components = list(nx.connected_components(G_undirected))
print("Number of components:", len(components))
components_sorted = sorted(components, key=lambda c: len(c), reverse=True)


component_stats = []

for i, comp in enumerate(components_sorted):
    ent, counts = component_entropy(comp, get_type)
    component_stats.append(
        {
            "component_id": i,
            "size": len(comp),
            "entropy_bits": ent,
            "type_distribution": counts,
        }
    )

print(component_stats[:5])


sizes = [c["size"] for c in component_stats]
entropies = [c["entropy_bits"] for c in component_stats]

plt.figure(figsize=(6, 4))
plt.scatter(sizes, entropies)
plt.xlabel("Component Size")
plt.ylabel("Type Entropy (bits)")
plt.title("Component Size vs Entropy")
plt.show()
plt.savefig("component_size_vs_entropy.png")

for comp in component_stats[:10]:
    print(
        f"Component {comp['component_id']}: size={comp['size']}, entropy={comp['entropy_bits']:.3f}"
    )
    print("  Top types:", comp["type_distribution"].most_common(5))
    print()
