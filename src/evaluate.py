import random
import numpy as np
import math
from collections import Counter, defaultdict


def sample_kg_edges(kg, k=50):
    edges = list(kg.edges(data=True))
    sample = random.sample(edges, min(k, len(edges)))

    formatted = []
    for h, t, data in sample:
        # GraphML might store clusters as a string, so normalize
        clusters = data.get("clusters", None)
        surfaces = data.get("surfaces", None)
        sentences = data.get("sentences", None)

        formatted.append(
            {
                "head": h,
                "tail": t,
                "clusters": clusters,
                "surface_forms": surfaces,
                "sentences": sentences,
            }
        )

    return formatted


def attach_provenance(sampled_edges, provenance_index):
    enriched = []
    for item in sampled_edges:
        h = item["head"]
        t = item["tail"]

        # find any provenance triples (relation ignored)
        prov_sentences = []
        for (hh, r, tt), sents in provenance_index.items():
            if hh == h and tt == t:
                prov_sentences.extend(sents)

        enriched.append({**item, "evidence_sentences": list(set(prov_sentences))})

    return enriched


def compute_precision(responses):
    correct = sum(1 for r in responses if r.lower().strip() == "yes")
    total = len(responses)
    precision = correct / total if total > 0 else 0
    return {"correct": correct, "total": total, "precision": precision}


def compute_semantic_coherence(clusterer, embedding_map):
    cid_to_embs = {cl.cluster_id: [] for cl in clusterer.clusters}

    for h, r, t, cid, _ in clusterer.fact_list:
        emb = embedding_map.get((h, r, t))
        if emb is not None:
            cid_to_embs[cid].append(emb)

    semico = {}
    for cl in clusterer.clusters:
        cid = cl.cluster_id
        embs = cid_to_embs[cid]
        if len(embs) == 0:
            semico[cid] = None
            continue

        mu = np.mean(embs, axis=0)
        dists = [np.sum((e - mu) ** 2) for e in embs]
        semico[cid] = float(np.mean(dists))

    return semico


def compute_type_entropy(clusterer):
    results = {}

    for cl in clusterer.clusters:
        cid = cl.cluster_id
        total = sum(cl.type_counts.values())

        H = 0.0
        for tp, cnt in cl.type_counts.items():
            p = cnt / total
            H -= p * math.log(p + 1e-12, 2)

        results[cid] = round(H, 1)

    return results


def build_edge_sets(clusterer):
    E = defaultdict(set)
    for h, r, t, cid, _ in clusterer.fact_list:
        E[cid].add((h, t))
    return E


def compute_inverse_alignment(E, inverse_map):
    results = {}

    for c, c_inv in inverse_map.items():
        edges_c = E[c]
        edges_inv = E[c_inv]

        if len(edges_c) == 0:
            results[(c, c_inv)] = None
            continue

        swapped_edges = {(t, h) for (h, t) in edges_c}
        overlap = swapped_edges & edges_inv

        score = len(overlap) / len(edges_c)
        results[(c, c_inv)] = score

    return results


def cluster_level_reduction(clusterer, kg):
    from collections import defaultdict

    cluster_counts_raw = defaultdict(int)
    cluster_counts_final = defaultdict(int)

    # Raw counts per cluster
    for _, _, _, cid, _ in clusterer.fact_list:
        cluster_counts_raw[cid] += 1

    # Final KG counts per cluster
    for h, t, data in kg.edges(data=True):
        clusters = data.get("clusters")
        if clusters is None:
            continue

        if isinstance(clusters, str):
            clusters = clusters.strip("{} ").strip("[]").split(",")
        clusters = set(int(c) for c in clusters if str(c).strip().isdigit())

        for cid in clusters:
            cluster_counts_final[cid] += 1

    # Compute reductions
    results = {}
    for cid in cluster_counts_raw:
        raw = cluster_counts_raw[cid]
        final = cluster_counts_final.get(cid, 0)
        red = 1 - (final / raw) if raw > 0 else None
        results[cid] = {"raw": raw, "final": final, "reduction": red}

    return results


def degree_summary(deg_list):
    deg = np.array(deg_list)
    return {
        "min": int(deg.min()),
        "max": int(deg.max()),
        "mean": float(deg.mean()),
        "median": float(np.median(deg)),
        "p90": float(np.percentile(deg, 90)),
        "p99": float(np.percentile(deg, 99)),
    }


def get_top_hubs(kg, n=10):
    sorted_nodes = sorted(kg.G.degree(), key=lambda x: x[1], reverse=True)
    return sorted_nodes[:n]


def component_entropy(component_nodes, type_fn):
    counts = Counter(type_fn(n) for n in component_nodes)
    total = sum(counts.values())

    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log(p + 1e-12, 2)
    return entropy, counts
