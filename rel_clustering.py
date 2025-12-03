from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Set
import math
import numpy as np
from pprint import pprint
from dotenv import load_dotenv
import os
from pathlib import Path
import openai
import torch
from tqdm import tqdm
from collections import defaultdict
import ast
import json
import pickle

from model.openai_model import OpenAIModel
from agent.core_agent import Agent
from typing import Union


load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


Triple = Tuple[str, str, str]  # (head, relation, tail)
TypePair = Tuple[str, str]  # (type(head), type(tail))
EmbeddingFn = Callable[[str, str, str, str], np.ndarray]
TypeFn = Callable[[str], str]
TripleExtractorFn = Callable[[str, int], List[Triple]]

triplet_path = Path.cwd() / "output" / "webnlg" / "triplets.txt"
data_path = Path.cwd() / "data" / "webnlg.txt"

triplets_text = triplet_path.read_text().splitlines()

all_triplets = [ast.literal_eval(line) for line in triplets_text]


model = OpenAIModel(
    model_name="openai/gpt-4o-mini",
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
    temperature=0.0,
)
type_function_prompt = """
You are an expert at classifying entities into types. Given an entity, return its type in one or two words. Be concise and specific.
Examples:
- "Barack Obama" -> "Person"
- "New York City" -> "Location"

Only return the type without any additional explanation.
Input: "{entity}"
Output:
"""

type_function_agent = Agent(llm=model, prompt=type_function_prompt)


def default_triple_extractor(sentence: str, idx=0) -> List[Triple]:
    """
    Stub: Extract (head, relation, tail) triples from a sentence.
    Replace with your actual IE model (OpenIE, SRL, custom, etc.).

    For now, returns [] so nothing happens unless you swap it out.
    """
    return all_triplets[idx]


def type_function(entity: str) -> str:
    """
    GPT-based type function.
    """
    return type_function_agent.run({"entity": entity}).strip().upper()


client = openai.OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)


def embedding_fn(
    head: str, relation: str, tail: str, triple_type: Union[str, Tuple[str, str]]
) -> np.ndarray:
    """
    GPT-based embedding function.
    """
    if isinstance(triple_type, tuple):
        triple_type = f"{triple_type[0]}->{triple_type[1]}"
    text = f"{head} {relation} {tail} [{triple_type}]"
    response = client.embeddings.create(input=[text], model="text-embedding-3-small")
    embeddings = [torch.tensor(data_point.embedding) for data_point in response.data]
    return torch.stack(embeddings).numpy()


# MI computation utility
def mutual_information_binary(N11, N10, N01, N00):
    """
    Compute mutual information I(X;Y) for binary events:
        X = "edge of cluster c1 under mapping M"
        Y = "edge of cluster c2 under mapping M"

    Where:
      N11 = count(X=1, Y=1)
      N10 = count(X=1, Y=0)
      N01 = count(X=0, Y=1)
      N00 = count(X=0, Y=0)

    Returns MI in bits.
    """
    N = N11 + N10 + N01 + N00
    if N == 0:
        return 0.0

    P11 = N11 / N
    P10 = N10 / N
    P01 = N01 / N
    P00 = N00 / N

    PX1 = P11 + P10
    PY1 = P11 + P01
    PX0 = 1 - PX1
    PY0 = 1 - PY1

    MI = 0.0

    def add_term(Pxy, Px, Py):
        return Pxy * math.log2(Pxy / (Px * Py)) if Pxy > 0 else 0.0

    MI += add_term(P11, PX1, PY1)
    MI += add_term(P10, PX1, PY0)
    MI += add_term(P01, PX0, PY1)
    MI += add_term(P00, PX0, PY0)

    return MI


@dataclass
class RelationClusterView:
    """
    Lightweight extensional view:
    - cluster_id
    - canonical type_pair
    - edges: set of (h, t)
    """

    cluster_id: int
    type_pair: TypePair
    edges: Set[Tuple[str, str]] = field(default_factory=set)


@dataclass
class RelationCluster:
    """
    Represents an induced relation type.

    Maintains:
    - mean embedding μ and diagonal variance σ^2 (online)
    - type_counts over (type(head), type(tail))
    - surface_relations: set of raw relation strings
    - cluster_id: int assigned by clusterer
    """

    cluster_id: int
    mean: np.ndarray
    var_diag: np.ndarray
    count: int

    type_counts: Dict[TypePair, int] = field(default_factory=dict)
    surface_relations: Set[str] = field(default_factory=set)
    eps: float = 1e-6

    @classmethod
    def from_first_example(
        cls,
        cluster_id: int,
        head: str,
        relation: str,
        tail: str,
        emb: np.ndarray,
        type_pair: TypePair,
    ) -> "RelationCluster":
        return cls(
            cluster_id=cluster_id,
            mean=emb.copy(),
            var_diag=np.ones_like(emb, dtype=np.float32),
            count=1,
            type_counts={type_pair: 1},
            surface_relations={relation},
        )

    def update(
        self,
        head: str,
        relation: str,
        tail: str,
        emb: np.ndarray,
        type_pair: TypePair,
    ) -> None:
        """
        Online update of:
        - mean & diag variance via Welford per-dimension
        - type_counts
        - surface_relations
        """
        self.surface_relations.add(relation)

        self.count += 1
        delta = emb - self.mean
        self.mean += delta / float(self.count)
        delta2 = emb - self.mean

        # Online diag variance: M2_diag / (count-1) = var_diag
        M2_diag = self.var_diag * (self.count - 2)  # previous (count-1) = current-2
        M2_diag += delta * delta2
        if self.count > 1:
            self.var_diag = M2_diag / float(self.count - 1)
        else:
            self.var_diag = np.ones_like(self.mean, dtype=np.float32)

        self.var_diag = np.maximum(self.var_diag, self.eps)

        self.type_counts[type_pair] = self.type_counts.get(type_pair, 0) + 1

    # ---- type distortion (argument-role compatibility) ----

    def canonical_type_pair(self) -> TypePair:
        """
        Return the most frequent type_pair (for MI / equivalence).
        """
        if not self.type_counts:
            return ("UNKNOWN", "UNKNOWN")
        return max(self.type_counts.items(), key=lambda kv: kv[1])[0]

    def type_probability(
        self,
        type_pair: TypePair,
        all_type_pairs: Sequence[TypePair],
        alpha_smooth: float = 1.0,
    ) -> float:
        K = len(all_type_pairs)
        total = sum(self.type_counts.values())
        count = self.type_counts.get(type_pair, 0)
        return (count + alpha_smooth) / (total + alpha_smooth * K)

    def type_distortion(
        self,
        type_pair: TypePair,
        all_type_pairs: Sequence[TypePair],
    ) -> float:
        """D_type = -log2 P(type_pair | cluster). Lower is better."""
        p = self.type_probability(type_pair, all_type_pairs)
        return -math.log(p + 1e-12, 2.0)

    # ---- semantic distortion (embedding distance / KL-ish) ----

    def semantic_distortion(self, emb: np.ndarray) -> float:
        """
        Mahalanobis-like distance:
        D_sem ≈ (x - μ)^T diag(1/σ^2) (x - μ)
        """
        diff = emb - self.mean
        inv_var = 1.0 / (self.var_diag + self.eps)
        return float(np.sum(diff * diff * inv_var))


class PragmaticEquivalenceLearner:
    """
    Learns pragmatic equivalence (same-direction or inverse) between
    induced relation clusters using extensional mutual information.
    """

    def __init__(self, mi_threshold: float = 0.25, min_shared_pairs: int = 2) -> None:
        self.mi_threshold = mi_threshold
        self.min_shared_pairs = min_shared_pairs

        self.views: Dict[int, RelationClusterView] = {}
        self.equivalence_classes: Dict[int, Set[int]] = defaultdict(set)
        self.inverse_map: Dict[int, int] = {}  # cluster_id -> inverse_cluster_id

    def ingest(
        self,
        clusters: List[RelationCluster],
        fact_list: List[Tuple[str, str, str, int]],
    ) -> None:
        """
        Build RelationClusterView objects from clusters and facts.
        """
        for c in clusters:
            self.views[c.cluster_id] = RelationClusterView(
                cluster_id=c.cluster_id,
                type_pair=c.canonical_type_pair(),
            )

        for h, r, t, cid in fact_list:
            if cid in self.views:
                self.views[cid].edges.add((h, t))

    def compute_equivalences(self) -> None:
        ids = sorted(self.views.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                v1 = self.views[ids[i]]
                v2 = self.views[ids[j]]

                same_dir = v1.type_pair == v2.type_pair
                inverse_dir = v1.type_pair == (v2.type_pair[1], v2.type_pair[0])

                if not (same_dir or inverse_dir):
                    continue

                if same_dir:
                    MI = self._mi_same_direction(v1, v2)
                    direction = "same"
                else:
                    MI = self._mi_inverse_direction(v1, v2)
                    direction = "inverse"

                if MI <= 0:
                    continue

                H1 = self._binary_entropy(v1.edges)
                H2 = self._binary_entropy(v2.edges)
                denom = min(H1, H2) if min(H1, H2) > 0 else 1.0
                NMI = MI / denom

                if NMI >= self.mi_threshold:
                    self.equivalence_classes[v1.cluster_id].add(v2.cluster_id)
                    self.equivalence_classes[v2.cluster_id].add(v1.cluster_id)
                    if direction == "inverse":
                        self.inverse_map[v1.cluster_id] = v2.cluster_id
                        self.inverse_map[v2.cluster_id] = v1.cluster_id

    def _mi_same_direction(
        self, v1: RelationClusterView, v2: RelationClusterView
    ) -> float:
        all_pairs = v1.edges.union(v2.edges)
        if len(all_pairs) < self.min_shared_pairs:
            return 0.0
        e1 = v1.edges
        e2 = v2.edges

        N11 = sum(1 for p in all_pairs if (p in e1 and p in e2))
        N10 = sum(1 for p in all_pairs if (p in e1 and p not in e2))
        N01 = sum(1 for p in all_pairs if (p in e2 and p not in e1))
        N00 = len(all_pairs) - (N11 + N10 + N01)
        return mutual_information_binary(N11, N10, N01, N00)

    def _mi_inverse_direction(
        self, v1: RelationClusterView, v2: RelationClusterView
    ) -> float:
        e1 = v1.edges
        e2_swapped = {(t, h) for (h, t) in v2.edges}
        all_pairs = e1.union(e2_swapped)
        if len(all_pairs) < self.min_shared_pairs:
            return 0.0

        N11 = sum(1 for p in all_pairs if (p in e1 and p in e2_swapped))
        N10 = sum(1 for p in all_pairs if (p in e1 and p not in e2_swapped))
        N01 = sum(1 for p in all_pairs if (p in e2_swapped and p not in e1))
        N00 = len(all_pairs) - (N11 + N10 + N01)
        return mutual_information_binary(N11, N10, N01, N00)

    @staticmethod
    def _binary_entropy(edge_set: Set[Tuple[str, str]]) -> float:
        N = len(edge_set)
        if N == 0:
            return 0.0
        P1 = N / (N + 1e-12)
        P0 = 1 - P1
        return -(P1 * math.log2(P1 + 1e-12) + P0 * math.log2(P0 + 1e-12))


############################################################
# Redundancy checker based on equivalence classes
############################################################


class PragmaticRedundancyChecker:
    """
    Uses learned pragmatic equivalence (same + inverse) to decide
    whether a new triple (h, cid, t) is redundant.
    """

    def __init__(self, learner: PragmaticEquivalenceLearner) -> None:
        self.learner = learner
        self.forward_edges = defaultdict(set)  # (h, cid) -> set(t)
        self.backward_edges = defaultdict(set)  # (t, cid) -> set(h)

    def add_fact(self, h: str, cid: int, t: str) -> None:
        self.forward_edges[(h, cid)].add(t)
        self.backward_edges[(t, cid)].add(h)

    def is_redundant(self, h: str, cid: int, t: str) -> bool:
        # Direct same-cluster fact
        if t in self.forward_edges.get((h, cid), set()):
            return True

        # Check equivalent clusters
        eq_class = self.learner.equivalence_classes.get(cid, set())

        for cid2 in eq_class:
            # same direction
            if t in self.forward_edges.get((h, cid2), set()):
                return True

            # inverse direction
            inv = self.learner.inverse_map.get(cid2)
            if inv is not None and inv == cid:
                if h in self.backward_edges.get((t, cid2), set()):
                    return True

        return False


class OnlineRelationClusterer:
    """
    Streaming clustering of relation instances into induced relation types.

    For each triple (h,r,t) extracted from sentences:
      - compute embedding
      - infer types (T1,T2)
      - cost(cluster) = w_sem * D_sem + w_type * D_type
      - if min cost < lambda_new: assign to cluster, else new cluster.

    Stores:
      - clusters (RelationCluster)
      - fact_list: (head, relation, tail, cluster_id) for all accepted facts
    """

    def __init__(
        self,
        embedding_fn: EmbeddingFn,
        triple_extractor: TripleExtractorFn = default_triple_extractor,
        type_fn: TypeFn = type_function,
        w_sem: float = 0.5,
        w_type: float = 1.0,
        lambda_new: float = 3.0,
    ) -> None:
        self.embedding_fn = embedding_fn
        self.triple_extractor = triple_extractor
        self.type_fn = type_fn

        self.w_sem = w_sem
        self.w_type = w_type
        self.lambda_new = lambda_new

        self.clusters: List[RelationCluster] = []
        self.all_type_pairs_ordered: List[TypePair] = []
        self._type_pair_index: Dict[TypePair, int] = {}
        self.fact_list: List[Tuple[str, str, str, int]] = []  # (h, r, t, cluster_id)

    # ---- type-pair registry ----

    def _register_type_pair(self, tp: TypePair) -> None:
        if tp not in self._type_pair_index:
            self._type_pair_index[tp] = len(self.all_type_pairs_ordered)
            self.all_type_pairs_ordered.append(tp)

    # ---- main API ----

    def process_sentences(self, sentences: Sequence[str]) -> None:
        for idx, sent in enumerate(sentences):
            triples = self.triple_extractor(sent, idx)
            for h, r, t in triples:
                self._process_triple(sent, h, r, t)

    def _process_triple(
        self, sentence: str, head: str, relation: str, tail: str
    ) -> None:
        emb = self.embedding_fn(head, relation, tail, sentence)
        t1 = self.type_fn(head)
        t2 = self.type_fn(tail)
        type_pair: TypePair = (t1, t2)
        self._register_type_pair(type_pair)

        # Compute best cluster by cost
        best_idx: Optional[int] = None
        best_cost: float = float("inf")

        for idx, cluster in enumerate(self.clusters):
            d_sem = cluster.semantic_distortion(emb)
            d_type = cluster.type_distortion(type_pair, self.all_type_pairs_ordered)
            cost = self.w_sem * d_sem + self.w_type * d_type
            if cost < best_cost:
                best_cost = cost
                best_idx = idx

        if best_idx is not None and best_cost < self.lambda_new:
            cluster = self.clusters[best_idx]
            cluster.update(head, relation, tail, emb, type_pair)
            cid = cluster.cluster_id
        else:
            cid = len(self.clusters)
            new_cluster = RelationCluster.from_first_example(
                cluster_id=cid,
                head=head,
                relation=relation,
                tail=tail,
                emb=emb,
                type_pair=type_pair,
            )
            self.clusters.append(new_cluster)

        # append fact with its assigned cluster
        self.fact_list.append((head, relation, tail, cid))

    # ---- inspection helpers ----

    def get_clusters_summary(self) -> List[Dict]:
        summaries = []
        for c in self.clusters:
            summaries.append(
                {
                    "cluster_id": c.cluster_id,
                    "surface_relations": sorted(c.surface_relations),
                    "n_triples": c.count,
                    "canonical_type_pair": c.canonical_type_pair(),
                    "type_counts": dict(c.type_counts),
                }
            )
        return summaries


def _type_pair_to_dict(tp: TypePair) -> Dict[str, str]:
    return {"head_type": tp[0], "tail_type": tp[1]}


def _serialize_cluster(cluster: RelationCluster) -> Dict:
    return {
        "cluster_id": cluster.cluster_id,
        "mean": cluster.mean.tolist(),
        "var_diag": cluster.var_diag.tolist(),
        "count": cluster.count,
        "surface_relations": sorted(cluster.surface_relations),
        "type_counts": [
            {"type_pair": _type_pair_to_dict(tp), "count": count}
            for tp, count in cluster.type_counts.items()
        ],
    }


def build_processing_artifacts(
    clusterer: OnlineRelationClusterer, learner: PragmaticEquivalenceLearner
) -> Dict:
    artifacts = {
        "clusters": [_serialize_cluster(c) for c in clusterer.clusters],
        "cluster_summaries": clusterer.get_clusters_summary(),
        "facts": [
            {"head": h, "relation": r, "tail": t, "cluster_id": cid}
            for h, r, t, cid in clusterer.fact_list
        ],
        "type_pairs": [_type_pair_to_dict(tp) for tp in clusterer.all_type_pairs_ordered],
        "equivalence_classes": {
            cid: sorted(eq) for cid, eq in learner.equivalence_classes.items()
        },
        "inverse_map": dict(learner.inverse_map),
    }
    return artifacts


def save_processing_artifacts(
    clusterer: OnlineRelationClusterer,
    learner: PragmaticEquivalenceLearner,
    output_file: Optional[Path] = None,
) -> Path:
    artifacts = build_processing_artifacts(clusterer, learner)
    if output_file is None:
        output_dir = Path.cwd() / "output" / "rel_clustering"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "artifacts.json"
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)

    output_file.write_text(json.dumps(artifacts, indent=2))
    return output_file


def save_clusterer_state(
    clusterer: OnlineRelationClusterer, output_file: Optional[Path] = None
) -> Path:
    if output_file is None:
        output_dir = Path.cwd() / "output" / "rel_clustering"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "clusterer.pkl"
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(pickle.dumps(clusterer))
    return output_file


def load_clusterer_state(input_file: Path) -> OnlineRelationClusterer:
    return pickle.loads(input_file.read_bytes())


def save_learner_state(
    learner: PragmaticEquivalenceLearner, output_file: Optional[Path] = None
) -> Path:
    if output_file is None:
        output_dir = Path.cwd() / "output" / "rel_clustering"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "learner.pkl"
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(pickle.dumps(learner))
    return output_file


def load_learner_state(input_file: Path) -> PragmaticEquivalenceLearner:
    return pickle.loads(input_file.read_bytes())


def save_redundancy_checker_state(
    checker: PragmaticRedundancyChecker, output_file: Optional[Path] = None
) -> Path:
    if output_file is None:
        output_dir = Path.cwd() / "output" / "rel_clustering"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "redundancy_checker.pkl"
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(pickle.dumps(checker))
    return output_file


def load_redundancy_checker_state(
    input_file: Path,
) -> PragmaticRedundancyChecker:
    return pickle.loads(input_file.read_bytes())


if __name__ == "__main__":
    # Tiny toy example: you will replace this with your real IE + embeddings.
    sentences = Path(data_path).read_text().splitlines()

    # Use random embeddings as placeholder; replace with OpenAI embeddings
    clusterer = OnlineRelationClusterer(
        embedding_fn=embedding_fn,
        triple_extractor=default_triple_extractor,
        type_fn=type_function,
        w_sem=0.5,
        w_type=1.0,
        lambda_new=3.0,
    )

    clusterer.process_sentences(sentences)

    print("=== Relation Clusters ===")
    for summary in clusterer.get_clusters_summary():
        print(summary)

    # Learn pragmatic equivalence from the current graph
    learner = PragmaticEquivalenceLearner(mi_threshold=0.3, min_shared_pairs=1)
    learner.ingest(clusterer.clusters, clusterer.fact_list)
    learner.compute_equivalences()

    print("\n=== Equivalence Classes ===")
    for cid, eq in learner.equivalence_classes.items():
        print(f"Cluster {cid} equivalent to {sorted(eq)}")
    print("\n=== Inverse Map ===")
    for cid, inv in learner.inverse_map.items():
        print(f"Cluster {cid} inverse of {inv}")

    # Redundancy checker
    red = PragmaticRedundancyChecker(learner)
    # Seed with existing facts
    for h, r, t, cid in clusterer.fact_list:
        red.add_fact(h, cid, t)

    artifact_path = save_processing_artifacts(clusterer, learner)
    print(f"\nSaved clustering artifacts to {artifact_path}")
    clusterer_state_path = save_clusterer_state(clusterer)
    learner_state_path = save_learner_state(learner)
    redundancy_state_path = save_redundancy_checker_state(red)
    print(f"Saved clusterer state to {clusterer_state_path}")
    print(f"Saved learner state to {learner_state_path}")
    print(f"Saved redundancy checker state to {redundancy_state_path}")
