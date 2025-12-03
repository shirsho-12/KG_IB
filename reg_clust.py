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
import json
import pickle
import json

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

import ast

all_triplets = [ast.literal_eval(line) for line in triplets_text]

from model.openai_model import OpenAIModel
from agent.core_agent import Agent

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


from typing import Union


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


@dataclass
class RelationCluster:
    """
    Represents an induced relation type.

    Maintains:
    - mean embedding μ
    - diagonal variance estimate σ^2 (via online Welford updates)
    - multinomial over argument type pairs (type(head), type(tail))
    - set of surface relation strings that landed here
    """

    # running stats for embeddings
    mean: np.ndarray
    var_diag: np.ndarray
    count: int

    # argument-type distribution
    type_counts: Dict[TypePair, int] = field(default_factory=dict)

    # surface labels
    surface_relations: set = field(default_factory=set)

    # small constant for numerical stability
    eps: float = 1e-6

    @classmethod
    def from_first_example(
        cls,
        head: str,
        relation: str,
        tail: str,
        emb: np.ndarray,
        type_pair: TypePair,
    ) -> "RelationCluster":
        return cls(
            mean=emb.copy(),
            var_diag=np.ones_like(emb, dtype=np.float32),  # initial variance guess
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
        - mean and diag variance (Welford-like per-dimension)
        - type_counts
        - surface_relations
        """
        self.surface_relations.add(relation)

        # Welford update for diagonal covariance
        self.count += 1
        delta = emb - self.mean
        # update mean
        self.mean += delta / float(self.count)
        # recompute delta to new mean
        delta2 = emb - self.mean
        # online update of diagonal variance estimate
        # M2_diag = (count-1)*var_diag; var_new = M2_new / (count-1)
        M2_diag = self.var_diag * (self.count - 2)  # previous count-1 = current-2
        M2_diag += delta * delta2
        if self.count > 1:
            self.var_diag = M2_diag / float(self.count - 1)
        else:
            self.var_diag = np.ones_like(self.mean, dtype=np.float32)

        # keep some minimum variance
        self.var_diag = np.maximum(self.var_diag, self.eps)

        # update type distribution
        self.type_counts[type_pair] = self.type_counts.get(type_pair, 0) + 1

    # ---------- type distortion (argument-role compatibility) ----------

    def type_probability(
        self,
        type_pair: TypePair,
        all_type_pairs: Sequence[TypePair],
        alpha_smooth: float = 1.0,
    ) -> float:
        """
        Laplace-smoothed probability P(type_pair | cluster).
        """
        K = len(all_type_pairs)
        total = sum(self.type_counts.values())
        count = self.type_counts.get(type_pair, 0)
        return (count + alpha_smooth) / (total + alpha_smooth * K)

    def type_distortion(
        self,
        type_pair: TypePair,
        all_type_pairs: Sequence[TypePair],
    ) -> float:
        """
        D_type = -log2 P(type_pair | cluster).
        Lower is better.
        """
        p = self.type_probability(type_pair, all_type_pairs)
        return -math.log(p + 1e-12, 2.0)  # bits

    # ---------- semantic distortion (embedding distance / KL-ish) ----------

    def semantic_distortion(self, emb: np.ndarray) -> float:
        """
        Mahalanobis-like distance between emb and cluster Gaussian:

        D_sem ≈ (x - μ)^T diag(1/σ^2) (x - μ)

        This is proportional to KL(N(μ,σ^2) || N(x, σ0^2 I)) under an isotropic
        assumption, so it's a reasonable proxy for semantic KL.
        """
        diff = emb - self.mean
        inv_var = 1.0 / (self.var_diag + self.eps)
        return float(np.sum(diff * diff * inv_var))


# ---------------------------------------------------------------------------
# Online relation clusterer
# ---------------------------------------------------------------------------


class RelationClusterer:
    """
    Online clustering of relation instances into induced relation types.

    For each triple (h, r, t) in a sentence:
      - Compute an embedding embedding_fn(h, r, t, sentence)
      - Infer types T1 = type_fn(h), T2 = type_fn(t)
      - For each cluster, compute:
            cost = w_sem * D_sem + w_type * D_type
        where:
            D_sem  = semantic_distortion(embedding)
            D_type = -log P((T1,T2) | cluster)
      - If min_cluster_cost < lambda_new:
            assign to that cluster and update its stats
        else:
            create a new cluster

    This is a greedy, streaming approximation to an IB-style objective:
        minimize E[cost] + lambda_new * (#clusters)
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
        self.all_type_pairs: Dict[TypePair, None] = {}  # use as ordered set
        self.fact_list: List[Tuple[str, str, str, int]] = []

    # ---------- main entry point ----------

    def process_sentences(self, sentences: Sequence[str]) -> None:
        """
        Process a sequence of sentences, updating clusters online.
        """
        for idx, sent in tqdm(enumerate(sentences), total=len(sentences)):
            triples = self.triple_extractor(sent, idx)
            for h, r, t in triples:
                self._process_triple(sent, h, r, t)

    # ---------- internals ----------

    def _process_triple(
        self, sentence: str, head: str, relation: str, tail: str
    ) -> None:
        # 1) embedding
        emb = self.embedding_fn(head, relation, tail, sentence)

        # 2) argument types
        t1 = self.type_fn(head)
        t2 = self.type_fn(tail)
        type_pair: TypePair = (t1, t2)
        if type_pair not in self.all_type_pairs:
            self.all_type_pairs[type_pair] = None

        # 3) compute cost for each existing cluster
        best_idx: Optional[int] = None
        best_cost: float = float("inf")
        best_Dsem: Optional[float] = None
        best_Dtype: Optional[float] = None

        type_pair_list = list(self.all_type_pairs.keys())

        for idx, cluster in enumerate(self.clusters):
            d_sem = cluster.semantic_distortion(emb)
            d_type = cluster.type_distortion(type_pair, type_pair_list)
            cost = self.w_sem * d_sem + self.w_type * d_type

            if cost < best_cost:
                best_cost = cost
                best_idx = idx
                best_Dsem = d_sem
                best_Dtype = d_type

        # 4) decide: assign vs create new
        if best_idx is not None and best_cost < self.lambda_new:
            cluster = self.clusters[best_idx]
            cluster.update(head, relation, tail, emb, type_pair)
            # could log best_Dsem/best_Dtype here if desired
            cid = best_idx
        else:
            cid = len(self.clusters)
            new_cluster = RelationCluster.from_first_example(
                head=head,
                relation=relation,
                tail=tail,
                emb=emb,
                type_pair=type_pair,
            )
            self.clusters.append(new_cluster)

        self.fact_list.append((head, relation, tail, cid))

    # ---------- convenience methods / inspection ----------

    def get_clusters_summary(self) -> List[Dict]:
        """
        Return a light-weight summary of clusters for inspection.
        """
        summaries = []
        for idx, c in enumerate(self.clusters):
            summaries.append(
                {
                    "cluster_id": idx,
                    "surface_relations": sorted(c.surface_relations),
                    "n_triples": c.count,
                    "type_counts": dict(c.type_counts),
                }
            )
        return summaries

    def print_clusters(self, max_width: int = 120) -> None:
        """
        Pretty-print clusters in a compact way.
        """
        import textwrap

        for summary in self.get_clusters_summary():
            s = textwrap.shorten(str(summary), width=max_width)
            print(s)


def _type_pair_to_dict(tp: TypePair) -> Dict[str, str]:
    return {"head_type": tp[0], "tail_type": tp[1]}


def _serialize_cluster(idx: int, cluster: RelationCluster) -> Dict:
    return {
        "cluster_id": idx,
        "mean": cluster.mean.tolist(),
        "var_diag": cluster.var_diag.tolist(),
        "count": cluster.count,
        "surface_relations": sorted(cluster.surface_relations),
        "type_counts": [
            {"type_pair": _type_pair_to_dict(tp), "count": count}
            for tp, count in cluster.type_counts.items()
        ],
    }


def build_processing_artifacts(clusterer: RelationClusterer) -> Dict:
    return {
        "clusters": [
            _serialize_cluster(idx, cluster)
            for idx, cluster in enumerate(clusterer.clusters)
        ],
        "cluster_summaries": clusterer.get_clusters_summary(),
        "facts": [
            {"head": h, "relation": r, "tail": t, "cluster_id": cid}
            for h, r, t, cid in clusterer.fact_list
        ],
        "type_pairs": [_type_pair_to_dict(tp) for tp in clusterer.all_type_pairs.keys()],
    }


def save_processing_artifacts(
    clusterer: RelationClusterer, output_file: Optional[Path] = None
) -> Path:
    artifacts = build_processing_artifacts(clusterer)
    if output_file is None:
        output_dir = Path.cwd() / "output" / "reg_clust"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "artifacts.json"
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)

    output_file.write_text(json.dumps(artifacts, indent=2))
    return output_file


def save_clusterer_state(
    clusterer: RelationClusterer, output_file: Optional[Path] = None
) -> Path:
    if output_file is None:
        output_dir = Path.cwd() / "output" / "reg_clust"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "clusterer.pkl"
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)

    output_file.write_bytes(pickle.dumps(clusterer))
    return output_file


def load_clusterer_state(input_file: Path) -> RelationClusterer:
    return pickle.loads(input_file.read_bytes())


sentences = Path(data_path).read_text().splitlines()

clusterer = RelationClusterer(
    embedding_fn=embedding_fn,
    triple_extractor=default_triple_extractor,
    type_fn=type_function,
    w_sem=0.5,
    w_type=1.0,
    lambda_new=3.0,
)

clusterer.process_sentences(sentences)
clusterer.print_clusters()
artifact_path = save_processing_artifacts(clusterer)
print(f"Saved clustering artifacts to {artifact_path}")
clusterer_state_path = save_clusterer_state(clusterer)
print(f"Saved clusterer state to {clusterer_state_path}")
