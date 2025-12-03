"""
- PMI for each triple (entity pair)
- Entropy of each relation
- Mutual information between entity-pair variable E and relation variable R
- InfoScore = PMI - beta * H(relation)
"""

import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import pickle
from typing import Any, Optional, TYPE_CHECKING

EPS = 1e-12

if TYPE_CHECKING:
    from rel_clustering import (
        OnlineRelationClusterer,
        PragmaticEquivalenceLearner,
        PragmaticRedundancyChecker,
    )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_pickle(obj: Any, output_file: Path) -> Path:
    _ensure_parent(output_file)
    output_file.write_bytes(pickle.dumps(obj))
    return output_file


def _load_pickle(input_file: Path) -> Any:
    return pickle.loads(input_file.read_bytes())


def save_clusterer_state(
    clusterer: "OnlineRelationClusterer", output_file: Optional[Path] = None
) -> Path:
    output_file = (
        output_file
        if output_file is not None
        else Path.cwd() / "output" / "ib" / "clusterer.pkl"
    )
    return _save_pickle(clusterer, output_file)


def load_clusterer_state(input_file: Path) -> "OnlineRelationClusterer":
    return _load_pickle(input_file)


def save_learner_state(
    learner: "PragmaticEquivalenceLearner", output_file: Optional[Path] = None
) -> Path:
    output_file = (
        output_file
        if output_file is not None
        else Path.cwd() / "output" / "ib" / "learner.pkl"
    )
    return _save_pickle(learner, output_file)


def load_learner_state(input_file: Path) -> "PragmaticEquivalenceLearner":
    return _load_pickle(input_file)


def save_redundancy_checker_state(
    checker: "PragmaticRedundancyChecker", output_file: Optional[Path] = None
) -> Path:
    output_file = (
        output_file
        if output_file is not None
        else Path.cwd() / "output" / "ib" / "redundancy_checker.pkl"
    )
    return _save_pickle(checker, output_file)


def load_redundancy_checker_state(
    input_file: Path,
) -> "PragmaticRedundancyChecker":
    return _load_pickle(input_file)


def compute_counts(triplets):
    """Compute counts needed for probability estimates."""
    n = len(triplets)
    counts = defaultdict(Counter)
    # entity count, pair count, relation count, triplet count
    for e1, r, e2 in triplets:
        counts[0][e1] += 1
        counts[0][e2] += 1
        counts[1][(e1, e2)] += 1
        counts[2][r] += 1
        counts[3][(e1, r, e2)] += 1
    return {
        "n": n,
        "count_e": counts[0],
        "count_pair": counts[1],
        "count_r": counts[2],
        "count_triplet": counts[3],
    }


def p_entity(e, counts):
    # Probability of entity appearing (as an endpoint in a triple)
    total_entity_mentions = sum(counts["count_e"].values())
    return counts["count_e"][e] / (total_entity_mentions + EPS)


def p_pair(pair, counts):
    # Probability of the ordered pair appearing in the triple set
    total_pairs = counts["n"]
    return counts["count_pair"][pair] / (total_pairs + EPS)


def p_pair_given_r(pair, r, counts):
    # P(pair | r)
    cr = counts["count_r"][r]
    if cr == 0:
        return 0.0
    triplet = (pair[0], r, pair[1])
    return counts["count_triplet"][triplet] / (cr + EPS)


def p_r(r, counts):
    # Probability of relation r appearing in the triple set
    return counts["count_r"][r] / (counts["n"] + EPS)


def p_triplet(triplet, counts):
    # Probability of the triplet appearing in the triple set
    return counts["count_triplet"][triplet] / (counts["n"] + EPS)


def pmi_pair(pair, counts, log_base=2):
    # Pointwise Mutual Information for a pair of entities
    e1, e2 = pair
    p_e1 = p_entity(e1, counts)
    p_e2 = p_entity(e2, counts)
    p_e1_e2 = p_pair(pair, counts)
    if p_e1_e2 == 0:
        return 0
    return np.log(p_e1_e2 / (p_e1 * p_e2) + EPS) / np.log(log_base)


def relation_entropy(r, counts, log_base=2):
    # Entropy of relation r
    cr = counts["count_r"][r]
    if cr == 0:
        return 0.0
    ent = 0.0
    for pair in counts["count_pair"]:
        p = p_pair_given_r(pair, r, counts)
        if p > 0:
            ent += -p * np.log(p + EPS) / np.log(log_base)
    return ent


def pair_entropy(counts, log_base=2):
    # Entropy of the entity-pair variable
    ent = 0.0
    for pair in counts["count_pair"]:
        p = p_pair(pair, counts)
        if p > 0:
            ent += -p * np.log(p + EPS) / np.log(log_base)
    return ent


def pair_entropy_given_r(counts, log_base=2):
    # Conditional entropy H(E | R)
    ent = 0.0
    for relation, cr in counts["count_r"].items():
        if cr == 0:
            continue
        ent_relation = relation_entropy(relation, counts, log_base)
        ent += p_r(relation, counts) * ent_relation
    return ent


def mutual_information(counts, log_base=2):
    # Mutual Information I(E; R)
    h_pair = pair_entropy(counts, log_base)
    h_pair_given_r = pair_entropy_given_r(counts, log_base)
    return h_pair - h_pair_given_r


def info_score(triple, counts, beta=0.1):
    # Information score for a given triple
    e1, r, e2 = triple
    pair = (e1, e2)
    pmi = pmi_pair(pair, counts)
    rel_entropy = relation_entropy(r, counts)
    return {
        "triple": triple,
        "pmi": pmi,
        "relation_entropy": rel_entropy,
        "info_score": pmi - beta * rel_entropy,
    }


def get_stats(counts, beta=0.1):
    """Compute relation-level aggregated statistics."""
    rel_stats = {}
    pmi_by_relation = defaultdict(list)
    for triplet, c in counts["count_triplet"].items():
        e1, r, e2 = triplet
        pair = (e1, e2)
        pmi = pmi_pair(pair, counts)
        pmi_by_relation[r].append((pair, pmi, c))
    for r, items in pmi_by_relation.items():
        pmis = [pmi for _pair, pmi, _c in items]
        mean_pmi = np.mean(pmis)
        median_pmi = np.median(pmis)
        rel_entropy = relation_entropy(r, counts)
        agg_info_score = mean_pmi - beta * rel_entropy
        rel_stats[r] = {
            "mean_pmi": mean_pmi,
            "median_pmi": median_pmi,
            "relation_entropy": rel_entropy,
            "info_score": agg_info_score,
            "num_triplets": len(items),
            "count_r": counts["count_r"][r],
        }
    return rel_stats
