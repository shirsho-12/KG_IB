import numpy as np
import math
from collections import defaultdict


class RelationCluster:
    """
    Online cluster representing an induced relation type.
    """

    def __init__(self, cluster_id, emb, type_pair):
        self.cluster_id = cluster_id

        # Statistics
        self.mean = emb.astype(np.float32)
        self.var_diag = np.ones_like(emb, dtype=np.float32)
        self.count = 1

        # Type distribution
        self.type_counts = defaultdict(int)
        self.type_counts[type_pair] += 1

        # Surface relation forms
        self.surface_forms = set()  # defaultdict(int)

    def update(self, emb, type_pair, surface_relation=None):
        """
        Online update of mean and diag-variance (Welford estimator).
        """
        self.count += 1
        # if surface_relation:
        #     self.surface_forms[surface_relation] += 1
        delta = emb - self.mean
        self.mean += delta / self.count
        delta2 = emb - self.mean

        # Update diag variance (M2 / (n-1))
        M2 = self.var_diag * (self.count - 2) + delta * delta2
        if self.count > 1:
            self.var_diag = M2 / (self.count - 1)

        self.var_diag = np.maximum(self.var_diag, 1e-6)
        self.type_counts[type_pair] += 1

    def semantic_distortion(self, emb):
        diff = emb - self.mean
        inv_var = 1.0 / (self.var_diag + 1e-6)
        return float(np.sum(diff * diff * inv_var))

    def type_probability(self, type_pair, all_types, alpha=1.0):
        total = sum(self.type_counts.values())
        count = self.type_counts[type_pair]
        K = len(all_types)
        return (count + alpha) / (total + alpha * K)

    def type_distortion(self, type_pair, all_types):
        p = self.type_probability(type_pair, all_types)
        return -math.log(p + 1e-12, 2.0)

    def __str__(self) -> str:
        return f"Cluster {self.cluster_id}: {self.count} elements"

    def __repr__(self) -> str:
        return self.__str__()


class OnlineRelationClusterer:
    def __init__(self, lambda_sem=0.5, lambda_type=1.0, lambda_new=3.0):
        self.lambda_sem = lambda_sem
        self.lambda_type = lambda_type
        self.lambda_new = lambda_new

        self.clusters = []
        self.all_type_pairs = set()
        self.fact_list = []  # (head, relation, tail, cluster_id)

    def _register_type_pair(self, tp):
        self.all_type_pairs.add(tp)

    def process_triple(self, triple):
        h = triple["head"]
        r = triple["relation"]
        t = triple["tail"]
        emb = triple["embedding"]
        type_pair = triple["type_pair"]

        self._register_type_pair(type_pair)

        # If no clusters, create one
        if len(self.clusters) == 0:
            cl = RelationCluster(0, emb, type_pair)
            cl.surface_forms.add(r)
            self.clusters.append(cl)
            self.fact_list.append((h, r, t, 0))
            return 0

        # Otherwise compute distortion for each cluster
        best_cost = float("inf")
        best_idx = None

        for idx, cl in enumerate(self.clusters):
            d_sem = cl.semantic_distortion(emb)
            d_type = cl.type_distortion(type_pair, self.all_type_pairs)
            cost = self.lambda_sem * d_sem + self.lambda_type * d_type

            if cost < best_cost:
                best_cost = cost
                best_idx = idx

        # Decide: assign or create new
        if best_cost < self.lambda_new and best_idx is not None:
            cl = self.clusters[best_idx]
            cl.update(emb, type_pair)
            cl.surface_forms.add(r)
            self.fact_list.append((h, r, t, best_idx))
            return best_idx

        else:
            new_id = len(self.clusters)
            cl = RelationCluster(new_id, emb, type_pair)
            cl.surface_forms.add(r)
            self.clusters.append(cl)
            self.fact_list.append((h, r, t, new_id))
            return new_id
