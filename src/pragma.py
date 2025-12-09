from collections import defaultdict
import math

from src.clusterer import OnlineRelationClusterer


class PragmaticEquivalenceLearner:
    """
    Learns:
      - equivalence_classes[z]: clusters pragmatically equivalent (same direction)
      - inverse_map[z]: single cluster that is an inverse of z (if any)
    """

    def __init__(self, mi_threshold=0.25, min_pairs=1):
        self.mi_threshold = mi_threshold
        self.min_pairs = min_pairs
        self.equivalence_classes = defaultdict(set)  # same direction
        self.inverse_map = {}  # z -> inverse_z

    def _binary_mi(self, E1, E2, universe):
        if not universe:
            return 0.0

        N11 = N10 = N01 = N00 = 0

        for p in universe:
            x = p in E1
            y = p in E2
            if x and y:
                N11 += 1
            elif x and not y:
                N10 += 1
            elif (not x) and y:
                N01 += 1
            else:
                N00 += 1

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

        def term(Pxy, Px, Py):
            if Pxy == 0 or Px == 0 or Py == 0:
                return 0.0
            return Pxy * math.log(Pxy / (Px * Py), 2)

        I = 0.0
        I += term(P11, PX1, PY1)
        I += term(P10, PX1, PY0)
        I += term(P01, PX0, PY1)
        I += term(P00, PX0, PY0)
        return I

    def _binary_entropy(self, E, universe):
        if not universe:
            return 0.0
        p1 = sum(1 for u in universe if u in E) / len(universe)
        p0 = 1 - p1
        H = 0.0
        if p1 > 0:
            H -= p1 * math.log(p1, 2)
        if p0 > 0:
            H -= p0 * math.log(p0, 2)
        return H

    def _nmi(self, E1, E2, universe):
        I = self._binary_mi(E1, E2, universe)
        H1 = self._binary_entropy(E1, universe)
        H2 = self._binary_entropy(E2, universe)
        denom = min(H1, H2)
        if denom == 0:
            return 0.0
        return I / denom

    def compute(self, clusterer: OnlineRelationClusterer):
        """
        clusterer: your OnlineRelationClusterer (from Step 2)
        Uses clusterer.fact_list: (h, r_surface, t, cid)
        and clusterer.clusters: RelationCluster objects
        """
        # Build extensional edge sets per cluster
        edge_sets = defaultdict(set)  # cid -> {(h,t)}
        for h, r, t, cid in clusterer.fact_list:
            edge_sets[cid].add((h, t))

        # Precompute swapped edges for inverse direction
        swapped_sets = {
            cid: {(t, h) for (h, t) in edges} for cid, edges in edge_sets.items()
        }

        # Universe: all pairs seen in any cluster or inverse
        universe = set()
        for E in edge_sets.values():
            universe |= E
        for E in swapped_sets.values():
            universe |= E

        cids = sorted(edge_sets.keys())
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                ci, cj = cids[i], cids[j]
                E_i = edge_sets[ci]
                E_j = edge_sets[cj]
                E_j_swapped = swapped_sets[cj]

                # Skip if too few pairs in union
                if len(E_i | E_j | E_j_swapped) < self.min_pairs:
                    continue

                n_same = self._nmi(E_i, E_j, universe)
                n_inv = self._nmi(E_i, E_j_swapped, universe)

                # Same-direction equivalence
                if n_same >= self.mi_threshold:
                    self.equivalence_classes[ci].add(cj)
                    self.equivalence_classes[cj].add(ci)

                # Inverse-direction equivalence
                if n_inv >= self.mi_threshold:
                    # we only keep single inverse partner per cluster (can relax)
                    self.inverse_map[ci] = cj
                    self.inverse_map[cj] = ci

        return self.equivalence_classes, self.inverse_map

    def __call__(self, clusters):
        return self.compute(clusters)
