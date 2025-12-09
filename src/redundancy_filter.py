from src.kg import NXKnowledgeGraph


class RedundancyFilter:
    """
    Implements redundancy checks w.r.t. same-direction equivalence
    and inverse-direction equivalence.
    """

    def __init__(self, kg: NXKnowledgeGraph, equiv_classes, inverse_map):
        self.kg = kg
        self.equiv_classes = equiv_classes  # dict cid -> set[cid]
        self.inverse_map = inverse_map  # dict cid -> cid

    def is_redundant(self, h, cid, t):
        G = self.kg

        # 1. Direct: h -cid-> t already exists
        if G.has_edge_with_cluster(h, cid, t):
            return True

        # 2. Same-direction equivalents: h -cid_eq-> t exists
        for cid_eq in self.equiv_classes.get(cid, []):
            if G.has_edge_with_cluster(h, cid_eq, t):
                return True

        # 3. Inverse: t -cid_inv-> h exists
        cid_inv = self.inverse_map.get(cid)
        if cid_inv is not None:
            if G.has_inverse_edge_with_cluster(h, cid_inv, t):
                return True

        # 4. Inverse-of-equivalents
        for cid_eq in self.equiv_classes.get(cid, []):
            cid_eq_inv = self.inverse_map.get(cid_eq)
            if cid_eq_inv is not None:
                if G.has_inverse_edge_with_cluster(h, cid_eq_inv, t):
                    return True

        return False

    def add_if_novel(self, h, cid, t, surface=None, sentence=None):
        """
        Adds triple to the graph if novel.
        Returns True if accepted, False if redundant.
        """
        if self.is_redundant(h, cid, t):
            return False

        self.kg.add_edge(h, cid, t, surface_relation=surface, sentence=sentence)
        return True
