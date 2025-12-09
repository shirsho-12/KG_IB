import networkx as nx


class NXKnowledgeGraph:
    """
    NetworkX-based directed KG.
    Edges store attributes including cluster_id and possibly the surface relation form.
    """

    def __init__(self):
        self.G = nx.DiGraph()

    def add_edge(self, h, cid, t, surface_relation=None, sentence=None):
        """
        Adds directed edge h -> t with relation cluster ID.
        Multiple surfaces can map to same cluster; store them in edge data.
        """
        if not self.G.has_edge(h, t):
            self.G.add_edge(h, t, clusters=set(), surfaces=set(), sentences=[])
        # update attributes
        self.G[h][t]["clusters"].add(cid)
        if surface_relation:
            self.G[h][t]["surfaces"].add(surface_relation)
        if sentence:
            self.G[h][t]["sentences"].append(sentence)

    def has_edge_with_cluster(self, h, cid, t):
        """
        True if h->t edge exists AND has cluster cid.
        """
        if self.G.has_edge(h, t):
            return cid in self.G[h][t]["clusters"]
        return False

    def has_inverse_edge_with_cluster(self, h, cid, t):
        """
        True if t->h has an edge of cluster cid.
        """
        if self.G.has_edge(t, h):
            return cid in self.G[t][h]["clusters"]
        return False
