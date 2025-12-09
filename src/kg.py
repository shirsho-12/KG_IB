import networkx as nx
import json


class NXKnowledgeGraph:
    """
    NetworkX-based directed KG.
    Edges store attributes including cluster_id and possibly the surface relation form.
    """

    def __init__(self, file_path=None):
        self.G = nx.DiGraph()
        if file_path:
            self.load(file_path)

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

    def _make_graphml_safe(self, G):
        """
        Returns a deep-copied graph where all attributes
        are GraphML-safe: strings, ints, floats, bools.
        """

        H = nx.DiGraph()

        for n, attrs in G.nodes(data=True):
            safe_attrs = {}
            for k, v in attrs.items():
                if isinstance(v, (set, list, dict)):
                    safe_attrs[k] = json.dumps(list(v))
                else:
                    safe_attrs[k] = v
            H.add_node(n, **safe_attrs)

        for u, v, attrs in G.edges(data=True):
            safe_attrs = {}
            for k, val in attrs.items():
                if isinstance(val, (set, list, dict)):
                    safe_attrs[k] = json.dumps(list(val))
                else:
                    safe_attrs[k] = val
            H.add_edge(u, v, **safe_attrs)

        return H

    def save(self, path):
        nx.write_graphml(self._make_graphml_safe(self.G), path)

    def load(self, path):
        self.G = nx.read_graphml(path)
        # Convert JSON strings back to original types
        for n, attrs in self.G.nodes(data=True):
            norm = {}
            for k, v in attrs.items():
                try:
                    parsed = json.loads(v)
                    # Automatically convert arrays -> sorted tuple or set
                    if isinstance(parsed, list) and len(parsed) > 0:
                        # sorted tuple for deterministic comparison
                        norm[k] = set(sorted(parsed))
                    else:
                        norm[k] = parsed
                except json.JSONDecodeError:
                    norm[k] = v
            self.G.nodes[n].update(norm)

        for u, v, attrs in self.G.edges(data=True):
            norm = {}
            for k, v in attrs.items():
                try:
                    parsed = json.loads(v)
                    # Automatically convert arrays -> sorted tuple or set
                    if isinstance(parsed, list) and len(parsed) > 0:
                        # sorted tuple for deterministic comparison
                        norm[k] = set(sorted(parsed))
                    else:
                        norm[k] = parsed
                except json.JSONDecodeError:
                    norm[k] = v
            self.G[u][v].update(norm)
