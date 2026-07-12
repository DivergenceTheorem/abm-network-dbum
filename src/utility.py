import networkx as nx


def utility_dbum(i: int,
                 G: nx.Graph,
                 radius: int,
                 delta: float,
                 c: float) -> float:
    """Distance-based utility for agent i: sum of delta^d over reachable nodes within radius, minus link cost."""
    lengths = nx.single_source_shortest_path_length(G, i, cutoff=radius)
    benefit = sum(delta ** d for j, d in lengths.items() if j != i)
    return benefit - c * G.degree[i]