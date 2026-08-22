import networkx as nx


METRIC_KEYS = ["density", "centralization", "avg_clustering", "largest_cc_frac", "global_efficiency"]


def compute_metrics(G: nx.Graph, N: int) -> dict:
    density = nx.density(G) # Density of the network

    degrees = [d for _, d in G.degree()]
    d_max = max(degrees) if degrees else 0
    centralization = (
        sum(d_max - d for d in degrees) / ((N - 1) * (N - 2))
        if N > 2 else 0.0
    ) # Degree centralization

    avg_clustering = nx.average_clustering(G) # Average local clustering coefficient

    components = list(nx.connected_components(G))
    largest_cc_frac = max(len(c) for c in components) / N if components else 0.0 # Fraction of nodes in the largest component

    global_efficiency = nx.global_efficiency(G) # Harmonic mean-path length

    return {
        "density": density,
        "centralization": centralization,
        "avg_clustering": avg_clustering,
        "largest_cc_frac": largest_cc_frac,
        "global_efficiency": global_efficiency,
    }


def benchmark_metrics(N: int) -> dict:
    """Metrics for the 4 benchmark graphs. Regular = cycle graph (2-regular)."""
    graphs = {
        "empty":    nx.empty_graph(N),
        "complete": nx.complete_graph(N),
        "star":     nx.star_graph(N - 1),
        "regular":  nx.cycle_graph(N),
    }
    return {name: compute_metrics(G, N) for name, G in graphs.items()}


def config_distance(m_a: dict, m_b: dict) -> float:
    """Normalized Euclidean distance in R^5: (1/sqrt(5)) * sqrt(sum of squared diffs)."""
    sq_sum = sum((m_a[k] - m_b[k]) ** 2 for k in METRIC_KEYS)
    return (sq_sum / 5) ** 0.5


def er_average_metrics(density: float, N: int, n_iters: int) -> dict:
    """Average metrics over n_iters ER(N, density) graphs with fixed seeds for reproducibility."""
    totals = {k: 0.0 for k in METRIC_KEYS}
    for i in range(n_iters):
        G_er = nx.erdos_renyi_graph(N, density, seed=i)
        for k, v in compute_metrics(G_er, N).items():
            totals[k] += v
    return {k: v / n_iters for k, v in totals.items()}