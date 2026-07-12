import random
import networkx as nx
from typing import Callable


def delta_u_add(i: int,
                j: int,
                G: nx.Graph,
                utility_fn: Callable[[int, nx.Graph], float]) -> float:
    """Marginal utility for agent i of adding edge (i, j)."""
    if G.has_edge(i, j):
        return 0.0
    u_before = utility_fn(i, G)
    G.add_edge(i, j)
    u_after = utility_fn(i, G)
    G.remove_edge(i, j)
    return u_after - u_before


def delta_u_remove(i: int,
                   j: int,
                   G: nx.Graph,
                   utility_fn: Callable[[int, nx.Graph], float]) -> float:
    """Marginal utility for agent i of removing edge (i, j)."""
    if not G.has_edge(i, j):
        return 0.0
    u_before = utility_fn(i, G)
    G.remove_edge(i, j)
    u_after = utility_fn(i, G)
    G.add_edge(i, j)
    return u_after - u_before


def one_step(G: nx.Graph,
             n: int,
             utility_fn: Callable[[int, nx.Graph], float],
             add_prob: float) -> str:
    """One update step: a random agent proposes to add or remove a link under mutual consent (add) or unilateral (remove)."""
    i = random.randrange(n)

    if random.random() < add_prob:
        # Search candidate for adding a link
        candidates = [j for j in range(n) if j != i and not G.has_edge(i, j)]
        if not candidates:
            return f"add skipped (i = {i})"
        j = random.choice(candidates)
        if delta_u_add(i, j, G, utility_fn) > 0 and delta_u_add(j, i, G, utility_fn) > 0:
            G.add_edge(i, j)
            return f"added edge ({i}, {j})"
        return f"add rejected (i = {i}, j = {j})"
    else:
        # Search candidate for removing a link
        neighborhood = list(G.neighbors(i))
        if not neighborhood:
            return f"remove skipped (i = {i})"
        j = random.choice(neighborhood)
        if delta_u_remove(i, j, G, utility_fn) > 0:
            G.remove_edge(i, j)
            return f"removed edge ({i}, {j})"
        return f"remove rejected (i = {i}, j = {j})"
