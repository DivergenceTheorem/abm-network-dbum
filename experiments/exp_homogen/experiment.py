import sys
import os
import collections
import functools
import random

import numpy as np
import networkx as nx
import mlflow

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utility import utility_dbum
from dynamics import one_step, delta_u_add, delta_u_remove
from metrics import compute_metrics, benchmark_metrics, config_distance, er_average_metrics, METRIC_KEYS

# --- Convergence parameters ---
WINDOW_SIZE     = 10
BURN_IN_CHECKS  = 10
CHECK_EVERY     = 10
K               = 5
EPSILON         = 0.0005
MAX_STEPS_MULT  = 20

# --- Experiment parameters ---
N_SEEDS  = 100
ER_ITERS = 100

param_grid_n       = list(range(10, 110, 10))          # [10, 20, ..., 100]
param_grid_delta   = [0.95, 0.65, 0.35]
param_grid_cost    = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
param_grid_radius  = [3, 5, 8, 10]
param_grid_add_prob = [0.25, 0.5, 0.75]

param_grid = [
    (n, delta, c, radius, add_prob)
    for n        in param_grid_n
    for delta    in param_grid_delta
    for c        in param_grid_cost
    for radius   in param_grid_radius
    for add_prob in param_grid_add_prob
]

# Precompute benchmark metrics once per N
ALL_BENCHMARKS = {n: benchmark_metrics(n) for n in param_grid_n}

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("exp_homogen")

# --- Helper functions ---

def run_single_seed(n: int, delta: float, c: float, radius: int,
                    add_prob: float, seed: int) -> tuple:
    """Run ABM until convergence or step cap. Returns (graph, steps_taken)."""

    G = nx.Graph()
    G.add_nodes_from(range(n))
    random.seed(seed)

    utility_fn = functools.partial(utility_dbum, radius=radius, delta=delta, c=c)
    max_steps  = MAX_STEPS_MULT * n * (n - 1) // 2

    metric_history   = collections.deque(maxlen=WINDOW_SIZE)
    prev_window_mean = None
    stable_count     = 0
    steps_since_check = 0
    n_checks_done    = 0

    for step in range(1, max_steps + 1):
        one_step(G, n, utility_fn, add_prob=add_prob)
        steps_since_check += 1

        if steps_since_check >= CHECK_EVERY:
            steps_since_check = 0
            n_checks_done += 1
            m = compute_metrics(G, n)
            metric_history.append([m[k] for k in METRIC_KEYS])

            if n_checks_done > BURN_IN_CHECKS and len(metric_history) == WINDOW_SIZE:
                window_mean = np.mean(list(metric_history), axis=0)
                if prev_window_mean is not None:
                    diff = float(np.max(np.abs(window_mean - prev_window_mean)))
                    stable_count = stable_count + 1 if diff < EPSILON else 0
                    if stable_count >= K:
                        return G, step
                prev_window_mean = window_mean

    return G, max_steps


def is_pairwise_stable(G: nx.Graph, n: int, utility_fn) -> bool:
    """Return True iff G satisfies pairwise stability conditions."""
    # Condition I: no agent wants to unilaterally sever an existing link
    for i, j in G.edges():
        if delta_u_remove(i, j, G, utility_fn) > 0:
            return False
        if delta_u_remove(j, i, G, utility_fn) > 0:
            return False
    # Condition II: no pair mutually benefits from adding a missing link
    for i in range(n):
        for j in range(i + 1, n):
            if not G.has_edge(i, j):
                if (delta_u_add(i, j, G, utility_fn) > 0 and
                        delta_u_add(j, i, G, utility_fn) > 0):
                    return False
    return True


# Resume: collect already-finished run names so they can be skipped. 
# It helps to continue experiment from where it stopped last time (if it did)

completed_runs: set = set()
try:
    existing = mlflow.search_runs(
        experiment_names=["exp_homogen"],
        filter_string="status = 'FINISHED'",
        max_results=50000,
    )
    if not existing.empty and "tags.mlflow.runName" in existing.columns:
        completed_runs = set(existing["tags.mlflow.runName"].dropna().tolist())
    print(f"Resume: {len(completed_runs)} runs already finished, skipping them.")
except Exception as e:
    print(f"Warning: could not fetch existing runs ({e}). Starting from scratch.")


# Main experiment loop

total = len(param_grid)
for run_idx, (n, delta, c, radius, add_prob) in enumerate(param_grid):
    run_name = f"n={n}_d={delta}_c={c}_r={radius}_p={add_prob}"
    if run_name in completed_runs:
        continue
    print(f"[{run_idx + 1}/{total}] n={n} δ={delta} c={c} r={radius} p={add_prob}")

    benchmarks = ALL_BENCHMARKS[n]

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "n":        n,
            "delta":    delta,
            "c":        c,
            "radius":   radius,
            "add_prob": add_prob,
            "n_seeds":  N_SEEDS,
            "er_iters": ER_ITERS,
        })

        # Analytically empty: no simulation needed
        if delta <= c:
            empty_G = nx.Graph()
            empty_G.add_nodes_from(range(n))
            empty_metrics = compute_metrics(empty_G, n)
            er_metrics    = er_average_metrics(0.0, n, ER_ITERS)

            log = {}
            for k in METRIC_KEYS:
                log[f"mean_{k}"]  = empty_metrics[k]
                log[f"std_{k}"]   = 0.0
                log[f"ci95_{k}"]  = 0.0
            for name, bench in benchmarks.items():
                log[f"g_dist_{name}"]  = config_distance(empty_metrics, bench)
                log[f"er_dist_{name}"] = config_distance(er_metrics, bench)
            log["pairwise_stable_frac"] = 1.0
            log["mean_steps"]           = 0.0
            log["analytic_empty"]       = 1.0
            mlflow.log_metrics(log)
            print("  -> analytic empty")
            continue

        # Multi-seed runs
        seed_metrics  = []   # list of metric dicts
        seed_g_dists  = []   # list of {bench_name: dist} dicts
        seed_er_dists = []
        seed_ps       = []   # pairwise stable bools
        seed_steps    = []

        utility_fn = functools.partial(utility_dbum, radius=radius, delta=delta, c=c)

        for seed in range(N_SEEDS):
            G_run, steps = run_single_seed(n, delta, c, radius, add_prob, seed)

            metrics = compute_metrics(G_run, n)
            ps      = is_pairwise_stable(G_run, n, utility_fn)
            er_m    = er_average_metrics(metrics["density"], n, ER_ITERS)

            g_dists  = {name: config_distance(metrics, bench) for name, bench in benchmarks.items()}
            er_dists = {name: config_distance(er_m, bench)    for name, bench in benchmarks.items()}

            seed_metrics.append(metrics)
            seed_g_dists.append(g_dists)
            seed_er_dists.append(er_dists)
            seed_ps.append(ps)
            seed_steps.append(steps)

        # Aggregate metrics over seed runs
        log = {}
        for k in METRIC_KEYS:
            vals = np.array([m[k] for m in seed_metrics])
            log[f"mean_{k}"] = float(np.mean(vals))
            log[f"std_{k}"]  = float(np.std(vals))
            log[f"ci95_{k}"] = float(1.96 * np.std(vals) / np.sqrt(N_SEEDS))

        for name in benchmarks:
            g_vals  = np.array([d[name] for d in seed_g_dists])
            er_vals = np.array([d[name] for d in seed_er_dists])
            log[f"g_dist_mean_{name}"]  = float(np.mean(g_vals))
            log[f"g_dist_std_{name}"]   = float(np.std(g_vals))
            log[f"g_dist_ci95_{name}"]  = float(1.96 * np.std(g_vals) / np.sqrt(N_SEEDS))
            log[f"er_dist_mean_{name}"] = float(np.mean(er_vals))

        log["pairwise_stable_frac"] = float(np.mean(seed_ps))
        log["mean_steps"]           = float(np.mean(seed_steps))
        log["analytic_empty"]       = 0.0

        mlflow.log_metrics(log)
        print(f"  -> PS frac={log['pairwise_stable_frac']:.2f}  mean_steps={log['mean_steps']:.0f}")