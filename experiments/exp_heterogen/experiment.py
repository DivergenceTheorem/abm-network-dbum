import sys
import os
import json
import collections
import tempfile
import random

import numpy as np
import networkx as nx
import mlflow

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utility import utility_dbum
from dynamics import one_step, delta_u_add, delta_u_remove
from metrics import compute_metrics, benchmark_metrics, config_distance, er_average_metrics, METRIC_KEYS

# --- Convergence parameters ---
WINDOW_SIZE    = 10
BURN_IN_CHECKS = 10
CHECK_EVERY    = 10
K              = 5
EPSILON        = 0.0005
MAX_STEPS_MULT = 20

# --- Fixed parameters ---
N        = 10
ADD_PROB = 0.5
N_SEEDS  = 100
ER_ITERS = 100

COST_VALUES = [round(0.1 * k, 1) for k in range(1, N + 1)]  # [0.1, 0.2, ..., 1.0]

# --- Parameter grid ---
param_grid_delta  = [0.35, 0.65, 0.95]
param_grid_radius = [3, 5, 7, 10]

param_grid = [
    (delta, radius)
    for delta  in param_grid_delta
    for radius in param_grid_radius
]

BENCHMARKS = benchmark_metrics(N)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("exp_heterogen")


# Helper functions

def make_utility_fn(delta: float, radius: int, costs: list):
    """Return a utility function with agent-specific costs."""
    def utility_fn(i, G):
        return utility_dbum(i, G, radius=radius, delta=delta, c=costs[i])
    return utility_fn


def run_single_seed(delta: float, radius: int, costs: list, seed: int) -> tuple:
    """Run ABM with agent-specific costs until convergence or step cap."""
    G = nx.Graph()
    G.add_nodes_from(range(N))
    random.seed(seed)

    utility_fn = make_utility_fn(delta, radius, costs)
    max_steps  = MAX_STEPS_MULT * N * (N - 1) // 2

    metric_history    = collections.deque(maxlen=WINDOW_SIZE)
    prev_window_mean  = None
    stable_count      = 0
    steps_since_check = 0
    n_checks_done     = 0

    for step in range(1, max_steps + 1):
        one_step(G, N, utility_fn, add_prob=ADD_PROB)
        steps_since_check += 1

        if steps_since_check >= CHECK_EVERY:
            steps_since_check = 0
            n_checks_done += 1
            m = compute_metrics(G, N)
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


def is_pairwise_stable(G: nx.Graph, utility_fn) -> bool:
    """Return True iff G satisfies pairwise stability conditions."""
    # Condition I: no agent wants to unilaterally sever an existing link
    for i, j in G.edges():
        if delta_u_remove(i, j, G, utility_fn) > 0:
            return False
        if delta_u_remove(j, i, G, utility_fn) > 0:
            return False
    # Condition II: no pair mutually benefits from adding a missing link
    for i in range(N):
        for j in range(i + 1, N):
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
        experiment_names=["exp_heterogen"],
        filter_string="status = 'FINISHED'",
        max_results=10000,
    )
    if not existing.empty and "tags.mlflow.runName" in existing.columns:
        completed_runs = set(existing["tags.mlflow.runName"].dropna().tolist())
    print(f"Resume: {len(completed_runs)} runs already finished, skipping them.")
except Exception as e:
    print(f"Warning: could not fetch existing runs ({e}). Starting from scratch.")


# Main experiment loop

total = len(param_grid)
for run_idx, (delta, radius) in enumerate(param_grid):
    run_name = f"d={delta}_r={radius}"
    if run_name in completed_runs:
        continue
    print(f"[{run_idx + 1}/{total}] δ={delta}  r={radius}")

    seed_metrics      = []
    seed_g_dists      = []
    seed_er_dists     = []
    seed_ps           = []
    seed_steps        = []
    cost_assignments  = {}   # seed -> [c_0, c_1, ..., c_9]

    for seed in range(N_SEEDS):
        # Independent RNG for cost assignment (numpy), separate from dynamics (random)
        costs = np.random.default_rng(seed).permutation(COST_VALUES).tolist()
        cost_assignments[seed] = costs

        utility_fn       = make_utility_fn(delta, radius, costs)
        G_run, steps     = run_single_seed(delta, radius, costs, seed)

        metrics  = compute_metrics(G_run, N)
        ps       = is_pairwise_stable(G_run, utility_fn)
        er_m     = er_average_metrics(metrics["density"], N, ER_ITERS)
        g_dists  = {name: config_distance(metrics, bench) for name, bench in BENCHMARKS.items()}
        er_dists = {name: config_distance(er_m, bench)    for name, bench in BENCHMARKS.items()}

        seed_metrics.append(metrics)
        seed_g_dists.append(g_dists)
        seed_er_dists.append(er_dists)
        seed_ps.append(ps)
        seed_steps.append(steps)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "n":        N,
            "delta":    delta,
            "radius":   radius,
            "add_prob": ADD_PROB,
            "n_seeds":  N_SEEDS,
            "er_iters": ER_ITERS,
        })

        log = {}
        for k in METRIC_KEYS:
            vals = np.array([m[k] for m in seed_metrics])
            log[f"mean_{k}"] = float(np.mean(vals))
            log[f"std_{k}"]  = float(np.std(vals))
            log[f"ci95_{k}"] = float(1.96 * np.std(vals) / np.sqrt(N_SEEDS))

        for name in BENCHMARKS:
            g_vals  = np.array([d[name] for d in seed_g_dists])
            er_vals = np.array([d[name] for d in seed_er_dists])
            log[f"g_dist_mean_{name}"]  = float(np.mean(g_vals))
            log[f"g_dist_std_{name}"]   = float(np.std(g_vals))
            log[f"g_dist_ci95_{name}"]  = float(1.96 * np.std(g_vals) / np.sqrt(N_SEEDS))
            log[f"er_dist_mean_{name}"] = float(np.mean(er_vals))

        log["pairwise_stable_frac"] = float(np.mean(seed_ps))
        log["mean_steps"]           = float(np.mean(seed_steps))

        mlflow.log_metrics(log)

        # Save cost assignments as a JSON artifact
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix=f"costs_{run_name}_", delete=False
        )
        json.dump({str(s): v for s, v in cost_assignments.items()}, tmp, indent=2)
        tmp.close()
        mlflow.log_artifact(tmp.name, artifact_path="cost_assignments")
        os.remove(tmp.name)

        print(f"  -> PS frac={log['pairwise_stable_frac']:.2f}  "
              f"mean_steps={log['mean_steps']:.0f}")
