# --- INITIAL SETUP ---

import sys
import os
import collections
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import random
import functools

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mlflow
import pickle

from utility import utility_dbum
from dynamics import one_step
from metrics import compute_metrics, benchmark_metrics, config_distance, er_average_metrics, METRIC_KEYS

# --- Experiment description ---
'''
In this experiment we expand population of agents - we will iterate from 10 to 30 and see how this affects the results.
Stopping is convergence-based: we compute a rolling window mean of the 5D feature vector every CHECK_EVERY steps
and stop when the window mean changes by less than EPSILON across all dimensions for K consecutive checks.
'''

# --- Simulation parameters ---
VISUALIZE = False  # set to True to show the animation window
PAUSE_FRAMES = 10  # small pause when switching settings

# --- Convergence parameters ---
WINDOW_SIZE = 10     # fixed number of metric observations in the rolling window (same for all N)
BURN_IN_CHECKS = 10  # fixed number of checks to skip before comparing (same for all N)
CHECK_EVERY = 10    # compute metrics every CHECK_EVERY steps -- after how many steps we measure the 5D metric
K = 5                # consecutive stable window comparisons to declare convergence
EPSILON = 0.0005      # max per-dimension change in window mean to count as stable
MAX_STEPS_MULT = 20   # hard cap: MAX_STEPS_MULT * N*(N-1)//2 steps per setting

# --- Model parameters ---
ADD_PROB = 0.5  # probability of trying to add a link
ER_ITERS = 30   # ER random graph realizations to average over per setting

param_grid_n = [10, 20, 30]
param_grid_delta = [0.95, 0.65, 0.35]
param_grid_cost = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
param_grid_radius = [3, 5, 8, 10]

# Test parameter grid
# param_grid_n = [20]
# param_grid_delta = [0.95]
# param_grid_cost = [1, 0.9, 0.8]
# param_grid_radius = [3]

param_grid = []
for n in param_grid_n:
    for r in param_grid_radius:
        for d in param_grid_delta:
            for c in param_grid_cost:
                param_grid.append((n, d, c, r))

# --- Precomputed benchmark metrics per N (avoids recomputing inside the loop) ---
ALL_BENCHMARKS = {n: benchmark_metrics(n) for n in param_grid_n}

# --- Helper/Derived variables ---
random.seed(1)
G = nx.Graph()
G.add_nodes_from(range(param_grid[0][0]))
current_param_idx = 0
step_in_setting = 0
in_pause = False
pause_left = 0
recorded_this_setting = False
current_mlflow_run = None

# Convergence state — reset per setting in reset_graph_for_setting()
metric_history = collections.deque()
prev_window_mean = None
stable_count = 0
steps_since_check = 0
n_checks_done = 0

pos = None  # layout for network drawing

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("abm_network_dbum_exp_main")

# --- Animation setup (only used if VISUALIZE=True) ---
fig, ax = plt.subplots(figsize=(9, 5))


def visualizer_network(n, delta, c, radius, step_in_setting, action_text="", pause=False):
    ax.clear()
    ax.set_axis_off()
    utilities = [utility_dbum(i, G, radius, delta, c) for i in range(n)]
    prefix = "(PAUSE) " if pause else ""
    ax.set_title(f"{prefix}n={n} δ={delta:.2f}, c={c:.2f}, R={radius} | step={step_in_setting}\n{action_text}")
    nx.draw(G, pos=pos, with_labels=True, node_size=600, node_color=utilities,
            cmap=plt.cm.viridis, edge_color="gray", font_size=8, ax=ax)
    return ax,


def reset_graph_for_setting(n: int, seed_offset: int = 0):
    global G, recorded_this_setting, pos, metric_history, prev_window_mean, stable_count, steps_since_check, n_checks_done

    G = nx.Graph()
    G.add_nodes_from(range(n))
    random.seed(1 + seed_offset)
    recorded_this_setting = False
    # reset convergence state for this setting
    metric_history = collections.deque(maxlen=WINDOW_SIZE)
    prev_window_mean = None
    stable_count = 0
    steps_since_check = 0
    n_checks_done = 0
    pos = nx.spring_layout(G, seed=42 + seed_offset)


def _log_and_close_run(G: nx.Graph, steps: int, n: int):
    benchmarks = ALL_BENCHMARKS[n]
    g_metrics = compute_metrics(G, n)
    er_metrics = er_average_metrics(g_metrics["density"], n, ER_ITERS)
    g_dists = {name: config_distance(g_metrics, bench) for name, bench in benchmarks.items()}
    er_dists = {name: config_distance(er_metrics, bench) for name, bench in benchmarks.items()}

    mlflow.log_metrics({
        **{f"g_{k}": v for k, v in g_metrics.items()},
        **{f"er_{k}": v for k, v in er_metrics.items()},
        **{f"g_dist_{name}": d for name, d in g_dists.items()},
        **{f"er_dist_{name}": d for name, d in er_dists.items()},
        "steps": steps,
        "edges": G.number_of_edges(),
    })

    tmp_path = os.path.join(tempfile.gettempdir(), "final_network.pkl")
    with open(tmp_path, "wb") as f:
        pickle.dump(G, f)
    mlflow.log_artifact(tmp_path, artifact_path="network")
    os.remove(tmp_path)

    mlflow.end_run()


def update(frame):
    global current_param_idx, step_in_setting, in_pause, pause_left, recorded_this_setting, current_mlflow_run, stable_count, steps_since_check, n_checks_done, prev_window_mean

    n, delta, c, radius = param_grid[current_param_idx]

    max_steps = MAX_STEPS_MULT * n * (n - 1) // 2

    # start of a new setting
    if step_in_setting == 0 and not recorded_this_setting:
        reset_graph_for_setting(n, seed_offset=current_param_idx)
        current_mlflow_run = mlflow.start_run(run_name=f"n={n}_d={delta}_c={c}_r={radius}")
        mlflow.log_params({
            "n": n,
            "delta": delta,
            "c": c,
            "radius": radius
        })

    # analytic empty: skip simulation entirely
    if delta <= c and not recorded_this_setting:
        recorded_this_setting = True
        _log_and_close_run(G, steps=0, n=n)
        in_pause = True
        pause_left = PAUSE_FRAMES
        if VISUALIZE:
            return visualizer_network(n, delta, c, radius, 0, action_text="analytic empty", pause=True)
        return []

    if in_pause:
        pause_left -= 1
        if pause_left <= 0:
            in_pause = False
            step_in_setting = 0
            current_param_idx += 1
            if current_param_idx >= len(param_grid):
                print("\nAll settings complete.")
                if VISUALIZE:
                    ani.event_source.stop()
                    plt.close(fig)
                return []
            n_next = param_grid[current_param_idx][0]
            reset_graph_for_setting(n_next, seed_offset=current_param_idx)
        if VISUALIZE:
            return visualizer_network(n, delta, c, radius, step_in_setting, action_text="pause", pause=True)
        return []

    # second reset guard for the very first frame of a setting
    if step_in_setting == 0:
        reset_graph_for_setting(n, seed_offset=current_param_idx)

    # ABM step
    utility_fn = functools.partial(utility_dbum, radius=radius, delta=delta, c=c)
    one_step(G, n, utility_fn, add_prob=ADD_PROB)
    step_in_setting += 1
    steps_since_check += 1

    # convergence check every CHECK_EVERY steps
    if steps_since_check >= CHECK_EVERY and not recorded_this_setting:
        steps_since_check = 0
        n_checks_done += 1
        m = compute_metrics(G, n)
        metric_history.append([m[k] for k in METRIC_KEYS])

        converged = False
        if n_checks_done > BURN_IN_CHECKS and len(metric_history) == WINDOW_SIZE:
            '''check if we went through burn-in steps or we have reached the desired window size'''
            window_mean = np.mean(list(metric_history), axis=0)
            if prev_window_mean is not None:
                diff = float(np.max(np.abs(window_mean - prev_window_mean)))
                stable_count = stable_count + 1 if diff < EPSILON else 0
                if stable_count >= K:
                    converged = True
            prev_window_mean = window_mean

        hit_cap = step_in_setting >= max_steps

        if converged or hit_cap:
            recorded_this_setting = True
            reason = "converged" if converged else "cap"
            print(f"  [{reason}] n={n} d={delta} c={c} r={radius} at step {step_in_setting}")
            _log_and_close_run(G, steps=step_in_setting, n=n)
            in_pause = True
            pause_left = PAUSE_FRAMES

    if VISUALIZE:
        return visualizer_network(n, delta, c, radius, step_in_setting)
    return []


if VISUALIZE:
    ani = FuncAnimation(fig, update, frames=10_000_000, interval=50, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()
else:
    frame = 0
    while current_param_idx < len(param_grid):
        update(frame)
        frame += 1