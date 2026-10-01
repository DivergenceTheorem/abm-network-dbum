# Agent-Based Modeling of Decentralized Network Formation: A Distance-Based Utility Approach

Simulation code for **Kurmanov & Melo Ponce (2026)**. The paper is included in this repository as [Kurmanov_Melo_2026.pdf](Kurmanov_Melo_2026.pdf).

The code implements the distance-based utility model (DBUM) of Jackson and Wolinsky, truncated at an information radius *r*. Links form through the myopic link-revision process of Watts: agents start unconnected, a link is formed only if both agents agree, and either agent can cut a link alone. Each run continues until the network stops changing. The final network is then described by five structural metrics, compared with benchmark topologies and with Erdős–Rényi graphs of the same density, and checked directly for pairwise stability.

There are two experiments:

| Experiment | Folder | Agents | Grid | Configurations | Paper |
|---|---|---|---|---|---|
| Main | [experiments/exp_homogen/](experiments/exp_homogen/) | Homogeneous (common link cost *c*) | *n* × δ × *c* × *r* × *p*<sub>add</sub> | 2,880 | Sections 4–5 |
| Supplementary | [experiments/exp_heterogen/](experiments/exp_heterogen/) | Heterogeneous (agent-specific costs) | δ × *r* | 12 | Section 5, heterogeneity trial |

Each configuration is run with 100 seeds.

---

## Repository layout

```
.
├── Kurmanov_Melo_2026.pdf          # the paper
├── requirements.txt                # pinned Python dependencies
├── src/                            # model code shared by both experiments
│   ├── utility.py                  # utility_dbum: truncated distance-based utility
│   ├── dynamics.py                 # one_step: a single link-revision step; marginal utilities
│   └── metrics.py                  # 5-D feature vector, benchmark graphs, distances, ER baseline
└── experiments/
    ├── exp_homogen/
    │   ├── experiment.py           # runs the 2,880-configuration grid and logs to MLflow
    │   ├── extract_mlflow_data.py  # exports MLflow runs to results_clean.csv
    │   └── results_clean.csv       # results used in the paper (2,880 rows)
    └── exp_heterogen/
        ├── experiment.py           # runs the 12-configuration grid and logs to MLflow
        ├── extract_mlflow_data.py
        └── results_clean.csv       # results used in the paper (12 rows)
```

You can use the committed `results_clean.csv` files to check the paper's tables and figures without re-running anything.

---

## 1. Environment setup

The code is pure Python and has been run with **Python 3.13**. A virtual environment is recommended.

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS / Linux:         source .venv/bin/activate

pip install -r requirements.txt
```

Pinned versions (from [requirements.txt](requirements.txt)):

| Package | Version | Used for |
|---|---|---|
| networkx | 3.6.1 | graphs, shortest paths, metrics, ER graphs |
| numpy | 2.4.4 | aggregation, cost permutations (heterogeneous experiment) |
| mlflow | 3.12.0 | experiment tracking and result storage |
| pandas | 2.3.3 | exporting results to CSV |
| scipy, matplotlib, seaborn, openpyxl | see file | analysis and plotting |

Use these exact versions if you want results that match the paper bit for bit (see [Reproducibility notes](#5-reproducibility-notes)).

---

## 2. Start an MLflow tracking server

Both `experiment.py` scripts log to an MLflow server at `http://localhost:5000`, and both refuse to start if no server is reachable there. Start one from the **repository root** in a separate terminal and keep it running for the whole experiment:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 --port 5000
```

On Windows PowerShell, put the command on one line or replace `\` with a backtick.

This creates `mlflow.db` and `mlartifacts/`, both of which are git-ignored. You can watch progress at <http://localhost:5000>. More background is in the [MLflow tracking quickstart](https://mlflow.org/docs/latest/ml/tracking/quickstart/).

---

## 3. Run the experiments

Run each script from **inside its own folder**. The extraction script writes `results_clean.csv` to the current working directory, so running it elsewhere puts the file in the wrong place.

### 3a. Main experiment (homogeneous agents)

```bash
cd experiments/exp_homogen
python experiment.py            # simulate and log to the MLflow experiment "exp_homogen"
python extract_mlflow_data.py   # export to results_clean.csv (overwrites the committed file)
```

Each configuration becomes one MLflow run named `n={n}_d={delta}_c={c}_r={radius}_p={add_prob}`.

### 3b. Supplementary experiment (heterogeneous costs)

```bash
cd experiments/exp_heterogen
python experiment.py            # logs to the MLflow experiment "exp_heterogen"
python extract_mlflow_data.py
```

Runs are named `d={delta}_r={radius}`. Each run also saves the per-seed cost assignment as a JSON artifact under `cost_assignments/`.

### Runtime and resuming

The main grid is computationally heavy: 2,880 configurations × 100 seeds, with populations up to *n* = 80, on a single process. Expect it to take a long time. The cost is driven mostly by large *n* and by configurations that hit the step cap.

Both scripts can **resume**. At startup they ask MLflow for runs that are already `FINISHED` and skip them, so you can interrupt the script and restart it safely. A run that was interrupted partway is not `FINISHED`, so it is simply re-run.

To start from scratch, stop the server and delete `mlflow.db` and `mlartifacts/`.

---

## 4. What the code does

### Model ([src/utility.py](src/utility.py))

The utility of agent *i* in network *G* is

$$u_i(G) = \sum_{j \neq i,\; d(i,j) \le r} \delta^{\,d(i,j)} \;-\; c_i \cdot \deg(i)$$

where *d* is the shortest-path distance. Agents outside radius *r* and agents that cannot be reached contribute nothing.

### Dynamics ([src/dynamics.py](src/dynamics.py))

At each step, one agent *i* is chosen uniformly at random. Then:

- With probability *p*<sub>add</sub>, *i* proposes a link to a random agent *j* it is not linked to. The link is added only if **both** Δ*u*<sub>i</sub> > 0 and Δ*u*<sub>j</sub> > 0.
- Otherwise, *i* considers cutting the link to a random neighbour *j*. The link is removed if Δ*u*<sub>i</sub> > 0.

Every run starts from the empty graph.

### Stopping rule (`run_single_seed` in each `experiment.py`)

Every τ = `CHECK_EVERY` = 10 steps, the 5-D feature vector is added to a rolling window of size W = 10. A run stops when the window mean changes by less than ε = 0.0005 (in the ∞-norm) for K = 5 consecutive checks. The first 10 checks are a burn-in and are not used. Runs that never meet this rule stop at a hard cap of `20 · n(n−1)/2` steps.

### Pairwise stability check

After a run ends, `is_pairwise_stable` tests the final graph directly:

1. No agent gains by cutting one of its links.
2. No unlinked pair would both gain by adding a link between them.

### Feature vector and benchmarks ([src/metrics.py](src/metrics.py))

Each final network is described by five metrics: **density**, **degree centralization** (Freeman), **average clustering**, **largest connected component fraction**, and **global efficiency**.

Its distance to each of four benchmark graphs on *n* nodes (empty, complete, star, and a cycle as the 2-regular benchmark) is

$$D(G, B) = \tfrac{1}{\sqrt{5}}\,\lVert m(G) - m(B) \rVert_2$$

The ER baseline averages the feature vector over 100 seeded G(*n*, ρ(*G*)) graphs, where ρ(*G*) is the density of the simulated network. The baseline's distances to the same benchmarks are computed the same way.

### Parameter grids

| | Main (`exp_homogen`) | Supplementary (`exp_heterogen`) |
|---|---|---|
| *n* | 10, 20, …, 80 | 10 |
| δ | 0.35, 0.65, 0.95 | 0.35, 0.65, 0.95 |
| *c* | 0.1, 0.2, …, 1.0 (shared by all agents) | 0.1, …, 1.0 assigned to the 10 agents by a random permutation for each seed |
| *r* | 3, 5, **8**, 10 | 3, 5, **7**, 10 |
| *p*<sub>add</sub> | 0.25, 0.5, 0.75 | 0.5 |
| Seeds | 100 (0–99) | 100 (0–99) |
| Configurations | 2,880 | 12 |

The third radius value differs between the two experiments (8 vs 7). This matches the paper.

**Analytic shortcut (main experiment only).** When δ ≤ *c*, adding the first link to an empty graph can never benefit either agent, so the empty graph is the outcome. These configurations are not simulated. They are logged with the empty graph's metrics, zero variance, `mean_steps = 0`, `pairwise_stable_frac = 1.0`, and `analytic_empty = 1`. The paper computes PS_frac over **simulated runs only**, so filter on `analytic_empty == 0` when you reproduce PS-fraction results.

---

## 5. Reproducibility notes

- **Seeding.** Seed *s* ∈ {0, …, 99} sets Python's `random` module, which drives the dynamics. The same seeds are reused for every configuration, so comparisons across parameters are paired. ER baseline graphs use `seed = 0, …, 99`. In the heterogeneous experiment, cost permutations come from a separate generator, `numpy.random.default_rng(seed)`, so drawing costs does not change the random stream used by the dynamics.
- **Determinism.** With the pinned dependency versions, re-running a configuration reproduces the committed CSV values exactly. This was checked on several homogeneous configurations, including `mean_density`, `pairwise_stable_frac` and `mean_steps`. Different versions of networkx or Python could change the random streams or the order in which graphs are iterated, so the numbers might not match bit for bit.
- **No parallelism.** Configurations and seeds run one after another, so results do not depend on how work is scheduled.

---

## 6. Output data dictionary (`results_clean.csv`)

There is one row per parameter configuration. Every statistic is computed over the 100 seeds. The column order is whatever MLflow returns, so select columns by name.

| Column(s) | Meaning |
|---|---|
| `n`, `delta`, `c`, `radius`, `add_prob` | Parameter configuration. `c` is not present in the heterogeneous file. |
| `n_seeds`, `er_iters` | Number of seeds (100) and ER graphs per baseline (100). |
| `mean_<metric>`, `std_<metric>`, `ci95_<metric>` | Mean, standard deviation and 95% CI half-width (1.96·σ/√100) of each feature: `density`, `centralization`, `avg_clustering`, `largest_cc_frac`, `global_efficiency`. |
| `g_dist_mean_<bench>`, `g_dist_std_<bench>`, `g_dist_ci95_<bench>` | Distance from the simulated network to each benchmark: `empty`, `complete`, `star`, `regular`. |
| `er_dist_mean_<bench>` | Mean distance from the density-matched ER baseline to each benchmark. |
| `pairwise_stable_frac` | Fraction of seeds whose final network is pairwise stable. |
| `mean_steps` | Mean number of steps until the run stopped, either by the stopping rule or at the cap. |
| `analytic_empty` | Main experiment only. 1 if the configuration was resolved analytically (δ ≤ *c*), otherwise 0. |
| `runName` | MLflow run name. |

---

## Citation

If you use this code, please cite:

> Kurmanov, M., & Melo Ponce, A. (2026). *Agent-Based Modeling of Decentralized Network Formation: A Distance-Based Utility Approach.*

Code: <https://github.com/DivergenceTheorem/abm-network-dbum>
