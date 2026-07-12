import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")

RUN_NAME = "abm_network_dbum_exp_main" # Replace with the name of the run from which you want to extract data

df = mlflow.search_runs(experiment_names=[RUN_NAME])

# strip the prefixes for readability
df.columns = [c.replace("params.", "").replace("metrics.", "").replace("tags.", "") for c in df.columns]
# drop MLflow housekeeping columns
df = df[[c for c in df.columns if not c.startswith(("run_id", "artifact_uri", "lifecycle", "status", "start_time", "end_time", "experiment_id", "mflow"))]]
df.to_csv("results_clean.csv", index=False)