import mlflow
import pandas as pd

# Before using, make sure you ran the experiment.py, because it extracts mlflow data and converts it into .csv

mlflow.set_tracking_uri("http://localhost:5000")

df = mlflow.search_runs(experiment_names=["exp_homogen"])

# strip the prefixes for readability
df.columns = [c.replace("params.", "").replace("metrics.", "").replace("tags.", "") for c in df.columns]
# drop MLflow housekeeping columns
df = df[[c for c in df.columns if not c.startswith(("run_id", "artifact_uri", "lifecycle", "status", "start_time", "end_time", "experiment_id", "mlflow"))]]
df.to_csv("results_clean.csv", index=False)