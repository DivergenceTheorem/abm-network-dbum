import mlflow
import pandas as pd

## I used this code for a different .db setup, it will not work with the current one, because it's in gitignore.

mlflow.set_tracking_uri("http://localhost:5000")

df = mlflow.search_runs(experiment_names=["exp_heterogen"])

# strip the prefixes for readability
df.columns = [c.replace("params.", "").replace("metrics.", "").replace("tags.", "") for c in df.columns]
# drop MLflow housekeeping columns
df = df[[c for c in df.columns if not c.startswith(("run_id", "artifact_uri", "lifecycle", "status", "start_time", "end_time", "experiment_id", "mlflow"))]]
df.to_csv("results_clean.csv", index=False)