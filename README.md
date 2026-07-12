# Experiment Run Manual

## Installing libraries

Check the list of required libraries in requirements.txt file. Install them via

```
pip install -r requirements.txt
```

## Setting up MLFlow environment

Install the MLFlow environment prior to running the experiment. For starting instructions check https://mlflow.org/docs/latest/ml/tracking/quickstart/

## Running the experiment

The main experiment is in the experiments/exp_main folder.

Run the experiment.py file to run the simulation and save data to MLFlow.

Run the extract_mlflow_data.py file to transfer data from MLFlow to .csv file.

The src folder contains all functions that are used for simulations.