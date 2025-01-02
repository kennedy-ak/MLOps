import os
import pickle
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']
client = MlflowClient()

# Set tracking URI (ensure it's consistent with hpo.py)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        new_params = {param: int(params[param]) for param in RF_PARAMS}

        rf = RandomForestRegressor(**new_params)
        rf.fit(X_train, y_train)

        val_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("test_rmse", test_rmse)

        mlflow.sklearn.log_model(rf, "model")

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):
    # Ensure the HPO experiment exists
    hpo_experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    if hpo_experiment is None:
        raise ValueError(f"Experiment '{HPO_EXPERIMENT_NAME}' does not exist. Run the HPO script first.")

    # Retrieve top runs from the HPO experiment
    runs = client.search_runs(
        experiment_ids=hpo_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    
    # Train and log the top models
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # Retrieve the best run from the new experiment
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"]
    )[0]

    # Register the best model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_name = "best_random_forest_model"
    mlflow.register_model(model_uri=model_uri, name=model_name)

    # Print the best test RMSE
    best_test_rmse = best_run.data.metrics["test_rmse"]
    print(f"Best test RMSE: {best_test_rmse}")

if __name__ == '__main__':
    run_register_model()
