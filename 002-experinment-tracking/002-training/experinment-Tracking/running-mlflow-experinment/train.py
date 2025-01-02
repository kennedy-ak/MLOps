import os
import pickle
import click
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def set_experiment():
    experiment_name = "nyc-taxi-experiment"
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Set the tracking URI
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")

    existing_experiment = mlflow.get_experiment_by_name(experiment_name)
    print(f"Existing Experiment: {existing_experiment}")

    if existing_experiment is None:
        mlflow.create_experiment(experiment_name)
        print(f"Experiment '{experiment_name}' created.")
    else:
        print(f"Experiment '{experiment_name}' already exists.")

    mlflow.set_experiment(experiment_name)

mlflow.autolog()

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    set_experiment()  # Ensure the experiment is set
    with mlflow.start_run():
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        print(f"RMSE: {rmse}")

if __name__ == '__main__':
    run_train()
