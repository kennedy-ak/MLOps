from mage_ai.io.file import FileIO
from pandas import DataFrame
if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

import mlflow
import pickle
from typing import Tuple
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer

@data_exporter
def export_model(dv_lr_tuple: Tuple[DictVectorizer, LinearRegression], **kwargs) -> None:
    """
    Exports the trained model and its DictVectorizer to MLflow.
    """
    dv, lr = dv_lr_tuple
    
    # Set the MLflow tracking URI
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    
    # Set an experiment name (optional)
    mlflow.set_experiment('my_experiment')
    
    with mlflow.start_run():
        # Log the model. This will create and log the MLmodel artifact.
        mlflow.sklearn.log_model(lr, artifact_path='model', registered_model_name='linear_regression_model')
        
        # Save the DictVectorizer to a file and log it as an artifact
        artifact_path = 'dict_vectorizer.pkl'
        with open(artifact_path, 'wb') as f:
            pickle.dump(dv, f)
        mlflow.log_artifact(artifact_path, artifact_path='artifacts')

        # Optionally, log additional parameters or metrics
        mlflow.log_param("model_type", "linear_regression")





from mage_ai.io.file import FileIO # modules for file operations,

from pandas import DataFrame# importing the pandas dataframe from pandas

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

import mlflow
import pickle
from typing import Tuple
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer


@data_exporter
def export_model(dv_lr_tuple: Tuple[DictVectorizer, LinearRegression], **kwargs) -> None:
    """
    Exports the trained machine learning model and its feature encoder (DictVectorizer) to MLflow,
    a platform for the machine learning lifecycle, including experimentation, reproducibility, and deployment.
    """
    # Unpack the DictVectorizer and Linear Regression model from the passed tuple
    dv, lr = dv_lr_tuple
    
    # Set the MLflow tracking server to the given URI for logging and tracking models.
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    
    # Define or set the name of the experiment under which this run will be logged.
    mlflow.set_experiment('my_experiment')
    
    # Start an MLflow run to log data from the model training session.
    with mlflow.start_run():
        # Log the Linear Regression model in a specified directory and also register the model with a given name.
        mlflow.sklearn.log_model(lr, artifact_path='model', registered_model_name='linear_regression_model')
        
        # Save the DictVectorizer to a file and log it as an artifact for reproducibility.
        artifact_path = 'dict_vectorizer.pkl'
        with open(artifact_path, 'wb') as f:
            pickle.dump(dv, f)
        mlflow.log_artifact(artifact_path, artifact_path='artifacts')

        # Log additional parameters or configurations of the model, useful for model tracking.
        mlflow.log_param("model_type", "linear_regression")
