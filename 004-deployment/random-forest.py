#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle

import pandas as pd
import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[3]:


from sklearn.pipeline import make_pipeline


# In[ ]:


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("green-taxi-duration")


# In[5]:


def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    return df


def prepare_dictionaries(df: pd.DataFrame):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts


# In[6]:


df_train = read_dataframe('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-01.parquet')
df_val = read_dataframe('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-02.parquet')

target = 'duration'
y_train = df_train[target].values
y_val = df_val[target].values

dict_train = prepare_dictionaries(df_train)
dict_val = prepare_dictionaries(df_val)


# In[7]:


with mlflow.start_run():
    params = dict(max_depth=20, n_estimators=100, min_samples_leaf=10, random_state=0)
    mlflow.log_params(params)

    pipeline = make_pipeline(
        DictVectorizer(),
        RandomForestRegressor(**params, n_jobs=-1)
    )

    pipeline.fit(dict_train, y_train)
    y_pred = pipeline.predict(dict_val)

    rmse = mean_squared_error(y_pred, y_val, squared=False)
    print(params, rmse)
    mlflow.log_metric('rmse', rmse)

    mlflow.sklearn.log_model(pipeline, artifact_path="model")


# In[9]:


from mlflow.tracking import MlflowClient


# In[21]:


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
RUN_ID = '5e47b3ff09524d12a7e92f0284352a09'

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
path = client.download_artifacts(run_id=RUN_ID, path='model')
print("Downloaded artifact to:", path)


# In[23]:


path = '/home/akogo/Desktop/MLOps/004-deployment/mlruns/1/5e47b3ff09524d12a7e92f0284352a09/artifacts/model/model.pkl'


# In[24]:


with open(path, 'rb') as f_out:
    dv = pickle.load(f_out)


# In[25]:


dv


# In[ ]:





# In[27]:


from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
experiment_id = client.get_experiment_by_name("green-taxi-duration").experiment_id
runs = client.search_runs(experiment_ids=[experiment_id])
for run in runs:
    print("Run ID:", run.info.run_id)


# In[29]:


client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
experiments = client.search_experiments()
for experiment in experiments:
    print(f"Experiment ID: {experiment.experiment_id}, Name: {experiment.name}")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    for run in runs:
        print(f"Run ID: {run.info.run_id}, Status: {run.info.status}")


# In[30]:


experiment = client.get_experiment_by_name("green-taxi-duration")
print("Artifact Location:", experiment.artifact_location)


# In[31]:


path = client.download_artifacts(run_id=RUN_ID, path="model", dst_path=experiment.artifact_location)
print("Downloaded artifact to:", path)


# In[33]:


client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
experiments = client.search_experiments()
for exp in experiments:
    print(f"Experiment ID: {exp.experiment_id}, Name: {exp.name}")
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    for run in runs:
        print(f"Run ID: {run.info.run_id}, Status: {run.info.status}")


# In[ ]:




