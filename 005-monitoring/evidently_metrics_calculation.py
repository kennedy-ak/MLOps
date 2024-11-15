# import datetime
# import time
# import random
# import logging 
# import uuid
# import pytz
# import pandas as pd
# import io
# import psycopg
# import joblib

# from prefect import task, flow

# from evidently.report import Report
# from evidently import ColumnMapping
# from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

# SEND_TIMEOUT = 10
# rand = random.Random()

# create_table_statement = """
# drop table if exists dummy_metrics;
# create table dummy_metrics(
# 	timestamp timestamp,
# 	prediction_drift float,
# 	num_drifted_columns integer,
# 	share_missing_values float
# )
# """

# reference_data = pd.read_parquet('data/reference.parquet')
# with open('models/lin_reg.bin', 'rb') as f_in:
# 	model = joblib.load(f_in)

# raw_data = pd.read_parquet('data/green_tripdata_2022-02.parquet')

# begin = datetime.datetime(2022, 2, 1, 0, 0)
# num_features = ['passenger_count', 'trip_distance', 'fare_amount', 'total_amount']
# cat_features = ['PULocationID', 'DOLocationID']
# column_mapping = ColumnMapping(
#     prediction='prediction',
#     numerical_features=num_features,
#     categorical_features=cat_features,
#     target=None
# )

# report = Report(metrics = [
#     ColumnDriftMetric(column_name='prediction'),
#     DatasetDriftMetric(),
#     DatasetMissingValuesMetric()
# ])

# @task
# def prep_db():
# 	with psycopg.connect("host=localhost port=5433 user=postgres password=example", autocommit=True) as conn:
# 		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
# 		if len(res.fetchall()) == 0:
# 			conn.execute("create database test;")
# 		with psycopg.connect("host=localhost port=5433 dbname=test user=postgres password=example") as conn:
# 			conn.execute(create_table_statement)

# @task
# def calculate_metrics_postgresql(curr, i):
# 	current_data = raw_data[(raw_data.lpep_pickup_datetime >= (begin + datetime.timedelta(i))) &
# 		(raw_data.lpep_pickup_datetime < (begin + datetime.timedelta(i + 1)))]

# 	#current_data.fillna(0, inplace=True)
# 	current_data['prediction'] = model.predict(current_data[num_features + cat_features].fillna(0))

# 	report.run(reference_data = reference_data, current_data = current_data,
# 		column_mapping=column_mapping)

# 	result = report.as_dict()

# 	prediction_drift = result['metrics'][0]['result']['drift_score']
# 	num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
# 	share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']

# 	curr.execute(
# 		"insert into dummy_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values) values (%s, %s, %s, %s)",
# 		(begin + datetime.timedelta(i), prediction_drift, num_drifted_columns, share_missing_values)
# 	)

# @flow
# def batch_monitoring_backfill():
# 	prep_db()
# 	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
# 	with psycopg.connect("host=localhost port=5433 dbname=test user=postgres password=example", autocommit=True) as conn:
# 		for i in range(0, 27):
# 			with conn.cursor() as curr:
# 				calculate_metrics_postgresql(curr, i)

# 			new_send = datetime.datetime.now()
# 			seconds_elapsed = (new_send - last_send).total_seconds()
# 			if seconds_elapsed < SEND_TIMEOUT:
# 				time.sleep(SEND_TIMEOUT - seconds_elapsed)
# 			while last_send < new_send:
# 				last_send = last_send + datetime.timedelta(seconds=10)
# 			logging.info("data sent")

# if __name__ == '__main__':
# 	batch_monitoring_backfill()
import datetime
import time
import pandas as pd
import psycopg
import joblib
import logging
import os

from prefect import task, flow
from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s]: %(message)s")

# Environment variables for database credentials
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5433')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'example')
DB_NAME = os.getenv('DB_NAME', 'test')

SEND_TIMEOUT = 10

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
    timestamp timestamp,
    prediction_drift float,
    num_drifted_columns integer,
    share_missing_values float
)
"""

try:
    reference_data = pd.read_parquet('data/reference.parquet')
    with open('models/lin_reg.bin', 'rb') as f_in:
        model = joblib.load(f_in)
    raw_data = pd.read_parquet('data/green_tripdata_2022-02.parquet')
except Exception as e:
    logging.error("Error loading data or model: ", exc_info=True)
    raise

begin = datetime.datetime(2022, 2, 1, 0, 0)
num_features = ['passenger_count', 'trip_distance', 'fare_amount', 'total_amount']
cat_features = ['PULocationID', 'DOLocationID']
column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None
)

report = Report(metrics=[
    ColumnDriftMetric(column_name='prediction'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric()
])

@task
def prep_db():
    try:
        with psycopg.connect(f"host={DB_HOST} port={DB_PORT} user={DB_USER} password={DB_PASSWORD} dbname={DB_NAME}", autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_statement)
        logging.info("Database prepared successfully.")
    except Exception as e:
        logging.error("Failed to prepare database: ", exc_info=True)
        raise

@task
def calculate_metrics_postgresql(i):
    try:
        # Select and predict in a timeframe
        current_data = raw_data[(raw_data.lpep_pickup_datetime >= (begin + datetime.timedelta(i))) &
                                (raw_data.lpep_pickup_datetime < (begin + datetime.timedelta(i + 1)))]
        if current_data.empty:
            logging.warning(f"No data for interval {i}. Skipping.")
            return

        current_data['prediction'] = model.predict(current_data[num_features + cat_features].fillna(0))
        
        if 'prediction' not in current_data.columns or current_data['prediction'].isna().any():
            logging.error("Prediction column is missing or incomplete.")
            return  # Skip this iteration or handle the error appropriately

        report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
        result = report.as_dict()

        # Insert results into the database
        with psycopg.connect(f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}", autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO dummy_metrics (timestamp, prediction_drift, num_drifted_columns, share_missing_values) VALUES (%s, %s, %s, %s)",
                    (begin + datetime.timedelta(i), result['metrics'][0]['result']['drift_score'], result['metrics'][1]['result']['number_of_drifted_columns'], result['metrics'][2]['result']['current']['share_of_missing_values'])
                )
        logging.info(f"Metrics calculated and stored for interval {i}.")
    except Exception as e:
        logging.error(f"Error during metric calculation for interval {i}: {e}", exc_info=True)
        raise

@flow
def batch_monitoring_backfill():
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    for i in range(0, 27):
        calculate_metrics_postgresql(i)
        new_send = datetime.datetime.now()
        seconds_elapsed = (new_send - last_send).total_seconds()
        if seconds_elapsed < SEND_TIMEOUT:
            time.sleep(SEND_TIMEOUT - seconds_elapsed)
        while last_send < new_send:
            last_send += datetime.timedelta(seconds=10)
        logging.info("Data sent for interval {}".format(i))

if __name__ == '__main__':
    try:
        batch_monitoring_backfill()
    except Exception as e:
        logging.error("Failed to run batch monitoring backfill: ", exc_info=True)
