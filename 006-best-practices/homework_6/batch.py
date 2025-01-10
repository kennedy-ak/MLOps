#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
# print(os.getenv('OUTPUT_FILE_PATTERN'))


def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)






categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    options = {
        'client_kwargs': {
            'endpoint_url': os.getenv('S3_ENDPOINT_URL'),
            'aws_secret_access_key': os.getenv('AWS_SECRET_KEY'),
            'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
            'region_name': 'us-east-1'
        }
    }


    # year = int(sys.argv[1])
    # month = int(sys.argv[2])

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df = pd.read_parquet(filename, storage_options=options)
   
    
    return prepare_data(df,categorical)

def prepare_data(df, categorical):
 
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def main(year,month):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
    print(output_file)

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
        


    df = read_data(input_file)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)


    print('predicted mean duration:', y_pred.mean())


    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    print("here")
    print(df_result.head())

    options = {
        'client_kwargs': {
            'endpoint_url': os.getenv('S3_ENDPOINT_URL'),
            "aws_secret_access_key" :os.getenv('AWS_SECRET_KEY'), 
            "aws_access_key_id" : os.getenv('AWS_ACCESS_KEY_ID'),
            "region_name" : "us-east-1"

        }
                       
        }
    df_result.to_parquet(output_file, storage_options=options,compression=None, engine='pyarrow', index=False)

    

if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)
