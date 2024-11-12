#!/usr/bin/env python
# coding: utf-8


import pickle
import pandas as pd
import sys




with open('models/model.bin','rb') as f_in:
    dv, model = pickle.load(f_in)



categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df




year = int(sys.argv[1])
month = int(sys.argv[2])
 
df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')

def run():
    print('System Running')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)



    print(round(y_pred.std(),2))





    df['ride_id'] = f'{year}/{month}_' + df.index.astype('str')



    df_results = pd.DataFrame(y_pred, columns=['predicted_duration'],)

    df_results['predicted_duration'] = df_results['predicted_duration'].round(2)

    # print(df_results.head())



    df_results['ride_id'] = df['ride_id']


    output_file = 'results.parquet'
    df_results.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


import os

file_size = os.path.getsize('results.parquet')


file_size_kb = file_size / 1024  # Kilobytes
file_size_mb = file_size / (1024 * 1024)  # Megabytes

# print(f"File size: {file_size_kb:.2f} KB")
# print(f"File size: {file_size_mb:.2f} MB")


# data = pd.read_parquet('results.parquet')




if __name__=='__main__':
    run()













