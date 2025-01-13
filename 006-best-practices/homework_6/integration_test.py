import pandas as pd
from datetime import datetime
import boto3
import os

from dotenv import load_dotenv
import json

load_dotenv()

s3_client = boto3.client('s3',
                         endpoint_url=os.getenv('S3_ENDPOINT_URL'), 
)

s3_client.put_bucket_policy(
    Bucket='nyc-duration',
    Policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": "*",
                "Action": ["s3:GetObject", "s3:PutObject"],
                "Resource": "arn:aws:s3:::nyc-duration/*"
            }
        ]
    })
)


print(os.getenv('S3_ENDPOINT_URL'))

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df = pd.DataFrame(data, columns=columns)

options = {
    'client_kwargs': {
        'endpoint_url': os.getenv('S3_ENDPOINT_URL'),
        "aws_secret_access_key" :os.getenv('AWS_SECRET_KEY'), 
            "aws_access_key_id" : os.getenv('AWS_ACCESS_KEY_ID'),
            "region_name" : "us-east-1"
    }
}

input_file = 's3://nyc-duration/in/2023-01.parquet'
# df = pd.read_parquet(input_file, storage_options=options)

df.to_parquet(
    input_file, engine="pyarrow", compression=None, index=False, storage_options=options
)

response = s3_client.list_objects(Bucket='nyc-duration', Prefix='in/2023-01.parquet') 

file_size = response['Contents'][0]['Size']
print(file_size)



os.system('python batch.py 2023 1')


output_file = 's3://nyc-duration/out/2023-01.parquet'

df_output = pd.read_parquet(output_file, storage_options=options)

prediction_sum = df_output["predicted_duration"].sum()
print(prediction_sum)