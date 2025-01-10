import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

options = {
    'client_kwargs': {
        'endpoint_url': os.getenv('S3_ENDPOINT_URL'),
        "aws_secret_access_key" :os.getenv('AWS_SECRET_KEY'), 
            "aws_access_key_id" : os.getenv('AWS_ACCESS_KEY_ID'),
            "region_name" : "us-east-1"
    }
}



output_file = 's3://nyc-duration/out/2023-01.parquet'

df_output = pd.read_parquet(output_file, storage_options=options)

prediction_sum = df_output.prediction.sum()
print(prediction_sum)

