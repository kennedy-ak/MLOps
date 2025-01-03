import sys
import pickle
import pandas as pd

def read_data(filename, categorical):
    """Reads and processes the data from the input parquet file."""
    try:
        print(f"Attempting to read data from: {filename}")
        df = pd.read_parquet(filename, storage_options={'User-Agent': 'Mozilla/5.0'})  # Use User-Agent to mimic a browser
    except Exception as e:
        print(f"Failed to read parquet file from {filename}. Error: {e}")
        raise e

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

def main(year, month):
    """Main function to process data and generate predictions."""
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']
    df = read_data(input_file, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('Predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(output_file, engine='pyarrow', index=False)

if __name__ == "__main__":
    try:
        year = int(sys.argv[1])
        month = int(sys.argv[2])
        main(year, month)
    except Exception as e:
        print(f"Error occurred: {e}")
