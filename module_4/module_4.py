## Load Libraries
import pickle
import pandas as pd
from datetime import date

categorical = ["PULocationID", "DOLocationID"]

def load_file(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def get_mean_ride(ride_url):

    df = read_data(ride_url)

    dicts = df[categorical].to_dict(orient='records')

    artifact_model = load_file("model.bin")
    
    dv = artifact_model[0]
    model = artifact_model[1]

    X_val = dv.transform(dicts)

    y_pred = model.predict(X_val)
    
    print(f"Mean ride prediction: {round(y_pred.mean(), 2)}")


def run():
    # Ride URL
    ride_url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-05.parquet'

    get_mean_ride(ride_url)

if __name__ == '__main__':
    run()