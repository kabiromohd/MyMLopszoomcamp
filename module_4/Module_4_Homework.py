#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd

with open('lin_reg.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-04.parquet')

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


print("Mean Prediction: ", y_pred.mean())



df['ride_id'] = f'{2023:04d}/{4:02d}_' + df.index.astype('str')


df['Prediction'] = y_pred

df_result = df[["ride_id", "Prediction"]]

df_result.to_parquet(
    "output_file_04",
    engine='pyarrow',
    compression=None,
    index=False
)
