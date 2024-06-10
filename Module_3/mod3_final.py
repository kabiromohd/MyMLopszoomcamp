import mlflow
import pandas as pd
import pickle

import requests
from io import BytesIO

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader


@transformer
def exp_train_model(**kwargs) -> pd.DataFrame:

    response = requests.get('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')

    if response.status_code != 200:
           raise Exception(response.text)

    data = pd.read_parquet(BytesIO(response.content))
    
    data['duration'] = data.tpep_dropoff_datetime - data.tpep_pickup_datetime
    data.duration = data.duration.dt.total_seconds() / 60

    data = data[(data.duration >= 1) & (data.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    data[categorical] = data[categorical].astype(str)

    train_dicts = data[categorical].to_dict(orient='records')

    with mlflow.start_run():

        dv = DictVectorizer()
        X_train = dv.fit_transform(train_dicts)

        target = 'duration'
        y_train = data[target].values

        lr = LinearRegression()

        lr.fit(X_train, y_train)

        mlflow.sklearn.log_model(lr, "linear_model")

        with open("mlops/dicv_preprocessor.bin", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact(dv, "DictVect)        

    return X_train, y_train, dv, lr