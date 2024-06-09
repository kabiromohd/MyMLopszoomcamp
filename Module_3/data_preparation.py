import requests
from io import BytesIO

import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

@transformer
def read_dataframe(**kwargs) -> pd.DataFrame:
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)

    """
    response = requests.get('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')

    if response.status_code != 200:
           raise Exception(response.text)

    data = pd.read_parquet(BytesIO(response.content))
    
    data['duration'] = data.tpep_dropoff_datetime - data.tpep_pickup_datetime
    data.duration = data.duration.dt.total_seconds() / 60

    data = data[(data.duration >= 1) & (data.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    data[categorical] = data[categorical].astype(str)
    
    return data