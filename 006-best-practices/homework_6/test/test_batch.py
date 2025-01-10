import batch
import pandas as pd
from datetime import datetime



def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


categorical = ['PULocationID', 'DOLocationID']


data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
]

columns = ['PULocationID',
            'DOLocationID', 
           'tpep_pickup_datetime',
            'tpep_dropoff_datetime'
]
           

df =pd.DataFrame(data,columns=columns)


def test_prepare_data():
    expected_prepared_data = pd.DataFrame(
        [
            (
                "-1",
                "-1",
                dt(1,1),
                dt(1,10),
                (dt(1,10)-dt(1,1)).total_seconds() / 60,

            ),
            (
                "1",
                "1",
                dt(1,2),
                dt(1,10),
                (dt(1,10)-dt(1,2)).total_seconds() / 60,


            ),
            (
                "1",
                "-1",
                dt(1,2),
                dt(1,2,59),
                (dt(1,2,59)-dt(1,2)).total_seconds() / 60,


            ),
            (
                "3",
                "4",
                dt(1,2),
                dt(2,2, 1),
                (dt(2,2,1)-dt(1,2)).total_seconds() / 60,


            )
        ],

    columns= columns + ["duration"],

    )



    expected_prepared_data = expected_prepared_data[
        expected_prepared_data['duration'].between(1,60)
    ]

    actual_prepared_df = batch.prepare_data(df,categorical)

    pd.testing.assert_frame_equal(expected_prepared_data,actual_prepared_df )