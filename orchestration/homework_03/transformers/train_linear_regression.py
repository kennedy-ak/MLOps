if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from typing import Tuple
from pandas import DataFrame
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression


@transformer
def train_model(df: DataFrame, **kwargs) -> Tuple[DictVectorizer, LinearRegression]:
    # Feature columns (categorical)
    categorical = ['PULocationID', 'DOLocationID']
    
    # Extract target (y)
    y = df['duration'].values
    
    # Convert the DataFrame’s categorical columns to a dictionary list for vectorization
    train_dicts = df[categorical].to_dict(orient='records')
    
    # Initialize the DictVectorizer
    dv = DictVectorizer()
    X = dv.fit_transform(train_dicts)
    
    # Initialize and train the LinearRegression model
    lr = LinearRegression()
    lr.fit(X, y)
    
    # Print the intercept to the Mage logs
    print(f"Model intercept: {lr.intercept_}")
    
    # Return both the dict vectorizer and the trained model for downstream usage
    return dv, lr


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
