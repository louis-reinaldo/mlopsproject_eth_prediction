import pytest
import pandas as pd
import mlflow 
from mlflow.tracking import MlflowClient
import pickle

from unittest.mock import patch
from utils import read_data, split_xy
from load_save_data import data_preparation, save_feature_data, create_ta_features, split_feature_data, download_data_from_yahoo
from sklearn.ensemble import RandomForestRegressor


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
COL_NAMES = ['SMA', 'EMA', 'MACD', 'RSI', 'UpperBB', 'LowerBB', 'Close']
HPO_EXPERIMENT_NAME = "mlops_zoomcamp_eth_prediction_hpo"

@patch("utils.load_data")
@patch("utils.create_features")
@patch("utils.split_data")
@patch("utils.save_data")
def test_data_preparation(mock_save_data, mock_split_data, mock_create_features, mock_load_data):
    # Define the columns that should exist in the saved dataframes
    expected_columns = ['SMA', 'EMA', 'MACD', 'RSI', 'UpperBB', 'LowerBB', 'Close']

    # Mock the behavior of your utility functions
    mock_load_data.return_value = pd.DataFrame(columns=expected_columns)
    mock_create_features.return_value = pd.DataFrame(columns=expected_columns)
    mock_split_data.return_value = (pd.DataFrame(columns=expected_columns),
                                    pd.DataFrame(columns=expected_columns),
                                    pd.DataFrame(columns=expected_columns))

    # Run the data_preparation function
    data_preparation()

    # Assert that save_data was called with the correct parameters
    call_args = mock_save_data.call_args[0]
    
    for df in call_args:
        assert sorted(df.columns) == sorted(expected_columns)

def create_test_data():
    df = pd.DataFrame({
    'Date': ['2023-07-01', '2023-07-02', '2023-07-03', '2023-07-04', '2023-07-05'],
    'SMA': [1810.615143, 1820.360626, 1831.178204, 1845.483917, 1857.737329],
    'EMA': [1847.829035, 1856.363256, 1865.794294, 1872.540890, 1876.164425],
    'MACD': [13.199699, 13.608082, 14.157440, 12.391575, 8.773673],
    'RSI': [177.482886, 167.460333, 167.191118, 214.211200, 1075.936062],
    'UpperBB': [1990.762900, 2006.004111, 2022.010460, 2021.624189, 2014.157248],
    'LowerBB': [1630.467385, 1634.717141, 1640.345949, 1669.343645, 1701.317411],
    'Close': [1924.565918, 1937.438354, 1955.389160, 1936.633545, 1910.588013]
    })
    df.set_index('Date', inplace=True)
    return df

def load_model():
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        with open('model/model_uri.txt', 'r') as file:
            model_uri = file.read()

        print(f"Load model for prediction with model_uri as: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri = model_uri)

    except:
        print('Unable to connect to Mlflow Tracking Server')
        with open(f'model/rf-best-model-eth-prediction.pkl', 'rb') as f:
            model = pickle.load(f)
    return model

def test_load_model():
    model = load_model()
    assert model is not None, "Failed to load the model"
    assert isinstance(model, RandomForestRegressor), "The loaded model is not of the correct type"

    df = create_test_data()
    # X_train, y_train = split_xy(df)
    y_train = df['Close']
    X_train = df.drop(['Close'], axis = 1)

    predictions = model.predict(X_train)
    
    # Add more tests based on the model's expected behavior.
    # For example, you could test that the predictions have the expected shape:
    assert predictions.shape == y_train.shape, "The model's predictions are not the expected shape"

    # You could also add tests to check the performance of the model:
    rmse = (((predictions - y_train) ** 2).mean()) ** 0.5
    assert rmse < 40, "The model's performance is worse than expected"

if __name__ == "__main__": 
    pytest.main()