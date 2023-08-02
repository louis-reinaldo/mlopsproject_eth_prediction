import os
import pickle
from pathlib import Path

import mlflow
import pandas as pd
from flask import Flask, jsonify, request
from prefect import flow, task, get_run_logger
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error

from utils import load_data

SYMBOL = 'ETH-USD'
# TEST_START_DATE = '2023-07-25'
TEST_START_DATE = os.getenv('TEST_START_DATE', '2023-07-25')

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


# client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@task
def load_model():
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        with open('model/model_uri.txt', 'r') as file:
            model_uri = file.read()

        print(f"Load model for prediction with model_uri as: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri=model_uri)

    except:
        print('Unable to connect to Mlflow Tracking Server')
        with open(f'model/rf-best-model-eth-prediction.pkl', 'rb') as f:
            model = pickle.load(f)
    return model


# @task
# def load_featurised_test_data():
#     print('Load Raw data for prediction')
#     test_df = pd.read_parquet('data/test.parquet')
#     return test_df

# @task
# def prepare_features_for_future_df(test_df):
#     X_test = test_df.drop(columns=['Close'])
#     X_test.index = X_test.index.strftime('%Y-%m-%d')
#     return X_test


@flow
def predict():
    test_df = pd.read_parquet('data/test.parquet')
    X_test = test_df.drop(columns=['Close'])
    X_test.index = X_test.index.strftime('%Y-%m-%d')

    model = load_model()
    print('predict with model')
    y_pred = list(model.predict(X_test))
    # print('load actual data')
    # y_test = list(future_actual_df['Close'])
    # test_rmse = mean_squared_error(y_test, y_pred, squared=False)
    # print(f"test_rmse from {TEST_START_DATE}: {test_rmse}")
    # future_actual_df['Predicted_Close'] = y_pred
    # print('Actual vs Predicted Close Price')
    # print(future_actual_df[['Close', 'Predicted_Close']])
    return y_pred


app = Flask('eth-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    future_actual_df = pd.DataFrame(request.get_json())
    prediction = predict()

    result = {
        'Actual_Dates': list(future_actual_df.index),
        'Actual_Close_Price': list(future_actual_df['Close']),
        'Predicted_Close_Price': list(prediction),
    }

    print('Saving results')
    pd.DataFrame(result).to_parquet('data/results.parquet')
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
