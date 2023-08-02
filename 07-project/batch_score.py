import os
import pickle
from pathlib import Path
from datetime import date

import mlflow
import pandas as pd
from prefect import flow, task, get_run_logger
from evidently import ColumnMapping
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error
from evidently.report import Report
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    ColumnQuantileMetric,
    DatasetMissingValuesMetric
)
from prefect.artifacts import create_markdown_artifact
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics.base_metric import generate_column_metrics

from hpo import hpo_flow
from utils import load_data
from register_model import register_model_flow

SYMBOL = 'ETH-USD'
TEST_START_DATE = os.getenv('TEST_START_DATE', '2023-07-25')
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


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


@task
def prepare_data_monitoring_report(numeric_features, target=None):
    column_mapping = ColumnMapping(
        prediction='prediction', numerical_features=numeric_features, target=target
    )

    report = Report(
        metrics=[
            ColumnDriftMetric(column_name='prediction'),
            DataDriftPreset(),
            DatasetMissingValuesMetric(),
            ColumnQuantileMetric('SMA', quantile=0.5),
        ]
    )

    return column_mapping, report


@task
def prepare_current_future_data():
    future_actual_df = load_data(SYMBOL, TEST_START_DATE).dropna()
    future_actual_df.index = future_actual_df.index.strftime('%Y-%m-%d')

    reference_data = pd.read_parquet('data/validation.parquet')

    current_data = pd.read_parquet('data/test.parquet')
    current_data.index = current_data.index.strftime('%Y-%m-%d')
    current_data = current_data[current_data.index.isin(future_actual_df.index)]
    X_test = current_data.drop(columns=['Close']).dropna()
    X_val = reference_data.drop(columns=['Close']).dropna()
    current_data['Close'] = future_actual_df['Close']
    return current_data, reference_data, X_test, X_val


@flow
def score():
    logger = get_run_logger()
    logger.info(f'Loading future {SYMBOL} data from Yahoo Finance')

    logger.info(f'Loading Model')
    model = load_model()

    logger.info(f'Preparing Reference and Current data for prediction')
    current_data, reference_data, X_test, X_val = prepare_current_future_data()

    test_y_pred = list(model.predict(X_test))
    val_y_pred = list(model.predict(X_val))

    current_data['prediction'] = test_y_pred
    reference_data['prediction'] = val_y_pred

    logger.info(f'test_y_pred: {test_y_pred}')
    logger.info(f'val_y_pred: {val_y_pred}')

    logger.info('Getting RMSE')

    val_rmse = mean_squared_error(
        reference_data['Close'], reference_data['prediction'], squared=False
    )
    logger.info(f'Validation RMSE: {val_rmse} ')

    test_rmse = mean_squared_error(
        current_data['Close'], current_data['prediction'], squared=False
    )
    logger.info(f'Test RMSE: {test_rmse} ')

    logger.info(f'Preparing Evidently Report')
    column_mapping, evidently_report = prepare_data_monitoring_report(
        list(X_test.columns), target='Close'
    )

    logger.info(f'Calculating Data Drift')
    evidently_report.run(
        reference_data=reference_data.reset_index(),
        current_data=current_data.reset_index(),
        column_mapping=column_mapping,
    )

    evidently_report.save_html("report.html")
    result = evidently_report.as_dict()

    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    # share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']
    # sma_quantile_median = result['metrics'][3]['result']['current']['value']

    logger.info(f'Data Monitoring results:')

    markdown__model_report = f"""# Model Report

    ## Summary

    Date: {date.today()}
    Prediction from: {TEST_START_DATE}

    ## RMSE Random Forest Model

    | Metric    | Value |
    |:----------|-------:|
    | Validation RMSE | {val_rmse:.2f} |
    | Test RMSE | {test_rmse:.2f} |
    | Prediction Drift | {prediction_drift} |
    | Num Drifted Columns | {num_drifted_columns} |
    """
    create_markdown_artifact(
        key="eth-prediction-model-report", markdown=markdown__model_report
    )

    logger.info(f'Saving featurised future data with prediction')
    current_data.to_parquet('data/future_data_with_prediction.parquet')

    ##logic to trigger retraining
    prediction_drift_thres = 0.003
    print(f'prediction_drift: {prediction_drift}')
    if prediction_drift > prediction_drift_thres:
        logger.info(
            f'Prediction Drift is {prediction_drift_thres},  more than {prediction_drift_thres}. Model retraining triggered'
        )
        hpo_flow()
        logger.info('Registering new model')
        register_model_flow()

    return result


if __name__ == '__main__':
    score()
