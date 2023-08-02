import pickle

import mlflow
from prefect import flow, task, get_run_logger
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from prefect.artifacts import create_markdown_artifact
from sklearn.feature_extraction import DictVectorizer

from utils import split_xy, read_data

EXPERIMENT_NAME = "mlops_zoomcamp_eth_prediction"
HPO_EXPERIMENT_NAME = "mlops_zoomcamp_eth_prediction_hpo"
RF_PARAMS = [
    'max_depth',
    'n_estimators',
    'min_samples_split',
    'min_samples_leaf',
    'random_state',
    'n_jobs',
]
BEST_MODEL_NAME = "rf-best-model-eth-prediction"
STAGE = 'PRODUCTION'
COL_NAMES = ['SMA', 'EMA', 'MACD', 'RSI', 'UpperBB', 'LowerBB', 'Close']
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


@task
def train_and_log_model(X_train, y_train, X_val, y_val, params):
    with mlflow.start_run():
        for param in RF_PARAMS:
            params[param] = int(params[param])

        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        # evaluate model on the validation and test sets
        valid_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
        mlflow.log_metric("valid_rmse", valid_rmse)


@task
def register_model(train, val, top_n):
    X_train, y_train, X_val, y_val = split_xy(train, val)

    data_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    }

    # Iterate over the dictionary and save each item
    for name, data in data_dict.items():
        with open(f'data/{name}.pkl', 'wb') as f:
            pickle.dump(data, f)

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    # runs = client.search_runs(
    #     experiment_ids=experiment.experiment_id,
    #     run_view_type=ViewType.ACTIVE_ONLY,
    #     max_results=top_n,
    #     order_by=["metrics.rmse ASC"]
    # )

    ###come back to this, is it training again?####
    # for run in runs:
    #     train_and_log_model(X_train, y_train, X_val, y_val, params=run.data.params)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.test_rmse ASC"],
    )[0]

    # Register the best model
    run_id = best_run.info.run_id
    # artifact_uri = run_id.artifact_uri
    # print(f'artifact uri: {artifact_uri}')
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, name=BEST_MODEL_NAME)
    model_version = 1
    client.transition_model_version_stage(
        name=BEST_MODEL_NAME,
        version=model_version,
        stage=STAGE,
        archive_existing_versions=False,
    )
    return model_uri


@flow(retries=3, retry_delay_seconds=5)
def register_model_flow():
    """Save to model registry and save to pickle as backup"""

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger = get_run_logger()

    logger.info(f'Mlflow experiment: {EXPERIMENT_NAME} set.')
    logger.info('reading data')
    train_df, val_df, test_df = read_data(COL_NAMES)

    logger.info('register model')
    model_uri = register_model(train_df, val_df, 20)

    logger.info('saving model uri')
    with open('model/model_uri.txt', 'w') as file:
        file.write(model_uri)

    logger.info(f'model registered with model uri: {model_uri}')
    logger.info("loading model from mlflow")
    model = mlflow.sklearn.load_model(model_uri=model_uri)
    params = model.get_params()
    mlflow.log_params(params)

    # logger.info(f"parameters for best model: {params}")

    logger.info(f"saving parameters for best model into artefacts")

    params_markdown = "# Best Model Parameters\n\n"
    for key, value in params.items():
        params_markdown += f"| {key} | {value} |\n"
    params_markdown += "\n"

    create_markdown_artifact(key="best-model-parameters", markdown=params_markdown)

    logger.info(f"saving model as {BEST_MODEL_NAME}.pkl")

    run_id = model_uri.split("/")[1]
    run = client.get_run(run_id)
    metrics = run.data.metrics
    rmse = metrics.get('rmse')

    logger.info(f"Best Model RMSE: {rmse}")

    logger.info('Saving model to pickle file')
    with open(f'model/{BEST_MODEL_NAME}.pkl', 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    register_model_flow()
