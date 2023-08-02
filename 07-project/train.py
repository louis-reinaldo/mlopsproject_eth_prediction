import mlflow
from prefect import flow, task, get_run_logger
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from utils import split_xy, read_data

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
COL_NAMES = ['SMA', 'EMA', 'MACD', 'RSI', 'UpperBB', 'LowerBB', 'Close']
EXPERIMENT_NAME = "random-forest-train"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


@task
def run_train(train, val):
    """Run a basic RF Regressor"""

    X_train, y_train, X_val, y_val = split_xy(train, val)

    mlflow.sklearn.autolog()

    with mlflow.start_run():
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        return y_pred


@flow(retries=3, retry_delay_seconds=5)
def train_flow():
    """flow to read data and train baseline model"""

    logger = get_run_logger()
    logger.info(f'Mlflow experiment: {EXPERIMENT_NAME} set.')
    logger.info('reading data')
    train_df, val_df, test_df = read_data(COL_NAMES)
    logger.info('training base model')
    run_train(train_df, val_df)
    logger.info('base model training completed!')


if __name__ == '__main__':
    train_flow()
