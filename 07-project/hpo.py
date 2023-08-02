from utils import split_xy, read_data
from prefect import flow, task, get_run_logger

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
COL_NAMES = ['SMA', 'EMA', 'MACD', 'RSI', 'UpperBB', 'LowerBB', 'Close']
HPO_EXPERIMENT_NAME = "mlops_zoomcamp_eth_prediction_hpo"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(HPO_EXPERIMENT_NAME)
mlflow.sklearn.autolog()

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 50, 1),
        'max_depth': trial.suggest_int('max_depth', 1, 20, 1),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, 1),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4, 1),
        'random_state': 42,
        'n_jobs': -1
    }
    with mlflow.start_run():
        mlflow.log_params(params)
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(rf, "model")

    return rmse

@task(log_prints = True)
def run_optimization(train, val, test, num_trials):    

    """Use Optuna to run through hyperparameter search space"""
    X_train, y_train, X_val, y_val = split_xy(train, val)

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=num_trials)

@flow(retries=3, retry_delay_seconds=5)
def hpo_flow():

    """Main flow for Hyperparameter optimization"""

    logger = get_run_logger()
    logger.info(f'Mlflow experiment: {HPO_EXPERIMENT_NAME} set.')
    logger.info('reading data')
    train_df, val_df, test_df = read_data(COL_NAMES)
    logger.info('Searching Hyperparameter Space for best model')
    run_optimization(train_df, val_df, test_df, 20)
    logger.info('optimised model training completed!')


if __name__ == '__main__':
    hpo_flow()
