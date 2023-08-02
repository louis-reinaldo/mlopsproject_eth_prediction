import utils
from prefect import flow, task, get_run_logger
import os

DEPENDENT_VARIABLE_NAME = 'Close'
SYMBOL = 'ETH-USD'
VALIDATION_START_DATE = '2023-07-01'
TRAINING_START_DATE = '2022-07-01'
# TEST_START_DATE = '2023-07-25'
TEST_START_DATE = os.getenv('TEST_START_DATE', '2023-07-25')
COL_NAMES = ['SMA', 'EMA', 'MACD', 'RSI', 'UpperBB', 'LowerBB', 'Close']



@task
def download_data_from_yahoo(SYMBOL, TRAINING_START_DATE):
    # logger.info('load data from yfinance')
    asset_df = utils.load_data(SYMBOL, TRAINING_START_DATE)
    return asset_df

@task
def create_ta_features(asset_df, DEPENDENT_VARIABLE_NAME):
    # logger.info('creating technical indicators as features')
    asset_df = utils.create_features(asset_df, DEPENDENT_VARIABLE_NAME)
    return asset_df

@task
def split_feature_data(asset_df, VALIDATION_START_DATE, TEST_START_DATE, DEPENDENT_VARIABLE_NAME):
    # logger.info('split to training, validation, test')
    train_df, val_df, test_df = utils.split_data(asset_df, VALIDATION_START_DATE, TEST_START_DATE, DEPENDENT_VARIABLE_NAME)
    return train_df, val_df, test_df

@task
def save_feature_data(train_df, val_df, test_df):
    # logger.info('saving data')
    utils.save_data(train_df, val_df, test_df)

@flow(retries=3, retry_delay_seconds=5)
def data_preparation():
    logger = get_run_logger()
    logger.info('load data from yfinance')
    asset_df = download_data_from_yahoo(SYMBOL, TRAINING_START_DATE)
    logger.info('creating technical indicators as features')
    asset_df = create_ta_features(asset_df, DEPENDENT_VARIABLE_NAME)
    logger.info('split to training, validation, test')
    train_df, val_df, test_df = split_feature_data(asset_df, VALIDATION_START_DATE, TEST_START_DATE, DEPENDENT_VARIABLE_NAME)
    logger.info('saving data')
    save_feature_data(train_df, val_df, test_df)


if __name__ == '__main__':
    data_preparation()