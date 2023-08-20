import os

import requests

import utils

SYMBOL = 'ETH-USD'
DEPENDENT_VARIABLE_NAME = 'Close'
TEST_START_DATE = os.getenv('TEST_START_DATE', '2023-07-25')
VALIDATION_START_DATE = '2023-07-01'
TRAINING_START_DATE = '2022-07-01'

asset_df = utils.load_data(SYMBOL, TEST_START_DATE).dropna()
print('load data from yfinance')
asset_df = utils.load_data(SYMBOL, TRAINING_START_DATE)
print('creating technical indicators as features')
asset_df = utils.create_features(asset_df, DEPENDENT_VARIABLE_NAME)
print('split to training, validation, test')
train_df, val_df, test_df = utils.split_data(
    asset_df, VALIDATION_START_DATE, TEST_START_DATE
)

test_df = test_df.fillna(0)
test_df.index = test_df.index.strftime('%Y-%m-%d')

# print(test_df)

# print('creating technical indicators as features')
# test_df = utils.create_features(future_actual_df, 'Close')
# print(test_df)
# X_test = test_df.drop(columns=['Close'])
# test_df.index = test_df.index.strftime('%Y-%m-%d')
test = test_df.to_dict()
# print(future_actual_data)
port = "9696"
# host = "lrmlopszoomcampproject.ap-southeast-1.elasticbeanstalk.com"
# host = "hadiatmajaya.pythonanywhere.com"
# url = f"{host}/predict"
# url = f"https://{host}"
url = 'http://localhost:9696/predict'
response = requests.post(url, json=test)
print(response.json())


# if __name__ == '__main__':
#     future_actual_df, X_test = prepare_features_for_future_df(test_df)
#     predict.predict(future_actual_df, X_test)
