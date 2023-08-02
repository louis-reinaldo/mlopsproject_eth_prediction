import os

import requests

from utils import load_data

SYMBOL = 'ETH-USD'
# TEST_START_DATE = '2023-07-25' #need to figure out how to let user
TEST_START_DATE = os.getenv('TEST_START_DATE', '2023-07-25')

future_actual_df = load_data(SYMBOL, TEST_START_DATE).dropna()
future_actual_df.index = future_actual_df.index.strftime('%Y-%m-%d')

future_actual_data = future_actual_df.to_dict()
# print(future_actual_data)

url = 'http://localhost:9696/predict'
response = requests.post(url, json=future_actual_data)
print(response.json())


# if __name__ == '__main__':
#     future_actual_df, X_test = prepare_features_for_future_df(test_df)
#     predict.predict(future_actual_df, X_test)
