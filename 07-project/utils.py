from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

# Create the directory if it doesn't exist


def load_data(symbol, start_date):
    """Load data from Yahoo Finance"""
    end_date = datetime.now() + timedelta(days=14)
    df = yf.download(symbol, start=start_date, end=end_date.strftime('%Y-%m-%d'))

    # Fill future dates with NaN
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), end=end_date)
    future_df = pd.DataFrame(index=future_dates)
    df = pd.concat([df, future_df], axis=0)

    return df


def split_data(df, validation_start_date, test_start_date, y_col_name):
    """Split data into train, validation and test sets."""
    train = df[df.index < validation_start_date]
    validation = df[(df.index >= validation_start_date) & (df.index < test_start_date)]
    test = df[df.index >= test_start_date]
    # test.drop(y_col_name, axis =1, inplace = True)
    return train, validation, test


def save_data(train, validation, test):
    """Save datasets to parquet files"""
    Path('data').mkdir(exist_ok=True)
    train.to_parquet('data/train.parquet')
    validation.to_parquet('data/validation.parquet')
    test.to_parquet('data/test.parquet')


def read_data(column_names):
    """Read train, validation and test data from parquet files."""
    train = pd.read_parquet('data/train.parquet')
    validation = pd.read_parquet('data/validation.parquet')
    test = pd.read_parquet('data/test.parquet')

    return train[column_names], validation[column_names], test[column_names]


def fill_future_dates(df, column):
    """Fill future dates with a rolling average of the last known 30 days"""
    last_known_index = df[column].last_valid_index()
    for i in range(1, df.loc[last_known_index:].shape[0]):
        last_30_days_avg = df[column].iloc[-(30 + i) : -i].mean()
        df[column].iloc[-i] = last_30_days_avg
    return df


def add_EMA(df, column, span=20):
    """Add Exponential Moving Average (EMA)"""
    df['EMA'] = df[column].ewm(span=span).mean()
    return df


def add_MACD(df, column):
    """Add Moving Average Convergence Divergence (MACD)"""
    exp12 = df[column].ewm(span=12, adjust=False).mean()
    exp26 = df[column].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD'] = macd - signal
    return df


def add_SMA(df, column, window=20):
    """Add Simple Moving Average (SMA)"""
    df['SMA'] = df[column].rolling(window=window).mean()
    df = fill_future_dates(df, 'SMA')
    return df


def add_RSI(df, column, window=14):
    """Add Relative Strength Index (RSI)"""
    change = df[column].diff()
    gain = change.mask(change < 0, 0)
    loss = change.mask(change > 0, 0)
    average_gain = gain.rolling(window=14).mean()
    average_loss = loss.rolling(window=14).mean()
    rs = average_gain / average_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df = fill_future_dates(df, 'RSI')
    return df


def add_BollingerBands(df, column, window=20):
    """Add Bollinger Bands (upperBB, lowerBB)"""
    df['UpperBB'] = (
        df[column].rolling(window=window).mean()
        + df[column].rolling(window=window).std() * 2
    )
    df['LowerBB'] = (
        df[column].rolling(window=window).mean()
        - df[column].rolling(window=window).std() * 2
    )
    for band in ['UpperBB', 'LowerBB']:
        df = fill_future_dates(df, band)
    return df


# Main function to preprocess the data
def create_features(df, column):
    """Combine all Technical Indicators features"""
    df = add_SMA(df, column)
    df = add_EMA(df, column)
    df = add_MACD(df, column)
    df = add_RSI(df, column)
    df = add_BollingerBands(df, column)
    return df[['SMA', 'EMA', 'MACD', 'RSI', 'UpperBB', 'LowerBB', 'Close']]


def split_xy(train, val):
    """Split training and validation data to X and y"""
    train = train.dropna()
    X_train = train.drop('Close', axis=1)
    y_train = train['Close']

    X_val = val.drop('Close', axis=1)
    y_val = val['Close']

    # X_test = test.drop('Close', axis=1)

    return X_train, y_train, X_val, y_val
