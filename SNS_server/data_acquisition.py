import os
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from config import dataset_dir


def get_historical_data(ticker):
    """
    Retrieves historical stock data from Yahoo Finance.

    Args:
        None

    Returns:
        pandas.DataFrame:
            A DataFrame containing the historical stock data.

    """
    print("Fetching historical data...")
    # create a Ticker object
    stock = yf.Ticker(ticker)

    # get the historical stock data from 5 years ago until today.
    stock_data = stock.history(period="5y", interval="1d")

    # reset the index of the DataFrame
    stock_data.reset_index(inplace=True)

    # return the DataFrame containing the stock data
    return stock_data


def get_updated_stock_data(ticker, begin_date):
    """
    Retrieves the stock data from the previous trading day.

    Args:
        None

    Returns:
        pandas.DataFrame:
            A DataFrame containing the stock data from the previous
            trading day.

    """
    print("Updating historical data...")
    # create a Ticker object for Apple Inc.
    stock = yf.Ticker(ticker)

    # get the stock data for Apple Inc. from begin_date to end_date
    stock_data = stock.history(start=begin_date, interval="1d")

    # reset the index of the DataFrame
    stock_data.reset_index(inplace=True)

    # remove first row since it is the same as the last row of the previous DataFrame
    # aapl_stock = aapl_stock.iloc[1:, :]

    # return the DataFrame containing the stock data
    return stock_data


def process_stock_data(stock):
    """
    Processes stock data by removing unnecessary columns, calculating daily returns, and
    converting the datetime column to just the date. Also draws a heatmap correlation matrix
    between stock features and saves the image.

    Args:
        stock: A Pandas DataFrame containing stock data.

    Returns:
        A Pandas DataFrame containing the processed stock data.

    Example Usage:
        apple_data = pd.read_csv('apple_stock_data.csv')
        processed_apple_data = process_apple_stock(apple_data)
    """
    print("Processing stock data...")
    # remove unnecessary columns
    # stock = stock.drop(["Stock Splits"], axis=1)
    # add daily return column
    stock["Daily Return"] = stock["Close"].pct_change()
    # multiply daily return by 100 to get percentage
    stock["Daily Return"] = stock["Daily Return"].apply(lambda x: x * 100)
    # fill NaN values with 0
    stock["Daily Return"] = stock["Daily Return"].fillna(0)
    # convert datetime column to just date
    stock["Date"] = stock["Date"].apply(lambda x: x.date())

    return stock


def save_locally(data, directory):
    print("Saving data locally...")
    data.to_csv(directory)
    print("Data saved locally to: " + directory)
