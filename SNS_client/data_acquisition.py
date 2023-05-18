import os
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd


def get_historical_data(ticker):
    """
    Retrieves historical stock data from Yahoo Finance.

    Args:
        None

    Returns:
        pandas.DataFrame:
            A DataFrame containing the historical stock data.

    """
    # print("Fetching historical data...")
    # create a Ticker object
    stock = yf.Ticker(ticker)

    # get the historical stock data from 5 months ago until today.
    stock_data = stock.history(period="6mo", interval="1d")

    # reset the index of the DataFrame
    stock_data.reset_index(inplace=True)

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
    # print("Processing stock data...")
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
