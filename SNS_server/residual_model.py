import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from data_acquisition import (
    get_historical_data,
    get_updated_stock_data,
    process_stock_data,
    save_locally,
)
import os
import pickle
import tensorflow as tf
from pandas_market_calendars import get_calendar
from datetime import datetime
import matplotlib.dates as mdates
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)
from config import figures_dir, dataset_dir, models_dir

# Define constants
ROLLING_WINDOW = 30
EVALUATE = False


# Helper functions to get paths for dataset and model
def get_data_dir(ticker):
    stock_csv = ticker + "_data.csv"
    stock_dir = os.path.join(dataset_dir, stock_csv)
    return stock_dir


def get_real_data_dir(ticker):
    stock_csv = ticker + "_data_real.csv"
    stock_dir = os.path.join(dataset_dir, stock_csv)
    return stock_dir


def get_model_dir(ticker):
    model_name = ticker + "_model.tf"
    model_dir = os.path.join(models_dir, model_name)
    return model_dir


# Function to find the next market date given a date string
def next_market_date(date_str):
    date = pd.to_datetime(date_str)
    nyse = get_calendar("NYSE")
    end_date = pd.Timestamp(date.year + 1, 12, 31)
    schedule = nyse.schedule(start_date=date, end_date=end_date)
    return schedule.iloc[1]["market_open"].strftime("%Y-%m-%d")


# Function to find the most recent market date given a date string
def most_recent_market_date(date_str):
    date = pd.to_datetime(date_str)
    nyse = get_calendar("NYSE")

    # Check the schedule for the previous 30 days, which should be sufficient
    start_date = date - pd.DateOffset(days=30)
    schedule = nyse.schedule(start_date=start_date, end_date=date)

    # If the given date is a market day, return it
    if not schedule.empty and schedule.iloc[-1]["market_close"].date() == date.date():
        recent_market_day = date
    else:
        # Otherwise, return the most recent market day
        recent_market_day = schedule.iloc[-1]["market_close"].normalize()

    return recent_market_day.strftime("%Y-%m-%d")


# Function to update the stock data with the most recent data
def update_data(ticker, last_date):
    print("Updating data...")
    next_date = next_market_date(last_date)
    data = get_updated_stock_data(ticker, next_date)
    data = process_stock_data(data)
    data.set_index("Date", inplace=True)

    stock_dir = get_data_dir(ticker)
    dataset = pd.read_csv(stock_dir, index_col=0)

    # add new data to the dataset
    dataset_total = pd.concat((dataset, data), axis=0)

    stock_dir = get_data_dir(ticker)
    save_locally(dataset_total, stock_dir)


# Wrap the model to include residuals
class ResidualWrap(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)
        return inputs + delta


# Function to train the model
def train_model(ticker):
    print("Training model...")
    dataset = get_historical_data(ticker)
    dataset = process_stock_data(dataset)
    dataset.set_index("Date", inplace=True)

    if EVALUATE:
        real_stock_dir = get_real_data_dir(ticker)
        save_locally(dataset, real_stock_dir)
        dataset = dataset.iloc[:-ROLLING_WINDOW]
        stock_dir = get_data_dir(ticker)
        save_locally(dataset, stock_dir)
    else:
        stock_dir = get_data_dir(ticker)
        save_locally(dataset, stock_dir)

    train_data = dataset.loc[:, ["Close"]]  # extract the closing price column

    # Scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    training_scaled = sc.fit_transform(train_data)

    pickle.dump(sc, open("scaler.pkl", "wb"))

    # Split train data between x and y components
    x_train = []
    y_train = []

    future_days = ROLLING_WINDOW  # Predict the next X trading days
    for i in range(ROLLING_WINDOW, training_scaled.shape[0] - future_days):
        x_train.append(training_scaled[i - ROLLING_WINDOW : i, 0])
        y_train.append(training_scaled[i : i + future_days, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Arranging keras layers in sequential order
    regressor = Sequential()

    # Layer setup
    regressor.add(
        LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1))
    )
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=100))
    regressor.add(Dropout(0.2))

    regressor.add(
        Dense(units=ROLLING_WINDOW, kernel_initializer=tf.initializers.zeros())
    )
    regressor = ResidualWrap(regressor)
    regressor.compile(optimizer="adam", loss="mean_squared_error")

    regressor.fit(x_train, y_train, epochs=10, batch_size=32)

    # Save the model
    model_dir = get_model_dir(ticker)
    regressor.save(model_dir)


# Function to make predictions using the trained model
def predict(ticker):
    print("Predicting...")
    # Load the saved model
    model_dir = get_model_dir(ticker)
    regressor = load_model(model_dir)
    stock_dir = get_data_dir(ticker)
    dataset = pd.read_csv(stock_dir, index_col=0)

    # load the scaler
    sc = pickle.load(open("scaler.pkl", "rb"))

    prediced_close_prices = []
    dates = []

    inputs = dataset.tail(ROLLING_WINDOW)["Close"]
    inputs = inputs.values.reshape(-1, 1)

    inputs = sc.transform(inputs)
    inputs = np.array(inputs[:, 0])
    inputs = np.reshape(inputs, (1, inputs.shape[0], 1))
    inputs = inputs.reshape(1, -1)
    predicted_stock_price = regressor.predict(inputs)
    predicted_stock_price = predicted_stock_price.reshape(-1, 1)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    last_date = dataset.index[-1]
    for price in predicted_stock_price:
        next_date = next_market_date(last_date)
        last_date = next_date
        prediced_close_prices.append(price[0])
        dates.append(next_date)

    predictions = pd.DataFrame({"Date": dates, "Close": prediced_close_prices})
    predictions.set_index("Date", inplace=True)
    print("predictions", predictions)
    return predictions


# Function to plot the predicted stock prices against the real stock prices
def plot_prediction_vs_real(ticker, real_stock_prices, predictions):
    # Ensure the index is of datetime type
    real_stock_prices.index = pd.to_datetime(real_stock_prices.index)
    predictions.index = pd.to_datetime(predictions.index)

    fig, ax = plt.subplots()

    ax.plot(real_stock_prices, color="red", label=f"Real {ticker} Price")
    ax.plot(predictions, color="blue", label=f"Predicted {ticker} Price")
    ax.set_title(f"{ticker} Price Prediction vs Real Price, {ROLLING_WINDOW} days.")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{ticker} Price")

    # Set the x-axis to display one label per month
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Rotate the x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

    ax.legend()
    # plt.savefig(os.path.join(figures_dir, ticker + f"_comparison_{ROLLING_WINDOW}.png"))

    real_stock_prices, predictions = (
        real_stock_prices[["Close"]].to_numpy(),
        predictions[["Close"]].to_numpy(),
    )

    # accuracy = accuracy_score(real_stock_prices, predictions)
    # f1 = f1_score(real_stock_prices, predictions)
    # print(f"Accuracy: {accuracy}")
    # print(f"F1 score: {f1}")
    metrics_file = os.path.join(figures_dir, "metrics_file.csv")
    if not os.path.exists(metrics_file):
        with open(metrics_file, "w") as fd:
            fd.write("days,ticker,mse,mape,r2_score\n")
    mse = mean_squared_error(real_stock_prices, predictions)
    mape = mean_absolute_percentage_error(real_stock_prices, predictions)
    r2 = r2_score(real_stock_prices, predictions)
    friendly_mse = "{:.2f}".format(mse)
    friendly_mape = "{:.2f}".format(mape)
    friendly_r2 = "{:.2f}".format(r2)
    csv_row = (
        f"{ROLLING_WINDOW},{ticker},{friendly_mse},{friendly_mape},{friendly_r2}\n"
    )
    with open(metrics_file, "a") as fd:
        fd.write(csv_row)
    # font = {'family': 'serif',
    #     'color':  'darkred',
    #     'weight': 'normal',
    #     'size': 16,
    # }
    ax.grid(True, which="both", ls="-", color="0.65")
    plt.savefig(os.path.join(figures_dir, ticker + f"_comparison_{ROLLING_WINDOW}.png"))


def plot_prediction(ticker, predictions):
    stock_dir = get_data_dir(ticker)
    dataset = pd.read_csv(stock_dir, index_col=0)

    # Convert the index to datetime objects
    dataset.index = pd.to_datetime(dataset.index)
    predictions.index = pd.to_datetime(predictions.index)

    fig, ax = plt.subplots()

    ax.plot(
        dataset.tail(60).index,
        dataset.tail(60)["Close"],
        color="red",
        label=f"Historical {ticker} Price",
    )
    ax.plot(
        predictions.index,
        predictions["Close"],
        color="blue",
        label=f"Predicted {ticker} Price",
    )
    ax.set_title(f"{ticker} Price Prediction")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{ticker} Price")

    # Set the x-axis to display one label per month
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True)

    # Rotate the x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

    ax.legend()
    plt.savefig(os.path.join(figures_dir, ticker + "_prediction.png"))


def get_prediction(ticker, days):
    model_dir = get_model_dir(ticker)
    stock_dir = get_data_dir(ticker)
    if not os.path.exists(model_dir) or not os.path.exists(stock_dir) or EVALUATE:
        train_model(ticker)

    data = pd.read_csv(stock_dir, index_col=0)
    last_date = data.index[-1]
    last_date = datetime.strptime(last_date, "%Y-%m-%d")

    last_market_date = most_recent_market_date(datetime.today())
    last_market_date = datetime.strptime(last_market_date, "%Y-%m-%d")

    if last_market_date != last_date and not EVALUATE:
        update_data(ticker, last_date)

    prediction = predict(ticker)

    try:
        adjusted_prediction = prediction.head(days)
    # except out of index error
    except:
        print("The maximum number of days you can predict is 30")
        adjusted_prediction = prediction

    return adjusted_prediction


def main(tkr):
    if EVALUATE:
        out = get_prediction(tkr, ROLLING_WINDOW)

        stock_dir = get_real_data_dir(tkr)
        dataset = pd.read_csv(stock_dir, index_col=0)
        real_close_prices = []
        real_dates = []

        inputs = dataset.tail(ROLLING_WINDOW)["Close"]
        inputs = inputs.values.reshape(-1, 1)

        index = -1 - ROLLING_WINDOW
        last_date = dataset.index[index]
        print("last_date", last_date)
        for price in inputs:
            next_date = next_market_date(last_date)
            last_date = next_date
            real_close_prices.append(price[0])
            real_dates.append(next_date)

        real_data = pd.DataFrame({"Date": real_dates, "Close": real_close_prices})
        real_data.set_index("Date", inplace=True)

        print("\nreal data")
        print(real_data.tail(ROLLING_WINDOW))

        # plot_prediction(tkr, out)
        plot_prediction_vs_real(tkr, real_data, out)

    else:
        days = 30

        out = get_prediction(tkr, days)
        print("\nout")
        print(out.tail(days))


if __name__ == "__main__":
    main("GOOGL")
    main("AAPL")
    main("MSFT")
    main("AMZN")
    main("TSLA")
    main("NFLX")
    main("NVDA")
    main("PYPL")
    main("GME")
    main("QQQ")
    main("KO")
    main("ORCL")
    main("IBM")
    main("MCD")
    main("NKE")
    main("DIS")
    main("ADBE")
    main("UPS")
