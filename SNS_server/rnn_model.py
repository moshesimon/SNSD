import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from SNS_server.data_acquisition import (
    get_historical_data,
    get_updated_stock_data,
    process_stock_data,
    save_locally,
)
import os
import pickle
from config import figures_dir, dataset_dir, models_dir
from pandas_market_calendars import get_calendar

ROLLING_WINDOW = 60


def get_data_dir(ticker):
    stock_csv = ticker + "_data.csv"
    stock_dir = os.path.join(dataset_dir, stock_csv)
    return stock_dir


def get_model_dir(ticker):
    model_name = ticker + "_model.h5"
    model_dir = os.path.join(models_dir, model_name)
    return model_dir


def next_market_date(date_str):
    date = pd.to_datetime(date_str)
    nyse = get_calendar("NYSE")
    end_date = pd.Timestamp(date.year + 1, 12, 31)
    schedule = nyse.schedule(start_date=date, end_date=end_date)
    return schedule.iloc[1]["market_open"].strftime("%Y-%m-%d")


# Train the model
def train_model(ticker):
    print("Training model...")
    dataset = get_historical_data(ticker)
    dataset = process_stock_data(dataset)
    dataset.set_index("Date", inplace=True)

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

    for i in range(ROLLING_WINDOW, training_scaled.shape[0]):
        x_train.append(training_scaled[i - ROLLING_WINDOW : i, 0])
        y_train.append(training_scaled[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape inputs
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Arranging keras layers in sequential order
    regressor = Sequential()

    # Layer setup
    regressor.add(
        LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1))
    )
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=100))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units=1))

    regressor.compile(optimizer="adam", loss="mean_squared_error")

    regressor.fit(x_train, y_train, epochs=10, batch_size=32)

    # Save the model
    model_dir = get_model_dir(ticker)
    regressor.save(model_dir)


# Update the model with yesterday's price
def update_model(ticker, last_date):
    print("Updating model...")
    # Load the saved model
    model_dir = get_model_dir(ticker)
    regressor = load_model(model_dir)

    # Get updated data
    new_data = get_updated_stock_data(ticker, last_date)

    print(new_data.tail())
    new_data = process_stock_data(new_data)
    new_data.set_index("Date", inplace=True)

    stock_dir = get_data_dir(ticker)
    dataset = pd.read_csv(stock_dir, index_col=0)

    # add new data to the dataset
    dataset_total = pd.concat((dataset, new_data), axis=0)
    # save the updated dataset
    save_locally(dataset_total, stock_dir)

    # extract the closing price column
    dataset_total = dataset_total.loc[:, ["Close"]]

    sc = MinMaxScaler(feature_range=(0, 1))
    sc = sc.fit(dataset_total)

    training_scaled = sc.transform(dataset_total)
    pickle.dump(sc, open("scaler.pkl", "wb"))
    x_train = []
    y_train = []
    print(new_data.shape[0])
    for i in range(new_data.shape[0]):
        if i == 0:
            x_train.append(training_scaled[-ROLLING_WINDOW:, 0])
            y_train.append(training_scaled[-1, 0])
        else:
            x_train.append(training_scaled[-i - ROLLING_WINDOW : -i, 0])
            y_train.append(training_scaled[-i - 1, 0])
    x_train.reverse()

    y_train.reverse()

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Train the model
    regressor.fit(x_train, y_train, epochs=1, batch_size=1)
    # Save the updated model
    model_dir = get_model_dir(ticker)
    regressor.save(model_dir)


def predict(ticker, days):
    print("Predicting...")
    # Load the saved model
    model_dir = get_model_dir(ticker)
    regressor = load_model(model_dir)
    stock_dir = get_data_dir(ticker)
    dataset = pd.read_csv(stock_dir, index_col=0)
    # load the scaler
    sc = pickle.load(open("scaler.pkl", "rb"))

    # remove the 30 rows from the dataset
    # dataset = dataset.iloc[:-30, :]

    prediced_close_prices = []
    dates = []
    for x in range(days):
        inputs = dataset.tail(ROLLING_WINDOW)["Close"]
        inputs = inputs.values.reshape(-1, 1)

        inputs = sc.transform(inputs)
        inputs = np.array(inputs[:, 0])
        inputs = np.reshape(inputs, (1, inputs.shape[0], 1))
        predicted_stock_price = regressor.predict(inputs)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        # append the predicted value to the train data with the next date generated by next_market_date

        # last close price
        last_close = dataset.iloc[-1, 3]
        print(dataset.tail(1))
        print(f"last close price ({x}): ", last_close)
        # if x < 10:
        #     last_date = dataset.index[-1]
        #     next_date = next_market_date(last_date)
        #     dataset.loc[next_date, 'Close'] = last_close #predicted_stock_price[0][0]
        #     continue
        # elif x == 10:
        #     # remove last 5 rows
        #     dataset = dataset.iloc[:-10, :]
        last_date = dataset.index[-1]
        next_date = next_market_date(last_date)
        dataset.loc[next_date, "Close"] = predicted_stock_price[0][0]
        prediced_close_prices.append(predicted_stock_price[0][0])
        dates.append(next_date)

    predictions = pd.DataFrame({"Date": dates, "Close": prediced_close_prices})
    predictions.set_index("Date", inplace=True)
    return predictions


def plot_prediction_vs_real(real_stock_prices, predictions):
    plt.plot(real_stock_prices, color="red", label="Real Apple Stock Price")
    plt.plot(predictions, color="blue", label="Predicted Apple Stock Price")
    plt.title("Apple Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Apple Stock Price")
    plt.legend()
    plt.show()


def plot_prediction(ticker, predictions):
    stock_dir = get_data_dir(ticker)
    dataset = pd.read_csv(stock_dir, index_col=0)
    plt.plot(
        dataset.tail(60).index,
        dataset.tail(60)["Close"],
        color="red",
        label=f"Historical {ticker} Price",
    )
    plt.plot(
        predictions.index,
        predictions["Close"],
        color="blue",
        label=f"Predicted {ticker} Price",
    )
    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel(f"{ticker} Stock Price")
    plt.legend()
    plt.savefig(os.path.join(figures_dir, ticker + "_prediction.png"))
    plt.show()


# check for stock_predictor.h5
def get_prediction(ticker, days):
    model_dir = get_model_dir(ticker)
    stock_dir = get_data_dir(ticker)
    if not os.path.exists(model_dir) or not os.path.exists(stock_dir):
        train_model(ticker)

    data = pd.read_csv(stock_dir, index_col=0)
    last_date = data.index[-1]
    # check if the last row is today's date
    if last_date != f"{pd.Timestamp.today().date()}":
        update_model(ticker, last_date)

    prediction = predict(ticker, days)

    return prediction


def main():
    out = get_prediction('QQQ', 30)
    print(out)
    plot_prediction('QQQ', out)


if __name__ == "__main__":
    # main()
    pass
