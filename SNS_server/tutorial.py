import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

# Imports from rnn_model.py
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


# tutorial starts here
mpl.rcParams["figure.figsize"] = (8, 6)
mpl.rcParams["axes.grid"] = False

MAX_EPOCHS = 200
OUT_STEPS = 30
ticker = "AAPL"
TRAIN_PART = 0.7
TRAIN_VAL_PART = 0.9


stock_dir = get_data_dir(ticker)
# dataset = pd.read_csv(stock_dir, index_col=0)
dataset = get_historical_data(ticker)
dataset = process_stock_data(dataset)
dataset.set_index("Date", inplace=True)
# try:
#     sc = pickle.load(open("scaler.pkl", "rb"))
# except FileNotFoundError:
#     sc = MinMaxScaler(feature_range=(0, 1))
#     train_data = dataset.loc[:, ["Close"]]  # extract the closing price column
#     training_scaled = sc.fit_transform(train_data)
#     pickle.dump(sc, open("scaler.pkl", "wb"))

df = dataset
df = df.drop(columns=["Open", "High", "Low", "Volume", "Dividends", "Daily Return"])


column_indices = {name: i for i, name in enumerate(df.columns)}


n = len(df)
train_df = df[0 : int(n * TRAIN_PART)]
val_df = df[int(n * TRAIN_PART) : int(n * TRAIN_VAL_PART)]
test_df = df[int(n * TRAIN_VAL_PART) :]

num_features = df.shape[1]

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

print("df head: ", df.head())
print("...")
print("df tail: ", df.tail())
print("train_df.shape: ", train_df.shape)
print("val_df.shape: ", val_df.shape)
print("test_df.shape: ", test_df.shape)
print("num_features: ", num_features)
print("column_indices: ", column_indices)
print("df.columns: ", df.columns)


class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)

        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state, training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions


class WindowGenerator:
    def __init__(
        self,
        input_width,
        label_width,
        shift,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_columns=None,
    ):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}",
            ]
        )

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [
                    labels[:, :, self.column_indices[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col="Close", max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f"{plot_col} [normed]")
            plt.plot(
                self.input_indices,
                inputs[n, :, plot_col_index],
                label="Inputs",
                marker=".",
                zorder=-10,
            )

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(
                self.label_indices,
                labels[n, :, label_col_index],
                edgecolors="k",
                label="Labels",
                c="#2ca02c",
                s=64,
            )
            if model is not None:
                predictions = model(inputs)
                plt.scatter(
                    self.label_indices,
                    predictions[n, :, label_col_index],
                    marker="X",
                    edgecolors="k",
                    label="Predictions",
                    c="#ff7f0e",
                    s=64,
                )

            if n == 0:
                plt.legend()

        plt.xlabel("Time [h]")
        plt.savefig(os.path.join(figures_dir, ticker + "_prediction.png"))

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=32,
        )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min"
    )

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    history = model.fit(
        window.train,
        epochs=MAX_EPOCHS,
        validation_data=window.val,
        callbacks=[early_stopping],
    )
    return history


multi_window = WindowGenerator(
    input_width=df.shape[1], label_width=OUT_STEPS, shift=OUT_STEPS
)


feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)

prediction, state = feedback_model.warmup(multi_window.example[0])
prediction.shape

print(
    "Output shape (batch, time, features): ",
    feedback_model(multi_window.example[0]).shape,
)

history = compile_and_fit(feedback_model, multi_window)

IPython.display.clear_output()

multi_val_performance = {}
multi_performance = {}

multi_val_performance["AR LSTM"] = feedback_model.evaluate(multi_window.val)
multi_performance["AR LSTM"] = feedback_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(feedback_model, plot_col="Close")

# x = np.arange(len(multi_performance))
# width = 0.3

# metric_name = 'mean_absolute_error'
# metric_index = lstm_model.metrics_names.index('mean_absolute_error')
# val_mae = [v[metric_index] for v in multi_val_performance.values()]
# test_mae = [v[metric_index] for v in multi_performance.values()]

# plt.bar(x - 0.17, val_mae, width, label='Validation')
# plt.bar(x + 0.17, test_mae, width, label='Test')
# plt.xticks(ticks=x, labels=multi_performance.keys(),
#            rotation=45)
# plt.ylabel(f'MAE (average over all times and outputs)')
# _ = plt.legend()

# for name, value in multi_performance.items():
#     print(f'{name:8s}: {value[1]:0.4f}')
