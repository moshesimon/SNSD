import random
import json
import string
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Flatten, SpatialDropout1D
from keras.models import Sequential
import matplotlib.pyplot as plt


# Load data from .json file
intents = json.loads(open("intents.json").read())

inputs = []
classes = []
responses = {}

# Append data from json file to lists
for intent in intents["intents"]:
    responses[intent["tag"]] = intent["responses"]
    for pattern in intent["patterns"]:
        inputs.append(pattern)
        classes.append(intent["tag"])

# Convert lists to pd dataframe
data = pd.DataFrame({"inputs": inputs, "classes": classes})


# Remove punctuation from inputs
data["inputs"] = data["inputs"].apply(
    lambda word: [
        letters.lower() for letters in word if letters not in string.punctuation
    ]
)
data["inputs"] = data["inputs"].apply(lambda word: "".join(word))

# Tokenize words and convert to np array
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data["inputs"])
train = tokenizer.texts_to_sequences(data["inputs"])
xtrain = pad_sequences(
    train
)  # Padding converts list of sequences to a 2D np array with length of longest sequence in list (makes all inputs uniform)


# Encode classes to numerical values
encoder = LabelEncoder()
ytrain = encoder.fit_transform(data["classes"])

# training data info
input_size = xtrain.shape[1]
vocab = len(tokenizer.word_index)
output_size = encoder.classes_.shape[0]

# creating the model
model = Sequential()
model.add(Embedding(input_dim=vocab + 1, output_dim=10, input_length=input_size))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(10, dropout=0.3, recurrent_dropout=0.3))
model.add(Flatten())
model.add(Dense(output_size, activation="softmax"))

# compile model
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# train the model
print("\nChatbot is loading...\n")
hist = model.fit(xtrain, ytrain, epochs=400, verbose=0)

fig, ax1 = plt.subplots()

ax1.plot(hist.history["accuracy"], color="tab:red")

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy", color="tab:red")
ax1.tick_params(axis="y", labelcolor="tab:red")

ax2 = ax1.twinx()

ax2.set_ylabel("Loss", color="tab:blue")
ax2.plot(hist.history["loss"], color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")

plt.title("NLP training metrics")
fig.tight_layout()

plt.savefig("NLP_metrics.png")
plt.show()


test_df_data = {
    "inputs": [
        "Predict the price of a stock",  # get_price
        "What price will NVDA be tomorrow",
        "Find the price of a stock in 5 days",
        "What price will my stock be in the future",
        "what will the price of GOOG be",
        "Find the returns of my portfolio",  # get_daily_returns
        "What will the daily profit of my stock be",
        "How much money will I make each day",
        "Predict the daily returns of PLTR",
        "How much profit will I make each day",
        "what will the average return of NVDA be",  # get_avg_returns
        "Predict the average profit of a stock",
        "How much money will I make on average",
        "what's the expected returns on this stock",
        "Find the average returns of QQQ",
        "What will the volatility of GOOG be",
        "Find the volatility of a stock",  # get_std
        "Will PLTR be volatile",
        "Predict the standard deviation of a stock",
        "Predict how volitile META will be",
        "What will the volatility of a stock be",
        "Predict the sharpe ratio of GOOG",  # get_sharpe
        "Assess the performance of TSLA",
        "What sharp ratio will this stock have",
        "Find the sharpe ratio",
        "Predict if a stock will perform well",
        "Optimise my portfolio for minimum variance",  # optimise
        "How should I invest in these stocks?"
        "Find the optimal way to invest in these stocks",
        "Minimise the variance of these stocks",
        "How can I maximise the sharpe ratio for my portfolio",
        "What can you do",  # help
        "Help",
        "Help me",
        "What are you",
        "What things can you do",
    ],
    "classes": [
        "get_price",
        "get_price",
        "get_price",
        "get_price",
        "get_price",
        "get_daily_return",
        "get_daily_return",
        "get_daily_return",
        "get_daily_return",
        "get_daily_return",
        "get_avg_return",
        "get_avg_return",
        "get_avg_return",
        "get_avg_return",
        "get_avg_return",
        "get_std",
        "get_std",
        "get_std",
        "get_std",
        "get_std",
        "get_sharpe",
        "get_sharpe",
        "get_sharpe",
        "get_sharpe",
        "get_sharpe",
        "optimise",
        "optimise",
        "optimise",
        "optimise",
        "optimise",
        "help",
        "help",
        "help",
        "help",
        "help",
    ],
}

test_df = pd.DataFrame(test_df_data)


# Remove punctuation from inputs
test_df["inputs"] = test_df["inputs"].apply(
    lambda word: [
        letters.lower() for letters in word if letters not in string.punctuation
    ]
)
test_df["inputs"] = test_df["inputs"].apply(lambda word: "".join(word))

# Tokenize words and convert to np array

test = tokenizer.texts_to_sequences(test_df["inputs"])

xtest = pad_sequences(test, input_size)


# Encode classes to numerical values
ytest = encoder.transform(test_df["classes"])


print("Predicting...\n")

results = model.evaluate(xtest, ytest)
print("Accuracy:", results[1])
