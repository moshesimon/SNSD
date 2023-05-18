#!/usr/bin/env python3
import random
import socket
import json
import string
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import LabelEncoder
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Flatten, SpatialDropout1D
from keras.models import Sequential
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import plots
from config import figures_dir, dataset_dir


HOST = "127.0.0.1"
PORT = 65433


def isTicker(
    t: str,
) -> bool:  # Finction to check if ticker/stock name exists on yahoo finance
    ticker = yf.Ticker(t)
    info = ticker.history(period="7d", interval="1d")
    return len(info) > 0


def isfloat(num):  # Function to return true if string is a float
    try:
        float(num)
        return True
    except ValueError:
        return False


def get_dir(ticker):
    stock_csv = ticker + "_data.csv"
    stock_dir = os.path.join(dataset_dir, stock_csv)
    return stock_dir


def send_stock(stock):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
            stock_pickle = pickle.dumps(
                stock
            )  # Convert list into pickle object (easy to send via socket)
            s.send(stock_pickle)
            results_pickle = s.recv(4096)  # Recieve results from server
            results = pickle.loads(results_pickle)  #'unpickle' data into a list
            return results
        except ConnectionRefusedError:
            print("Connection refused. Server not running?")


def main():
    ###### MACHINE LEARNING ALGORITHM ######

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

    ###### USER INTERFACE/SERVER COMMUNICATION ######

    continue_loop = True
    while continue_loop:
        try:
            user_input = str(input("\nHow can I help you?\n"))
        except TypeError:
            print("Invalid input, please try again.\n")

        ###Convert user input to same format as training data ###
        # Convert to lower case
        pred_words = []
        pred_input = [
            letters.lower()
            for letters in user_input
            if letters not in string.punctuation
        ]
        pred_input = "".join(pred_input)
        pred_words.append(pred_input)

        # Tokenize the words
        pred_input = tokenizer.texts_to_sequences(pred_words)
        pred_input = np.array(pred_input).reshape(-1)
        pred_input = pad_sequences([pred_input], input_size)

        # use the model to find most likely tag
        output = model.predict(pred_input, verbose=0)
        # print(output)
        if np.max(output) >= 0.50:  # Checks that the network is confident in its answer
            output = output.argmax()
            # print(output)
            tag = encoder.inverse_transform([output])[0]
            # Print a random response stored in the .json file (only used for greetings and goodbyes)
            print(random.choice(responses[tag]))
        else:
            tag = "no_tag"

        ### ACTIONS FOR EACH TAG ###
        """""" """""" """
        The client will send the relevant data, followed by a number (1-8) to the server in the form of a list. The number at the end of the list
        determins the function that the server will use.
        """ """""" """"""

        stockList = []

        # Find the price of a stock at a given day
        if tag == "get_price":
            valid = False
            while valid == False:
                stock = input("Which stock would you like to select? ")
                days = input("How many days ahead should I predict? (max 30): ")
                if isTicker(stock) and days.isdigit() and int(days) <= 30:
                    valid = True
                else:
                    print("Invalid input. Please try again.\n")

            stockList.append(stock)
            stockList.append(days)
            stockList.append("1")
            print("\nPredicting...\n")
            predictions = send_stock(stockList)
            pred_close = predictions["Close"].tolist()
            price = round(pred_close[int(days) - 1], 2)
            print(
                f"The closing price of {stock} {days} days from now will be ${price}\n"
            )
            print(
                "Here is a graph of my predictions, a copy has been saved in the 'Plots' folder. Please close the plot window to continue."
            )
            plots.plot_prediction(stock, predictions)

        # Find the daily returns of a stock/portfolio until a given day
        elif tag == "get_daily_return":
            valid = False
            num = 1
            while valid == False:
                stock = input(f"Please enter stock {num} (or type 'Done' to finish): ")
                if stock.lower() == "done" and len(stockList) > 0:
                    valid = True
                elif isTicker(stock):
                    stockList.append(stock)
                    num += 1
                else:
                    print("Invalid input. Please try again.\n")

            valid = False
            while valid == False:
                days = input("How many days ahead should I predict? (max 30): ")
                if days.isdigit() and int(days) <= 30:
                    valid = True
                else:
                    print("Invalid input. Please try again.\n")
            stockList.append(days)
            data_rec = []
            # Check if user wants to predict a single stock or multiple, and select the relevent choice
            if (
                len(stockList) == 2
            ):  # Checks that the list only contains one stock (and the number of days to predict)
                stockList.append("2")
                print("\nPredicting...\n")
                data_rec = send_stock(stockList)
                data_round = np.round(data_rec, 2)
                print(
                    f"The daily returns of {stockList[0]} over {days} days will be: \n"
                )
                print(data_round)
                print(
                    "Here is a graph of my predictions, a copy has been saved in the 'Plots' folder. Please close the plot window to continue."
                )
                plots.plot_daily_returns(stockList[0], data_rec)
            else:  # List contains more than one stock and no. of days, so must contain multiple stocks
                stockList.append("6")
                print("\nPredicting...\n")
                data_rec = send_stock(stockList)
                data_round = data_rec.round(2)
                print(
                    f"The daily returns for your chosen stocks over {days} days will be:\n"
                )
                print(data_round)
                valid = False
                while valid == False:
                    try:
                        graph_type = int(
                            input(
                                "\nWould you like to see a graph of 1. Daily returns or 2. Cumulative returns of the portfolio (please type 1 or 2 to select option): "
                            )
                        )
                    except TypeError:
                        print("Invalid input, please try again.\n")
                    if graph_type == 1 or graph_type == 2:
                        valid = True
                    else:
                        print(
                            f"{graph_type} is not a valid input, please enter 1 or 2 accordingly.\n"
                        )
                print(
                    "Here is a graph of my predictions, a copy has been saved in the 'Plots' folder. Please close the plot window to continue."
                )

                if graph_type == 1:
                    plots.plot_daily_portfolio_returns(
                        stockList[0 : len(stockList) - 2], data_rec
                    )
                else:
                    plots.plot_cumulative_portfolio_returns(
                        stockList[0 : len(stockList) - 2], data_rec
                    )

        # Find the average return of a stock over a number of days
        elif tag == "get_avg_return":
            valid = False
            while valid == False:
                stock = input("Which stock would you like to select? ")
                days = input("How many days ahead should I predict? (max 30): ")
                if isTicker(stock) and days.isdigit() and int(days) <= 30:
                    valid = True
                else:
                    print("Invalid input. Please try again.\n")
            stockList.append(stock)
            stockList.append(days)
            stockList.append("3")
            print("\nPredicting...\n")
            data_rec = round(send_stock(stockList), 2)
            print(
                f"The average return of {stock} over {days} days will be ${data_rec}\n"
            )

        # Find the volatility of a stock over a number of days
        elif tag == "get_std":
            valid = False
            while valid == False:
                stock = input("Which stock would you like to select? ")
                days = input("How many days ahead should I predict? (max 30): ")
                if isTicker(stock) and days.isdigit() and int(days) <= 30:
                    valid = True
                else:
                    print("Invalid input. Please try again.\n")
            stockList.append(stock)
            stockList.append(days)
            stockList.append("4")
            print("\nPredicting...\n")
            data_rec = round(send_stock(stockList), 2)
            print(f"The volatility of {stock} over {days} days will be {data_rec}\n")

        # Find the sharpe ratio of a stock at a given rfr over a number of days
        elif tag == "get_sharpe":
            valid = False
            while valid == False:
                stock = input("Which stock would you like to select? ")
                days = input("How many days ahead should I predict? (max 30): ")
                rfr = input("What risk free rate would you like to use? ")
                if (
                    isTicker(stock)
                    and days.isdigit()
                    and int(days) <= 30
                    and isfloat(rfr)
                ):
                    valid = True
                else:
                    print("Invalid input. Please try again.\n")

            stockList.append(stock)
            stockList.append(
                rfr
            )  # appended before days and choice so that the server can process the data correctly
            stockList.append(days)
            stockList.append("5")
            print("\nPredicting...\n")
            data_rec = round(send_stock(stockList), 2)
            print(f"The sharpe ratio of {stock} over {days} days will be {data_rec}\n")

        # Optimise a portfolio of stocks (find the optimal ratio to split the investment) to minimise variance or maximise sharp ratio (chosen by user)
        elif tag == "optimise":
            # Stock selection
            valid = False
            num = 1
            while valid == False:
                stock = input(f"Please enter stock {num} (or type 'Done' to finish): ")
                if stock.lower() == "done" and len(stockList) > 0:
                    valid = True
                elif isTicker(stock):
                    stockList.append(stock)
                    num += 1
                else:
                    print("Invalid input. Please try again.\n")

            valid = False
            while valid == False:
                days = input("How many days ahead should I predict? (max 30): ")
                if days.isdigit() and int(days) <= 30:
                    valid = True
                else:
                    print("Invalid input. Please try again.\n")

            # Optimisation type selection
            valid = False
            while valid == False:
                try:
                    optimise_type = int(
                        input(
                            "Would you like to optimise for 1. minimum variance or 2. maximum sharpe ratio? (please type 1 or 2 to select option): "
                        )
                    )
                except TypeError:
                    print("Invalid input, please try again.\n")
                if optimise_type == 1 or optimise_type == 2:
                    valid = True
                else:
                    print(
                        f"{optimise_type} is not a valid input, please enter 1 or 2 accordingly.\n"
                    )

            # Minimum variance optimisation actions
            if optimise_type == 1:
                stockList.append(days)
                stockList.append("7")
                print("\nPredicting...\n")
                data_rec = send_stock(stockList)
                min_var = data_rec.pop()

                print(f"The optimal ratio to split your investment is as follows:")

                for i in range(len(stockList) - 2):
                    print(f"{stockList[i]}: {round(data_rec[i]*100, 2)}%")
                print(f"This produces a variance of {min_var}")
                print(
                    "Here is a chart of my predictions, a copy has been saved in the 'Plots' folder. Please close the plot window to continue."
                )
                plots.plot_opt_portfolio(
                    stockList[0 : len(stockList) - 2],
                    data_rec[0 : len(stockList) - 2],
                    "var",
                )

            # Maximum sharpe ratio actions
            else:
                valid = False
                while valid == False:
                    rfr = input("What risk free rate would you like to use?: ")
                    if isfloat(rfr):
                        valid = True
                    else:
                        print("Invalid input. Please try again.\n")

                stockList.append(rfr)
                stockList.append(days)
                stockList.append("8")
                print("\nPredicting...\n")
                data_rec = send_stock(stockList)
                max_sharpe = data_rec.pop()

                print(f"The optimal ratio to split your investment is as follows:")

                for i in range(len(stockList) - 3):
                    print(f"{stockList[i]}: {round(data_rec[i]*100, 2)}%")
                print(f"This produces a sharpe ratio of {max_sharpe}")
                print(
                    "Here is a chart of my predictions, a copy has been saved in the 'Plots' folder. Please close the plot window to continue."
                )
                plots.plot_opt_portfolio(
                    stockList[0 : len(stockList) - 3],
                    data_rec[0 : len(stockList) - 3],
                    "sharpe",
                )

        # Stop the program is the user says goodbye
        elif tag == "goodbye":
            quit()

        elif tag == "no_tag":
            print(
                "I'm not sure how to help you with that. Please try asking me something different, or type 'help' to see a list of things that I can do.\n"
            )

        elif tag == "help":
            print(
                "I am a chatbot that can predict a number of different things to do with stocks. Here is a list of things I can help you with:"
            )
            print(" 1. Predict the closing price of a stock")
            print(" 2. Predict the daily returns of a stock/portfolio")
            print(" 3. Predict the average returns of a stock")
            print(" 4. Predict the volatility of a stock")
            print(" 5. Predict the sharpe ratio of a stock")
            print(
                " 6. Optimise a portfolio for either minimum variance or maximum sharpe ratio\n"
            )


if __name__ == "__main__":
    main()
