#!/usr/bin/env python3
import socket
from _thread import start_new_thread
import threading
import pickle
import analysis


HOST = "127.0.0.1"
PORT = 65433

### CALL ANALYSIS.PY FUNCIONS ###


def get_price(stock_name, day):
    predictions = analysis.get_price(stock_name, day)
    return predictions


def get_daily_returns(stock_name, days):
    returns = analysis.get_daily_returns(stock_name, days)
    return returns


def get_avg_daily_returns(stock_name, days):
    returns = analysis.get_avg_daily_return(stock_name, days)
    return returns


def get_std(stock_name, days):
    std = analysis.get_std(stock_name, days)
    return std


def get_sharpe(stock_name, days, rfr):
    sharpe = analysis.get_sharpe(stock_name, days, rfr)
    return sharpe


def get_portfolio_returns(stocks, days):
    returns, returns_df = analysis.get_portfolio_returns(stocks, days)
    return returns_df


def min_var_portfolio(stocks, days):
    combined_returns, combined_returns_df = analysis.get_portfolio_returns(stocks, days)
    opt_weights, min_var = analysis.min_var_portfolio(combined_returns)

    # Combine the two variables to a single list (simplifies communication between client and server)
    data_out = opt_weights.tolist()  # Convert np array to list
    data_out.append(min_var)

    return data_out


def max_sharpe_portfolio(stocks, days, rfr):
    combined_returns, combined_returns_df = analysis.get_portfolio_returns(stocks, days)
    opt_weights, max_sharpe = analysis.max_sharpe_portfolio(combined_returns, rfr)

    # Combine the two variables to a single list (simplifies communication between client and server)
    data_out = opt_weights.tolist()  # Convert np array to list
    data_out.append(max_sharpe)

    return data_out


### CLIENT HANDLING ###


def client_connection_thread(conn, port, lock):
    continue_loop = True
    while continue_loop:
        data = conn.recv(4096)  # Recieve pickle object from client
        if data:
            decoded = pickle.loads(data)  #'unpickle' data
            print("Received:", decoded)
            choice = str(
                decoded.pop()
            )  # Pop last element (the chocice based on the tag predicted by the NLP algorithm on the client side)
            days = int(
                decoded.pop()
            )  # Pop the 2nd last element (the number of days ahead to predict)
            stock_name = decoded[
                0
            ]  # This is only used in the functions involving a single stock, so will always be the first element in the list
            predictions = []
            if choice == "1":  # Predict price of stock
                data_out = get_price(stock_name, days)

            elif choice == "2":  # Predict returns of (single) stock
                data_out = get_daily_returns(stock_name, days)

            elif choice == "3":  # Predict average returns of a stock
                data_out = get_avg_daily_returns(stock_name, days)

            elif choice == "4":  # Predict volatility of stock
                data_out = get_std(stock_name, days)

            elif choice == "5":  # Predict sharpe ratio of stock
                rfr = float(
                    decoded.pop()
                )  # Pop the last element of the list (after choice and days are removed this will be the rfr)
                data_out = get_sharpe(stock_name, days, rfr)

            elif choice == "6":  # Predict the returns of a portfolio of stocks
                data_out = get_portfolio_returns(decoded, days)

            elif choice == "7":  # Optimise portfolio for minimum variance
                data_out = min_var_portfolio(decoded, days)

            elif choice == "8":  # Optimise portfolio for maximum sharpe ratio
                rfr = float(
                    decoded.pop()
                )  # Pop last element of the list (after choice and days are removed this will be the rfr)
                data_out = max_sharpe_portfolio(decoded, days, rfr)
        else:  # If no data is present close the connection
            print(f"Closing connection on port {port}")
            lock.release()  # release lock before breaking
            continue_loop = False

        data_pickle = pickle.dumps(data_out)  # Convert list to pickle object
        conn.send(data_pickle)  # send data back to client

    conn.close()


def main():
    print_lock = threading.Lock()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        print("Socket binded to port")
        s.listen()
        continue_loop = True
        running_threads = []
        while continue_loop:
            try:
                conn, addr = s.accept()
                print_lock.acquire()
                print("Connected to:", addr[0], ":", addr[1])
                new_thread = threading.Thread(
                    target=client_connection_thread,
                    args=(
                        conn,
                        addr[1],
                        print_lock,
                    ),
                    daemon=True,
                )
                new_thread.start()
                running_threads.append(new_thread)
            except KeyboardInterrupt:
                print("Quitting...")
                continue_loop = False
                for thread in running_threads:
                    thread.join()


if __name__ == "__main__":
    main()
