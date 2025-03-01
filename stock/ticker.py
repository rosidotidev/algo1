import yfinance as yf
import os
import numpy as np
import time
import pandas as pd

def read_from_csv(file_path):
    return pd.read_csv(file_path)
# Function to generate forward-looking labels for trading signals
def forward_looking_label(data, look_ahead=5, threshold_buy=0.08, threshold_sell=0.04):
    """
    Generates forward-looking labels based on future price movements.

    Parameters:
    - data: DataFrame containing at least a 'Close' column with closing prices.
    - look_ahead: Number of periods to look ahead to determine future returns.
    - threshold: Minimum return percentage to classify an entry as buy (1) or sell (-1).

    Returns:
    - A new DataFrame with an added 'Target' column containing:
      2  -> Buy signal if future return > threshold
      1 -> Sell signal if future return < -threshold
      0  -> Hold otherwise
    """

    # Create a copy to avoid modifying the original DataFrame
    data = data.copy()

    # Calculate the percentage return after 'look_ahead' periods
    data['Future_Return'] = (data['Close'].shift(-look_ahead) - data['Close']) / data['Close']

    # Define target labels based on future return thresholds
    data['Target'] = np.where(data['Future_Return'] > threshold_buy, 2,  # Buy signal
                              np.where(data['Future_Return'] < -threshold_sell, 1,  # Sell signal
                                       0))  # Hold signal

    # Drop the temporary 'Future_Return' column
    data.drop('Future_Return', axis=1, inplace=True)

    # Remove rows with NaN values caused by shift operation
    return data.dropna()


def read_pandas_ticker(stock:str,period='1y'):
   """
    Valid options for period are 1d, 5d, 1mo, 3mo, 6mo, 1y,
     2y, 5y, 10y and ytd.
   """
   history = yf.Ticker(ticker=stock).history(period=period)
   return history

def add_BB(data,window=21,k=2):
    data[f'BB_{window}_MID'] = data['Close'].rolling(window=window).mean()
    data[f'BB_{window}_UP'] = data[f'BB_{window}_MID'] + k*data['Close'].rolling(window=window).std()
    data[f'BB_{window}_LO'] = data[f'BB_{window}_MID'] - k*data['Close'].rolling(window=window).std()
    return data.dropna()

def save_to_file(df,file_name):
    data_dir=os.getenv("DATA_DIR")
    output_name = f"{data_dir}/{file_name}"
    df.to_csv(output_name)


# Function to fetch and save historical data for multiple tickers
def fetch_and_save_ticker_data(ticker_file="tickers.txt", output_directory="historical_data", period="1y"):
    """
    Reads tickers from a file and fetches their historical data,
    saving each one as a CSV file in the specified directory.
    """
    # Load tickers from file
    with open(ticker_file, "r") as f:
        tickers = [line.strip() for line in f.readlines()]

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    output_string=""
    # Fetch data for each ticker and save as CSV
    for ticker in tickers:
        try:
            df = read_pandas_ticker(ticker, period)
            if not df.empty:
                file_path = os.path.join(output_directory, f"{ticker}.csv")
                df.to_csv(file_path)
                res = f"Saved {ticker} data to {file_path} \r\n"
                output_string +=res
                print(res)
            else:
                print(f"No data for {ticker}")
        except Exception as e:
            res=f"Error retrieving data for {ticker}: {e} \r\n"
            output_string += res
            print(res)

        # Delay to prevent hitting API rate limits
        time.sleep(0.2)

    print("Data fetching complete.")
    return output_string

def add_total_signal(df,total_signal_function):
    """
    Adds the 'TotalSignal' column to the DataFrame without using progress_apply.
    """
    if 'TotalSignal' not in df.columns:
        df['TotalSignal'] = 0  # Default value if missing

    df['TotalSignal'] = df.apply(lambda row: total_signal_function(df, row.name), axis=1)
    return df
# Function to read tickers from the file
def read_tickers_from_file(filepath="../../data/tickers.txt"):
    try:
        with open(filepath, "r") as f:
            tickers = [line.strip() for line in f]
        return tickers
    except FileNotFoundError:
        return ["Tickers file not found."]
    except Exception as e:
        return [f"Error reading file: {str(e)}"]
# Example usage
if __name__ == "__main__":
    fetch_and_save_ticker_data("../../data/tickers.txt","../../data/","3y")


