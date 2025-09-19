import os
import numpy as np
import time
import pandas as pd
from stock.my_yfinance import MyYFinance

def calculate_performance_index(df: pd.DataFrame,
                                weight_return=0.4,
                                weight_win_rate=0.3,
                                weight_trades=0.1,
                                weight_bh=0.2) -> pd.DataFrame:
    """
    Calculate a normalized PerformanceIndex (0-100) for each strategy in the DataFrame.

    Parameters :
    - df: DataFrame with columns ['Return [%]', 'Win Rate [%]', '# Trades', 'Buy & Hold [%]']
    - weight_return: weight for Return [%]
    - weight_win_rate: weight for Win Rate [%]
    - weight_trades: weight for # Trades
    - weight_bh: weight for Buy & Hold Return [%]

    Returns:
    - DataFrame with an added 'PerformanceIndex' column scaled 0-100.
    """

    df = df.copy()
    cols = ['Return [%]', 'Win Rate [%]', '# Trades', 'Buy & Hold Return [%]']

    # Normalize each metric to 0-1 scale (min-max)
    for col in cols:
        min_val = df[col].min()
        max_val = df[col].max()
        range_val = max_val - min_val
        if range_val == 0:
            # Avoid division by zero if all values identical
            df[f'norm_{col}'] = 0.0
        else:
            df[f'norm_{col}'] = (df[col] - min_val) / range_val

    # Weighted sum of normalized features
    df['PerformanceIndex_raw'] = (
            weight_return * df['norm_Return [%]'] +
            weight_win_rate * df['norm_Win Rate [%]'] +
            weight_trades * df['norm_# Trades'] +
            weight_bh * df['norm_Buy & Hold Return [%]']
    )

    # Normalize final score 0-100
    min_pi = df['PerformanceIndex_raw'].min()
    max_pi = df['PerformanceIndex_raw'].max()
    range_pi = max_pi - min_pi
    if range_pi == 0:
        df['BobIndex'] = 100.0  # all equal → max score
    else:
        df['BobIndex'] = 100 * (df['PerformanceIndex_raw'] - min_pi) / range_pi
    # Round to 2 decimal places
    df['BobIndex'] = df['BobIndex'].round(2)
    # Cleanup intermediate columns
    df.drop(columns=[f'norm_{col}' for col in cols] + ['PerformanceIndex_raw'], inplace=True)

    return df

def get_ticker_from_file_path(file_path):
    filename = os.path.basename(file_path)  # 'ticker.csv'
    ticker = os.path.splitext(filename)[0]
    return ticker


def get_data_path_from_file_path(file_path: str) -> str:
    """
    Given a file path, returns the directory path.

    Example:
        '../data/AAPL.csv' -> '../data/'
    """
    dir_path = os.path.dirname(file_path)
    # Assicuriamoci che finisca con '/'
    if not dir_path.endswith("/"):
        dir_path += "/"
    return dir_path

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
   #history = yf.Ticker(ticker=stock).history(period=period)
   history = MyYFinance.fetch_by_period(stock,period=period)
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

def add_total_signal_old(df,total_signal_function):
    """
    Adds the 'TotalSignal' column to the DataFrame without using progress_apply.
    """
    if 'TotalSignal' not in df.columns:
        df.loc[:, 'TotalSignal'] = 0  # Default value if missing

    df.loc[:, 'TotalSignal'] = df.apply(lambda row: total_signal_function(df, row.name), axis=1)
    return df

def add_total_signal(df, total_signal_function):
    if 'TotalSignal' not in df.columns:
        df.loc[:, 'TotalSignal'] = 0
    df.loc[:, 'TotalSignal'] = [total_signal_function(df, idx) for idx in df.index]
    return df

def add_total_signal_vec(df, total_signal_function):
    if 'TotalSignal' not in df.columns:
        df.loc[:, 'TotalSignal'] = 0
    """
    Adds the 'TotalSignal' column to the DataFrame by applying a vectorized
    signal function.
    """
    # Simply call the vectorized signal function and assign its result directly to the column.
    # The function itself handles all the calculations for the entire DataFrame.
    df.loc[:, 'TotalSignal'] = total_signal_function(df)

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


import pandas as pd


def count_tickers_in_best_matrix(matrix_data_path="../../data/best_matrix.csv",file_tickers_path="../../data/tickers.txt"):
    """
    Counts the frequency of each ticker from the file in the best_matrix.csv.

    Returns:
        A pandas DataFrame with 'Ticker' and 'Count' columns.
    """

    # Read tickers from your existing function and convert to a set
    tickers_from_file = set(read_tickers_from_file(file_tickers_path))

    # Read the best_matrix.csv file
    try:
        best_matrix_df = pd.read_csv(matrix_data_path)
    except FileNotFoundError:
        print("Error: The file best_matrix.csv was not found.")
        return pd.DataFrame({'Ticker': [], 'Count': []})

    # Filter the matrix to exclude rows where the 'Strategy' is 'NO_STRATEGY'
    valid_strategies_df = best_matrix_df[best_matrix_df['strategy'] != 'NO_STRATEGY']

    # Identify tickers that have a valid strategy in the matrix
    tickers_in_matrix_with_strategy = set(valid_strategies_df['Ticker'].unique())

    # Find tickers that are in the file but NOT in the matrix with a valid strategy
    tickers_not_in_matrix = tickers_from_file.difference(tickers_in_matrix_with_strategy)

    # Create DataFrames for the two groups
    # 1. Tickers with valid strategies
    ticker_counts = valid_strategies_df['Ticker'].value_counts().reset_index()
    ticker_counts.columns = ['Ticker', 'Count']

    # 2. Tickers without a valid strategy
    tickers_not_present_df = pd.DataFrame({
        'Ticker': sorted(list(tickers_not_in_matrix)),
        'Count': 0  # Assign a count of 0 for these tickers
    })

    # Combine the two DataFrames
    final_df = pd.concat([ticker_counts, tickers_not_present_df], ignore_index=True)

    # Sort the results by Ticker for a clean output
    final_df = final_df.sort_values(by='Count', ascending=False).reset_index(drop=True)

    return final_df

# Example usage
if __name__ == "__main__":
    # https://github.com/ranaroussi/yfinance/issues/2422
    #fetch_and_save_ticker_data("../../data/tickers.txt","../../data/","3y")
    #print(read_pandas_ticker("MB.MI"))
    print(count_tickers_in_best_matrix())

