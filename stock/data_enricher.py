import pandas as pd
import pandas_ta as ta

def add_rsi_column(df, length=14):
    # Calculate RSI and add it as a new column to the DataFrame
    df['RSI'] = ta.rsi(df['Close'], length=length)
    return df


def add_macd_columns(df, macd_fast=12, macd_slow=26, macd_signal=9):
    # Calculate MACD and add it as new columns to the DataFrame
    macd = ta.macd(df['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    # MACD returns a DataFrame with three columns: MACD, Signal, and Histogram
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    df['MACD_Histogram'] = macd['MACDh_12_26_9']

    return df

def add_bb_columns(df, bb_length=20, bb_std=2):
    # Calculate Bollinger Bands and add them as new columns to the DataFrame
    bb = ta.bbands(df['Close'], length=bb_length, std=bb_std)

    # Bollinger Bands: Upper, Middle, and Lower bands
    df['BB_Upper'] = bb['BBU_20_2.0']  # Upper Bollinger Band
    df['BB_Middle'] = bb['BBM_20_2.0']  # Middle Bollinger Band (Moving Average)
    df['BB_Lower'] = bb['BBL_20_2.0']  # Lower Bollinger Band

    return df


def get_rsi(close, lookback):
    ret = close.diff()
    up = []
    down = []
    for i in range(len(ret)):
        if ret[i] < 0:
            up.append(0)
            down.append(ret[i])
        else:
            up.append(ret[i])
            down.append(0)
    up_series = pd.Series(up)
    down_series = pd.Series(down).abs()
    up_ewm = up_series.ewm(com = lookback - 1, adjust = False).mean()
    down_ewm = down_series.ewm(com = lookback - 1, adjust = False).mean()
    rs = up_ewm/down_ewm
    rsi = 100 - (100 / (1 + rs))
    rsi_df = pd.DataFrame(rsi).rename(columns = {0:'rsi'}).set_index(close.index)
    rsi_df = rsi_df.dropna()
    return rsi_df[3:]

def read_df(ticker_path):
    df = pd.read_csv(ticker_path, parse_dates=["Date"], index_col="Date")
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")
    return df

def add_indicators(df, sma_period=20, bb_period=20, macd_short=12, macd_long=26, macd_signal=9, rsi_period=14):
    """
    Adds Bollinger Bands, MACD, RSI, and SMA to a DataFrame with stock prices.

    Parameters:
        df (pd.DataFrame): DataFrame with a 'Close' column.
        sma_period (int): Period for Simple Moving Average.
        bb_period (int): Period for Bollinger Bands.
        macd_short (int): Short EMA period for MACD.
        macd_long (int): Long EMA period for MACD.
        macd_signal (int): Signal line period for MACD.
        rsi_period (int): Period for Relative Strength Index.

    Returns:
        pd.DataFrame: DataFrame with additional indicator columns.
    """


    # Simple Moving Average (SMA)
    df[f"SMA_{sma_period}"] = df['Close'].rolling(window=sma_period).mean()

    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(window=bb_period).mean()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['Close'].rolling(window=bb_period).std()
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['Close'].rolling(window=bb_period).std()

    # MACD (Moving Average Convergence Divergence)
    short_ema = df['Close'].ewm(span=macd_short, adjust=False).mean()
    long_ema = df['Close'].ewm(span=macd_long, adjust=False).mean()
    df['MACD'] = short_ema - long_ema
    df['MACD_Signal'] = df['MACD'].ewm(span=macd_signal, adjust=False).mean()

    df[f'rsi_{rsi_period}'] = get_rsi(df['Close'], rsi_period)

    print(df)
    return df.dropna()  # Remove NaN values at the start

def add_rsi_macd_bb(df):
    df = add_rsi_column(df)
    df = add_bb_columns(df)
    df = add_macd_columns(df).dropna()
    return df

# Example usage
if __name__ == "__main__":
    # Sample data
    #df_with_indicators = add_indicators("../../data/AAPL.csv")
    df = pd.read_csv("../../data/AAPL.csv", parse_dates=["Date"], index_col="Date")
    df=add_rsi_column(df)
    df=add_bb_columns(df)
    df=add_macd_columns(df).dropna()

    print(df.head())  # Show the first few rows


