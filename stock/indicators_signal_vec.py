
import pandas as pd


def adx_sma_tf_20_50_25(df) -> pd.Series:
    return adx_sma_trend_following(df, 20, 50, 25)


def adx_sma_tf_10_30_25(df) -> pd.Series:
    return adx_sma_trend_following(df, 10, 30, 25)


def adx_sma_tf_15_40_35(df) -> pd.Series:
    return adx_sma_trend_following(df, 15, 40, 35)


def adx_sma_tf_10_30_35(df) -> pd.Series:
    return adx_sma_trend_following(df, 10, 30, 25)


def adx_sma_trend_following(df, short_window: int = 20, long_window: int = 50, adx_threshold: float = 25) -> pd.Series:
    """
    Generates trading signals based on ADX + SMA Trend Following strategy.

    Buy (2): if short SMA > long SMA and ADX > adx_threshold
    Sell (1): if short SMA < long SMA and ADX > adx_threshold
    Hold (0): otherwise

    Args:
        df (pd.DataFrame): Must contain columns ['Close', 'ADX']
        short_window (int): Lookback for short SMA (default 20)
        long_window (int): Lookback for long SMA (default 50)
        adx_threshold (float): Minimum ADX to confirm trend (default 25)

    Returns:
        pd.Series: Trading signals (2=Buy, 1=Sell, 0=Hold)
    """

    # Calculate moving averages
    sma_short = df['Close'].rolling(window=short_window, min_periods=1).mean()
    sma_long = df['Close'].rolling(window=long_window, min_periods=1).mean()

    # Conditions for buy/sell
    buy_condition = (sma_short > sma_long) & (df['ADX_14'] > adx_threshold)
    sell_condition = (sma_short < sma_long) & (df['ADX_14'] > adx_threshold)

    # Initialize signals
    signals = pd.Series(0, index=df.index, dtype='int8')
    signals[buy_condition] = 2
    signals[sell_condition] = 1

    return signals


def adx_trend_breakout_20_25(df) -> pd.Series:
    return adx_trend_breakout(df, 20, 25)


def adx_trend_breakout_10_40(df) -> pd.Series:
    return adx_trend_breakout(df, 10, 40)


def adx_trend_breakout_10_35(df) -> pd.Series:
    return adx_trend_breakout(df, 10, 35)


def adx_trend_breakout(df, lookback: int = 20, adx_threshold: float = 25) -> pd.Series:
    """
    Generates trading signals based on ADX Trend Filter + Breakout strategy.

    Buy (2): if Close > highest high of last `lookback` days and ADX > `adx_threshold`
    Sell (1): if Close < lowest low of last `lookback` days and ADX > `adx_threshold`
    Hold (0): otherwise

    Args:
        df (pd.DataFrame): Must contain columns ['High', 'Low', 'Close', 'ADX']
        lookback (int): Period for breakout levels (default 20)
        adx_threshold (float): Minimum ADX to confirm trend (default 25)

    Returns:
        pd.Series: Trading signals (2=Buy, 1=Sell, 0=Hold)
    """

    # Calculate rolling breakout levels
    highest_high = df['High'].rolling(window=lookback, min_periods=1).max().shift(1)
    lowest_low = df['Low'].rolling(window=lookback, min_periods=1).min().shift(1)

    # Conditions for buy/sell
    buy_condition = (df['Close'] > highest_high) & (df['ADX_14'] > adx_threshold)
    sell_condition = (df['Close'] < lowest_low) & (df['ADX_14'] > adx_threshold)

    # Initialize signals
    signals = pd.Series(0, index=df.index, dtype='int8')
    signals[buy_condition] = 2
    signals[sell_condition] = 1

    return signals

def donchian_breakout_with_ma_20_50(df):
    return donchian_breakout_with_ma_filter(df, lookback=20, ma_window=50)

def donchian_breakout_with_ma_10_30(df):
    return donchian_breakout_with_ma_filter(df, lookback=10, ma_window=30)

def donchian_breakout_with_ma_15_40(df):
    return donchian_breakout_with_ma_filter(df, lookback=10, ma_window=30)

def donchian_breakout_with_ma_filter(df, lookback=20, ma_window=50):
    """
    Generates trading signals based on a Donchian Channel breakout
    with a long-term moving average trend filter.

    Logic:
        - Buy (2) when Close > Donchian high AND Close > long MA
        - Sell (1) when Close < Donchian low AND Close < long MA
        - Hold (0) otherwise

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex and 'Close' column
        lookback (int): Lookback window for Donchian channel (default 20)
        ma_window (int): Window for long moving average (default 50)

    Returns:
        pd.Series: Trading signals (2 = Buy, 1 = Sell, 0 = Hold)
    """
    # Donchian Channel
    rolling_high = df["Close"].rolling(lookback, min_periods=1).max().shift(1)
    rolling_low = df["Close"].rolling(lookback, min_periods=1).min().shift(1)

    # Long-term moving average
    long_ma = df["Close"].rolling(ma_window, min_periods=1).mean()

    # Conditions
    buy_condition = (df["Close"] > rolling_high) & (df["Close"] > long_ma)
    sell_condition = (df["Close"] < rolling_low) & (df["Close"] < long_ma)

    # Signals
    signals = pd.Series(0, index=df.index, dtype="int8")
    signals[buy_condition] = 2
    signals[sell_condition] = 1

    return signals


def keltner_channel_vectorized(df: pd.DataFrame, lookback_periods: int = 20, atr_multiplier: float = 2.0) -> pd.Series:
    """
    Generates trading signals based on the Keltner Channel breakout strategy.

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex and 'High', 'Low', 'Close' columns.
        lookback_periods (int): The number of periods to use for the moving average and ATR.
        atr_multiplier (float): The multiplier for the ATR to set the channel width.

    Returns:
        pd.Series: Trading signals (2 = Long, 1 = Short, 0 = No position).
    """
    # Calculate True Range (TR)
    # TR is the greatest of:
    # 1. Current High minus Current Low
    # 2. Absolute value of Current High minus Previous Close
    # 3. Absolute value of Current Low minus Previous Close
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Calculate Average True Range (ATR)
    # The ATR is the rolling mean of the True Range.
    atr = true_range.rolling(lookback_periods).mean()

    # Calculate the moving average (center line of the channel)
    center_line = df['Close'].rolling(lookback_periods).mean()

    # Calculate the upper and lower Keltner Channels
    upper_channel = center_line + (atr * atr_multiplier)
    lower_channel = center_line - (atr * atr_multiplier)

    # Initialize a signals Series with a default value of 0 (no position)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Generate long signals (2): if the price breaks above the upper channel
    long_condition = df['Close'] > upper_channel
    signals[long_condition] = 2

    # Generate short signals (1): if the price breaks below the lower channel
    short_condition = df['Close'] < lower_channel
    signals[short_condition] = 1

    return signals


def donchian_channel_vectorized(df: pd.DataFrame, lookback_periods: int = 20) -> pd.Series:
    """
    Generates trading signals based on the Donchian Channel breakout strategy.

    Logic:
        - A long signal (2) is generated when the price breaks above the N-period high.
        - A short signal (1) is generated when the price breaks below the N-period low.

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex and at least 'High', 'Low', 'Close' columns.
        lookback_periods (int): Number of periods (e.g., days) to consider for the high/low calculation.

    Returns:
        pd.Series: Trading signals (2 = Long, 1 = Short, 0 = No position).
    """
    # Calculate the upper and lower Donchian Channels
    # The upper channel is the highest high over the lookback period
    df['upper_channel'] = df['High'].rolling(lookback_periods).max().shift(1)

    # The lower channel is the lowest low over the lookback period
    df['lower_channel'] = df['Low'].rolling(lookback_periods).min().shift(1)

    # Initialize a signals Series with a default value of 0 (no position)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Generate long signals: if the current closing price breaks above the upper channel
    long_condition = df['Close'] > df['upper_channel']
    signals[long_condition] = 2

    # Generate short signals: if the current closing price breaks below the lower channel
    short_condition = df['Close'] < df['lower_channel']
    signals[short_condition] = 1

    return signals


def weekly_breakout_vectorized(df, lookback_weeks=1):
    """
    Generates trading signals based on a weekly breakout strategy.

    Logic:
        - Long signal (2) on Tuesday if Monday's close > high of previous N weeks
        - Short signal (1) on Tuesday if Monday's close < low of previous N weeks
        - Hold (0) otherwise

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex and at least 'Close' column
        lookback_weeks (int): Number of weeks to consider for high/low (default 1)

    Returns:
        pd.Series: Trading signals (2 = Buy, 1 = Short, 0 = Hold)
    """
    # Use "Close" as the fixed price column
    price_col = "Close"

    # Compute weekly highs and lows
    weekly = df.groupby(df.index.to_period("W"))[price_col].agg(["max", "min"])
    weekly["high"] = weekly["max"].rolling(lookback_weeks).max().shift(1)
    weekly["low"] = weekly["min"].rolling(lookback_weeks).min().shift(1)

    # Map previous highs/lows back to daily index
    prev_high = df.index.to_period("W").map(weekly["high"])
    prev_low = df.index.to_period("W").map(weekly["low"])

    # Day of the week (0=Mon, 1=Tue, ..., 4=Fri)
    dow = df.index.dayofweek

    # Entry conditions
    long_condition = (dow == 0) & (df[price_col] > prev_high)
    short_condition = (dow == 0) & (df[price_col] < prev_low)

    # Initialize signals
    signals = pd.Series(0, index=df.index, dtype="int8")
    signals[long_condition] = 2
    signals[short_condition] = 1

    return signals


def hammers(df):
    df['body'] = abs(df['Close'] - df['Open'])
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)

    hammer_condition_ = (df['lower_wick'] > 2 * df['body']) & \
                       (df['upper_wick'] < 0.3 * df['body'])
    return  hammer_condition_

def inverted_hammers(df):
    # Vectorized calculation of candle body and wicks
    df['body'] = abs(df['Close'] - df['Open'])
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']

    # --- Inverted Hammer conditions ---
    inverted_hammer_condition_ = (df['upper_wick'] > 2 * df['body']) & \
                                (df['lower_wick'] < 0.3 * df['body'])
    return inverted_hammer_condition_

def bollinger_bands_mean_reversion_sma_vectorized(df):
    """
    Generates trading signals for the entire DataFrame based on the
    Bollinger Bands Mean Reversion strategy with SMA confirmation.

    Args:
        df (pd.DataFrame): DataFrame with OHLC and pre-calculated indicators
                          (Close, BB_Lower, BB_Upper, SMA_short, SMA_long).

    Returns:
        pd.Series: A Series of trading signals (2 for Buy, 1 for Sell, 0 for Hold).
    """

    # Create boolean masks for entry conditions
    # This replaces the if/elif logic for each row
    buy_condition = (df['Close'] <= df['BB_Lower']) & \
                    (df['SMA_short'] > df['SMA_long'])

    sell_condition = (df['Close'] >= df['BB_Upper']) & \
                     (df['SMA_short'] < df['SMA_long'])

    # Initialize a signal Series with a default value of 0 (Hold)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Apply signals using vectorized assignment
    signals[buy_condition] = 2
    signals[sell_condition] = 1

    return signals

def sma_stoch_close_strategy_vectorized(df):
    """
    Generates less selective signals for the entire DataFrame based on SMA
    and Stochastic Oscillator, using a vectorized approach.

    Args:
        df (pd.DataFrame): DataFrame with OHLC and pre-calculated indicators
                          (Close, SMA_short, SMA_long, STOCHk_14_3_3, STOCHd_14_3_3, BB_Middle).

    Returns:
        pd.Series: A Series of trading signals (2 for Buy, 1 for Sell, 0 for Hold).
    """

    # Create boolean masks for buy conditions
    buy_conditions = (df['SMA_short'] > df['SMA_long']) & \
                     (df['STOCHk_14_3_3'] > df['STOCHd_14_3_3']) & \
                     (df['STOCHk_14_3_3'] < 35) & \
                     (df['Close'] > df['BB_Middle'])

    # Create boolean masks for sell conditions
    sell_conditions = (df['SMA_short'] < df['SMA_long']) & \
                      (df['STOCHk_14_3_3'] < df['STOCHd_14_3_3']) & \
                      (df['STOCHk_14_3_3'] > 65) & \
                      (df['Close'] < df['BB_Middle'])

    # Initialize a signal Series with a default value of 0 (Hold)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Apply signals using vectorized assignment
    signals[buy_conditions] = 2
    signals[sell_conditions] = 1

    return signals

def stochastic_oscillator_signal_vectorized(df):
    """
    Generates trading signals for the entire DataFrame based on the Stochastic Oscillator
    using a vectorized approach.

    Args:
        df (pandas.DataFrame): DataFrame containing financial data with 'STOCHk_14_3_3'
                                and 'STOCHd_14_3_3' columns.

    Returns:
        pandas.Series: A Series with trading signals (2 = Buy, 1 = Sell, 0 = Hold,
                       -1 = Exit Short, -2 = Exit Long).
    """

    # Create temporary columns for previous values using .shift()
    df['STOCHk_prev'] = df['STOCHk_14_3_3'].shift(1)
    df['STOCHd_prev'] = df['STOCHd_14_3_3'].shift(1)

    # Calculate boolean masks for all conditions in a single, vectorized operation
    # This replaces the if/elif blocks and .iloc calls.

    # Buy (Long Entry) Condition: %K crosses above %D in the oversold zone.
    buy_condition = (df['STOCHk_14_3_3'] > df['STOCHd_14_3_3']) & \
                    (df['STOCHk_prev'] <= df['STOCHd_prev']) & \
                    (df['STOCHk_14_3_3'] < 30)

    # Sell (Short Entry) Condition: %K crosses below %D in the overbought zone.
    sell_condition = (df['STOCHk_14_3_3'] < df['STOCHd_14_3_3']) & \
                     (df['STOCHk_prev'] >= df['STOCHd_prev']) & \
                     (df['STOCHk_14_3_3'] > 70)

    # Exit Long Condition: %K crosses below %D.
    exit_long_condition = (df['STOCHk_14_3_3'] < df['STOCHd_14_3_3']) & \
                          (df['STOCHk_prev'] >= df['STOCHd_prev'])

    # Exit Short Condition: %K crosses above %D.
    exit_short_condition = (df['STOCHk_14_3_3'] > df['STOCHd_14_3_3']) & \
                           (df['STOCHk_prev'] <= df['STOCHd_prev'])

    # Initialize a signal Series with a default value of 0 (Hold)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Apply all signals using vectorized assignment. Note that buy/sell signals
    # are often mutually exclusive in a single candle, but for safety, we assign
    # entries and exits based on your original logic.
    signals[buy_condition] = 2
    signals[sell_condition] = 1
    signals[exit_long_condition] = -2
    signals[exit_short_condition] = -1

    # In a real backtesting scenario, entry and exit logic should be handled separately
    # to avoid overwriting signals. Here's a version that prioritizes entries:
    signals[exit_long_condition] = -2
    signals[exit_short_condition] = -1
    signals[buy_condition] = 2
    signals[sell_condition] = 1

    # Remove temporary columns to keep the DataFrame clean
    df.drop(columns=['STOCHk_prev', 'STOCHd_prev'], inplace=True)

    return signals

def moving_average_crossover_signal_vectorized(df):
    """
    Generates trading signals based on moving average crossover using a vectorized approach.

    Args:
        df (pandas.DataFrame): DataFrame containing financial data with 'SMA_short' and 'SMA_long' columns.

    Returns:
        pandas.Series: A Series with trading signals (2 = Buy, 1 = Sell, 0 = Hold, -1 = Exit Short, -2 = Exit Long).
    """

    # Create temporary columns for the previous values using .shift()
    df['SMA_short_prev'] = df['SMA_short'].shift(1)
    df['SMA_long_prev'] = df['SMA_long'].shift(1)

    # Calculate boolean masks for all conditions in a single, vectorized operation
    # Buy (Long Entry) Condition: Short SMA crosses above Long SMA
    buy_condition = (df['SMA_short'] > df['SMA_long']) & (df['SMA_short_prev'] <= df['SMA_long_prev'])

    # Sell (Short Entry) Condition: Short SMA crosses below Long SMA
    sell_condition = (df['SMA_short'] < df['SMA_long']) & (df['SMA_short_prev'] >= df['SMA_long_prev'])

    # Initialize a signal Series with a default value of 0 (Hold)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Apply entry signals using vectorized assignment.
    # Entry signals are typically prioritized over exit signals.
    signals[buy_condition] = 2
    signals[sell_condition] = 1

    # Apply exit signals. The logic in your original function has a duplicate condition
    # for Buy/Exit Short and Sell/Exit Long, which would cause an overwrite.
    # In a real trading system, entries and exits would be handled separately.
    # Here, we'll assign them based on the opposite crossover, which is a common approach.
    signals[buy_condition] = -1  # Exit Short
    signals[sell_condition] = -2  # Exit Long

    # Correcting the duplicate assignments from the previous step.
    # We assign entries first, and then conditionally override with exits.
    signals[(df['SMA_short'] > df['SMA_long']) & (df['SMA_short_prev'] <= df['SMA_long_prev'])] = 2
    signals[(df['SMA_short'] < df['SMA_long']) & (df['SMA_short_prev'] >= df['SMA_long_prev'])] = 1

    # Remove temporary columns to keep the DataFrame clean
    df.drop(columns=['SMA_short_prev', 'SMA_long_prev'], inplace=True)

    return signals

def rsi_bollinger_macd_total_signal_v5_vectorized(df, tolerance_percent=5):
    """
    Generates trading signals based on RSI, Bollinger Bands, and MACD
    using a vectorized approach.

    Args:
        df (pandas.DataFrame): DataFrame containing OHLC data and indicators
                               ('RSI', 'BB_Lower', 'BB_Middle', 'BB_Upper', 'MACD', 'MACD_Signal').
        tolerance_percent (int): Percentage tolerance for "near" (e.g., 5).

    Returns:
        pandas.Series: A Series of trading signals (2 = Buy, 1 = Sell, 0 = Hold,
                       -1 = Exit Short, -2 = Exit Long).
    """

    # Vettorializzazione del calcolo delle tolleranze
    tolerance = tolerance_percent / 100

    # Buy (Long Entry) Conditions
    rsi_condition = df['RSI'] < 50
    lower_tolerance_min = df['BB_Lower'] * (1 - tolerance)
    lower_tolerance_max = df['BB_Lower'] * (1 + tolerance)
    bollinger_lower_condition = (df['Close'] > lower_tolerance_min) & (df['Close'] < lower_tolerance_max)
    macd_condition = df['MACD'] > df['MACD_Signal']

    # Complete Buy Condition
    buy_condition = rsi_condition & bollinger_lower_condition & macd_condition

    # Sell (Short Entry) Conditions
    rsi_sell_condition = df['RSI'] > 70
    middle_tolerance_min = df['BB_Middle'] * (1 - tolerance)
    middle_tolerance_max = df['BB_Middle'] * (1 + tolerance)
    bollinger_middle_condition = (df['Close'] > middle_tolerance_min) & (df['Close'] < middle_tolerance_max)
    macd_sell_condition = df['MACD'] < df['MACD_Signal']

    # Complete Sell Condition
    sell_condition = rsi_sell_condition & bollinger_middle_condition & macd_sell_condition

    # Exit Long Condition
    exit_long_rsi = df['RSI'] > 70
    exit_long_bollinger = df['Close'] >= df['BB_Upper']
    exit_long_condition = exit_long_rsi | exit_long_bollinger

    # Exit Short Condition
    exit_short_rsi = df['RSI'] < 50
    exit_short_bollinger = df['Close'] <= df['BB_Lower']
    exit_short_condition = exit_short_rsi | exit_short_bollinger

    # Initialize a signal Series with a default value of 0 (Hold)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Apply signals using vectorized assignment, prioritizing entry signals
    # over exit signals if they occur on the same candle.
    signals[buy_condition] = 2
    signals[sell_condition] = 1

    # Apply exit signals
    signals[exit_long_condition] = -2
    signals[exit_short_condition] = -1

    # Re-apply entry signals to ensure they take precedence
    signals[buy_condition] = 2
    signals[sell_condition] = 1

    return signals

def x_bollinger_macd_total_signal_5_vectorized(df):
    return x_rsi_bollinger_macd_total_signal_base_vectorized(df,tolerance_percent=5,up_rsi_bound=70,low_rsi_bound=30,rsi_enabled=False)

def x_bollinger_macd_total_signal_10_vectorized(df):
    return x_rsi_bollinger_macd_total_signal_base_vectorized(df,tolerance_percent=10,up_rsi_bound=70,low_rsi_bound=30,rsi_enabled=False)

def x_rsi_bollinger_total_signal_15_65_35_vectorized(df):
    return x_rsi_bollinger_macd_total_signal_base_vectorized(df,tolerance_percent=15,up_rsi_bound=65,low_rsi_bound=35,macd_enabled=False)

def x_rsi_bollinger_total_signal_10_70_30_vectorized(df):
    return x_rsi_bollinger_macd_total_signal_base_vectorized(df, tolerance_percent=10, up_rsi_bound=70,
                                                             low_rsi_bound=30,macd_enabled=False)


def x_rsi_bollinger_macd_total_signal_base_vectorized(df, tolerance_percent=5,up_rsi_bound=70,low_rsi_bound=30,macd_enabled=True,rsi_enabled=True):

    tolerance = tolerance_percent / 100

    # Buy (Long Entry) Conditions
    rsi_condition= None
    if rsi_enabled:
        rsi_condition = df['RSI'] < low_rsi_bound
    else:
        rsi_condition = df['RSI'] < 1000
    lower_tolerance_min = df['BB_Lower'] * (1 - tolerance)
    lower_tolerance_max = df['BB_Lower'] * (1 + tolerance)
    bollinger_lower_condition = (df['Close'] > lower_tolerance_min) & (df['Close'] < lower_tolerance_max)

    macd_condition= None
    if macd_enabled:
        macd_condition = df['MACD'] > df['MACD_Signal']
    else:
        macd_condition = df['MACD'] < 10000
    # Complete Buy Condition
    buy_condition = rsi_condition & bollinger_lower_condition & macd_condition

    # Sell (Short Entry) Conditions
    rsi_sell_condition= None
    if rsi_enabled:
        rsi_sell_condition = df['RSI'] > up_rsi_bound
    else:
        rsi_sell_condition = df['RSI'] > 1000

    upper_tolerance_min = df['BB_Upper'] * (1 - tolerance)
    upper_tolerance_max = df['BB_Upper'] * (1 + tolerance)
    bollinger_upper_condition = (df['Close'] > upper_tolerance_min) & (df['Close'] < upper_tolerance_max)
    macd_sell_condition= None
    if macd_enabled:
        macd_sell_condition = df['MACD'] < df['MACD_Signal']
    else:
        macd_sell_condition = df['MACD'] <10000
    # Complete Sell Condition
    sell_condition = rsi_sell_condition & bollinger_upper_condition & macd_sell_condition

    # Exit Long Condition
    exit_long_rsi = df['RSI'] > up_rsi_bound
    exit_long_bollinger = df['Close'] >= df['BB_Upper']
    exit_long_condition = exit_long_rsi | exit_long_bollinger

    # Exit Short Condition
    exit_short_rsi = df['RSI'] < low_rsi_bound
    exit_short_bollinger = df['Close'] <= df['BB_Lower']
    exit_short_condition = exit_short_rsi | exit_short_bollinger

    # Initialize a signal Series with a default value of 0 (Hold)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Apply signals using vectorized assignment, prioritizing entry signals
    # over exit signals if they occur on the same candle.
    signals[buy_condition] = 2
    signals[sell_condition] = 1

    # Apply exit signals
    signals[exit_long_condition] = -2
    signals[exit_short_condition] = -1

    # Re-apply entry signals to ensure they take precedence
    signals[buy_condition] = 2
    signals[sell_condition] = 1

    return signals

def rsi_hammer_70_30_vectorized(df):
    return rsi_hammer_vectorized(df,70,30)
def rsi_hammer_65_35_vectorized(df):
    return rsi_hammer_vectorized(df,65,35)
def rsi_hammer_80_20_vectorized(df):
    return rsi_hammer_vectorized(df,80,20)

def rsi_hammer_vectorized(df, up_rsi_bound=70,low_rsi_bound=30):
    hammer_cond = hammers(df) | inverted_hammers(df)
    rsi_buy_condition = df['RSI'] < low_rsi_bound
    buy_condition = rsi_buy_condition & hammer_cond

    rsi_sell_condition = df['RSI'] > up_rsi_bound
    # Complete Sell Condition
    sell_condition = rsi_sell_condition & hammer_cond

    # Initialize a signal Series with a default value of 0 (Hold)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Re-apply entry signals to ensure they take precedence
    signals[buy_condition] = 2
    signals[sell_condition] = 1

    return signals

def rsi_bollinger_macd_total_signal_v1_vectorized(df):
    """
    Generates trading signals based on RSI, Bollinger Bands, and MACD
    using a vectorized approach.

    Args:
        df (pandas.DataFrame): DataFrame containing OHLC data and indicators
                               ('RSI', 'BB_Lower', 'BB_Middle', 'BB_Upper', 'MACD', 'MACD_Signal').

    Returns:
        pandas.Series: A Series of trading signals (2 = Buy, 1 = Sell, 0 = Hold,
                       -1 = Exit Short, -2 = Exit Long).
    """

    # Buy (Long Entry) Conditions
    # Corrisponde a c1, c2, c3 nel codice originale.
    buy_condition = (df['RSI'] < 50) & (df['Close'] >= df['BB_Lower']) & (df['MACD'] > df['MACD_Signal'])

    # Sell (Short Entry) Conditions
    # Corrisponde a c4, c5, c6 nel codice originale.
    sell_condition = (df['RSI'] > 70) & (df['Close'] <= df['BB_Middle']) & (df['MACD'] < df['MACD_Signal'])

    # Exit Long Condition
    # Corrisponde a c7 e c8 nel codice originale.
    exit_long_condition = (df['RSI'] > 70) & (df['Close'] >= df['BB_Upper'])

    # Exit Short Condition
    # Corrisponde a c9 e c10 nel codice originale.
    exit_short_condition = (df['RSI'] < 50) & (df['Close'] <= df['BB_Lower'])

    # Inizializza una Series di segnali con un valore predefinito di 0 (Hold)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Applica i segnali tramite assegnazione vettoriale.
    # L'ordine è importante per la logica. Gli ingressi (entry) devono avere la precedenza
    # sulle uscite (exit) se le condizioni si sovrappongono.
    signals[buy_condition] = 2
    signals[sell_condition] = 1

    signals[exit_long_condition] = -2
    signals[exit_short_condition] = -1

    # Per garantire che i segnali di ingresso non vengano sovrascritti, li riapplichiamo.
    signals[buy_condition] = 2
    signals[sell_condition] = 1

    return signals

def rsi_bollinger_macd_total_signal_v2_vectorized(df):
    """
    Generates trading signals based on RSI, Bollinger Bands, and MACD
    using a vectorized approach.

    Args:
        df (pandas.DataFrame): DataFrame containing OHLC data and indicators
                               ('RSI', 'BB_Lower', 'BB_Middle', 'BB_Upper', 'MACD', 'MACD_Signal').

    Returns:
        pandas.Series: A Series of trading signals (2 = Buy, 1 = Sell, 0 = Hold,
                       -1 = Exit Short, -2 = Exit Long).
    """

    # **Buy (Long Entry) Conditions**
    # These replace c1, c2, and c3 from the original code.
    buy_condition = (df['RSI'] < 50) & (df['Close'] >= df['BB_Lower']) & (df['MACD'] > df['MACD_Signal'])

    # **Sell (Short Entry) Conditions**
    # These replace c4, c5, and c6.
    sell_condition = (df['RSI'] > 70) & (df['Close'] <= df['BB_Upper']) & (df['MACD'] < df['MACD_Signal'])

    # **Exit Long Conditions**
    # These replace c7 and c8.
    exit_long_condition = (df['RSI'] > 70) | (df['Close'] >= df['BB_Upper'])

    # **Exit Short Conditions**
    # These replace c9 and c10.
    exit_short_condition = (df['RSI'] < 50) | (df['Close'] <= df['BB_Lower'])

    # Initialize a signal Series with a default value of 0 (Hold)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Apply signals using vectorized assignment. The order is important
    # to handle signal priority, with entries typically taking precedence
    # over exits.
    signals[exit_long_condition] = -2
    signals[exit_short_condition] = -1
    signals[buy_condition] = 2
    signals[sell_condition] = 1
    return signals

def mean_reversion_signal_v1_vectorized(df,  tolerance_percent=5):
    return mean_reversion_signal_generic_vectorized(df, tolerance_percent)

def mean_reversion_signal_v2_vectorized(df,  tolerance_percent=10):
    return mean_reversion_signal_generic_vectorized(df, tolerance_percent)

def mean_reversion_signal_v3_vectorized(df, tolerance_percent=15):
    return mean_reversion_signal_generic_vectorized(df, tolerance_percent)

def mean_reversion_signal_generic_vectorized(df, tolerance_percent=5):
    """
    Generates mean reversion trading signals based on RSI and Bollinger Bands,
    using a vectorized approach.

    Args:
        df (pandas.DataFrame): DataFrame containing OHLC and indicator data
                               ('RSI', 'BB_Lower', 'BB_Upper').
        tolerance_percent (int): Percentage tolerance for "near" (e.g., 5).

    Returns:
        pandas.Series: A Series of trading signals (2 = Buy, 1 = Sell, 0 = Hold,
                       -1 = Exit Short, -2 = Exit Long).
    """

    # Vettorializzazione del calcolo delle tolleranze
    tolerance = tolerance_percent / 100

    # Buy (Long Entry) Conditions
    rsi_long_condition = df['RSI'] < 30
    lower_tolerance_min = df['BB_Lower'] * (1 - tolerance)
    lower_tolerance_max = df['BB_Lower'] * (1 + tolerance)
    bollinger_lower_condition = (df['Close'] > lower_tolerance_min) & (df['Close'] < lower_tolerance_max)
    buy_condition = bollinger_lower_condition & rsi_long_condition

    # Sell (Short Entry) Conditions
    rsi_short_condition = df['RSI'] > 70
    upper_tolerance_min = df['BB_Upper'] * (1 - tolerance)
    upper_tolerance_max = df['BB_Upper'] * (1 + tolerance)
    bollinger_upper_condition = (df['Close'] > upper_tolerance_min) & (df['Close'] < upper_tolerance_max)
    sell_condition = bollinger_upper_condition & rsi_short_condition

    # Exit Long Conditions
    exit_long_rsi = df['RSI'] > 50
    exit_long_bollinger = df['Close'] >= df['BB_Upper']
    exit_long_condition = exit_long_rsi | exit_long_bollinger

    # Exit Short Conditions
    exit_short_rsi = df['RSI'] < 50
    exit_short_bollinger = df['Close'] <= df['BB_Lower']
    exit_short_condition = exit_short_rsi | exit_short_bollinger

    # Initialize a signal Series with a default value of 0 (Hold)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Apply all signals using vectorized assignment
    # The order is important to handle signal precedence.
    signals[exit_long_condition] = -2
    signals[exit_short_condition] = -1
    signals[buy_condition] = 2
    signals[sell_condition] = 1

    return signals

def rsi_engulfing_signals_vectorized(df):
    """
    Generates trading signals based on Engulfing candlestick patterns and RSI,
    confirmed by overbought/oversold conditions, using a vectorized approach.

    Args:
        df (pandas.DataFrame): DataFrame containing OHLC data and a pre-calculated 'RSI' column.

    Returns:
        pandas.Series: A Series of trading signals (2 = Buy, 1 = Sell Short, 0 = Hold,
                       -1 = Exit Short, -2 = Exit Long).
    """

    # Create temporary columns for previous candle data using .shift()
    df['prev_open'] = df['Open'].shift(1)
    df['prev_close'] = df['Close'].shift(1)
    df['prev_high'] = df['High'].shift(1)
    df['prev_low'] = df['Low'].shift(1)

    # Vettorializza le condizioni di engulfing
    is_bullish_engulfing = (df['Close'] > df['prev_open']) & \
                           (df['Open'] < df['prev_close']) & \
                           (df['Low'] < df['prev_low'])

    is_bearish_engulfing = (df['Open'] > df['prev_close']) & \
                           (df['Close'] < df['prev_open']) & \
                           (df['High'] > df['prev_high'])

    # Vettorializza le condizioni RSI
    rsi_long_condition = df['RSI'] < 30
    rsi_short_condition = df['RSI'] > 70
    exit_rsi_condition = df['RSI'] < 50
    exit_long_rsi_condition = df['RSI'] > 50

    # Inizializza una Series di segnali con un valore predefinito di 0
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Applica i segnali di ingresso usando le maschere booleane
    signals[exit_rsi_condition] = -1
    signals[exit_long_rsi_condition] = -2
    signals[is_bearish_engulfing & rsi_short_condition] = 1
    signals[is_bullish_engulfing & rsi_long_condition] = 2


    # Applica i segnali di uscita



    # Rimuovi le colonne temporanee per pulire il DataFrame
    df.drop(columns=['prev_open', 'prev_close', 'prev_high', 'prev_low'], inplace=True)

    return signals

def t_indicators_combined_signal_vectorized(df):
    """
    Combines signals from various candlestick strategies, excluding itself.

    Args:
        df (pd.DataFrame): DataFrame containing candlestick data.
        current_candle (index): Index of the current candle.


    Returns:
        int: 2 for buy signal, 1 for sell signal, 0 for no signal.
    """
    # Create a new list excluding combined_signal
    # 1. Run each vectorized strategy function and store the signals.
    # We pass a copy of the DataFrame to each function to avoid conflicts
    # if they add temporary columns.
    signal_series_list = [func(df.copy()) for func in indicators_strategy if func != t_indicators_combined_signal_vectorized]

    # 2. Combine signals into a single DataFrame.
    # The .T (transpose) aligns the Series correctly.
    combined_signals_df = pd.DataFrame(signal_series_list).T

    # 3. Count buy and sell signals for each candle.
    # This replaces the individual counts (signals.count(2)) for each candle.
    buy_counts = (combined_signals_df == 2).sum(axis=1)
    sell_counts = (combined_signals_df == 1).sum(axis=1)

    # 4. Initialize the final signal Series.
    # This replaces the default 'return 0' at the end of the original function.
    final_signals = pd.Series(0, index=df.index, dtype='int8')

    # 5. Apply the combination logic using vectorized operations.
    # This replaces the if/elif/else block for each candle.
    # Buy signals take priority if they outnumber sell signals and exceed the threshold.
    final_signals[(buy_counts > sell_counts) & (buy_counts > 2)] = 2

    # Sell signals take priority if they outnumber buy signals and exceed the threshold.
    final_signals[(sell_counts > buy_counts) & (sell_counts > 2)] = 1

    return final_signals

def mixed_signal_strategy_vectorized(df):
    """
    Generates trading signals by combining RSI, Stochastic, and Moving Averages
    using a vectorized approach.

    Args:
        df (pandas.DataFrame): DataFrame with 'Close', 'RSI', 'STOCHk_14_3_3',
                               'STOCHd_14_3_3', 'SMA_short', 'SMA_long',
                               'BB_Lower', and 'BB_Upper' columns.

    Returns:
        pandas.Series: A Series of trading signals (2 = Buy, 1 = Sell, 0 = Hold,
                       -1 = Exit Short, -2 = Exit Long).
    """

    # Conditions to confirm the trend (vectorized)
    bullish_trend = df['SMA_short'] > df['SMA_long']
    bearish_trend = df['SMA_short'] < df['SMA_long']

    # 📈 LONG SIGNAL (2)
    # Enter if the trend is bullish AND momentum indicators are in an oversold zone
    buy_condition = bullish_trend & \
                    ((df['RSI'] < 30) | ((df['STOCHk_14_3_3'] < 20) & (df['STOCHd_14_3_3'] < 20))) & \
                    (df['Close'] < df['BB_Lower'])

    # 📉 SHORT SIGNAL (1)
    # Enter if the trend is bearish AND momentum indicators are in an overbought zone
    sell_condition = bearish_trend & \
                     ((df['RSI'] > 70) | ((df['STOCHk_14_3_3'] > 80) & (df['STOCHd_14_3_3'] > 80))) & \
                     (df['Close'] > df['BB_Upper'])

    # 🚪 EXIT LONG (-2)
    # Exit a long position if the trend reverses or momentum is overbought
    exit_long_condition = bearish_trend | ((df['RSI'] > 70) & (df['Close'] > df['BB_Upper']))

    # 🚪 EXIT SHORT (-1)
    # Exit a short position if the trend reverses or momentum is oversold
    exit_short_condition = bullish_trend | ((df['RSI'] < 30) & (df['Close'] < df['BB_Lower']))

    # Initialize a signal Series with a default value of 0 (Hold)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Apply all signals using vectorized assignment
    # The order is crucial: exits are typically processed first to
    # prevent entries from overwriting them on the same candle.
    signals[exit_long_condition] = -2
    signals[exit_short_condition] = -1
    signals[buy_condition] = 2
    signals[sell_condition] = 1

    return signals

def pinbar_macd_strategy_vectorized(df):
    """
    Generates trading signals based on Pin Bars, MACD, and Moving Averages
    using a vectorized approach.

    Args:
        df (pandas.DataFrame): DataFrame with 'Open', 'Close', 'High', 'Low',
                               'SMA_short', 'SMA_long', 'MACD', and 'MACD_Signal' columns.

    Returns:
        pandas.Series: A Series with trading signals (2 = Buy, 1 = Sell, 0 = Hold,
                       -1 = Exit Short, -2 = Exit Long).
    """

    # Vettorializza le condizioni del Pin Bar
    # Calcola le dimensioni del corpo e delle ombre per l'intera Series
    body_size = np.abs(df['Open'] - df['Close'])
    lower_shadow = np.minimum(df['Open'], df['Close']) - df['Low']
    upper_shadow = df['High'] - np.maximum(df['Open'], df['Close'])

    is_bullish_pinbar = (lower_shadow > 2 * body_size) & (upper_shadow < body_size)
    is_bearish_pinbar = (upper_shadow > 2 * body_size) & (lower_shadow < body_size)

    # Crea colonne temporanee per i valori precedenti usando .shift()
    df['SMA_short_prev'] = df['SMA_short'].shift(1)
    df['SMA_long_prev'] = df['SMA_long'].shift(1)
    df['MACD_prev'] = df['MACD'].shift(1)
    df['MACD_Signal_prev'] = df['MACD_Signal'].shift(1)

    # Vettorializza le condizioni di cross per SMA e MACD
    bullish_sma_cross = (df['SMA_short_prev'] < df['SMA_long_prev']) & (df['SMA_short'] > df['SMA_long'])
    bearish_sma_cross = (df['SMA_short_prev'] > df['SMA_long_prev']) & (df['SMA_short'] < df['SMA_long'])
    bullish_macd_cross = (df['MACD_prev'] < df['MACD_Signal_prev']) & (df['MACD'] > df['MACD_Signal'])
    bearish_macd_cross = (df['MACD_prev'] > df['MACD_Signal_prev']) & (df['MACD'] < df['MACD_Signal'])

    # 📈 LONG SIGNAL (2)
    buy_condition = is_bullish_pinbar & (bullish_sma_cross | bullish_macd_cross)

    # 📉 SHORT SIGNAL (1)
    sell_condition = is_bearish_pinbar & (bearish_sma_cross | bearish_macd_cross)

    # 🚪 EXIT LONG (-2)
    exit_long_condition = bearish_sma_cross | bearish_macd_cross

    # 🚪 EXIT SHORT (-1)
    exit_short_condition = bullish_sma_cross | bullish_macd_cross

    # Inizializza una Series di segnali con un valore predefinito di 0 (Hold)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Applica i segnali con assegnazione vettoriale, gestendo la priorità.
    # Gli ingressi hanno la precedenza sulle uscite se si verificano sulla stessa candela.
    signals[exit_long_condition] = -2
    signals[exit_short_condition] = -1
    signals[buy_condition] = 2
    signals[sell_condition] = 1

    # Rimuovi le colonne temporanee
    df.drop(columns=['SMA_short_prev', 'SMA_long_prev', 'MACD_prev', 'MACD_Signal_prev'], inplace=True)

    return signals

def doji_rsi_simplified_vectorized(df):
    """
    Generates trading signals based on a Doji and RSI using a vectorized approach.

    Args:
        df (pandas.DataFrame): DataFrame with OHLC and RSI data.

    Returns:
        pandas.Series: A Series with trading signals (2 = Buy, 1 = Sell Short, 0 = Hold,
                       -1 = Exit Short, -2 = Exit Long).
    """

    # Vettorializza la condizione per identificare un Doji (corpo piccolo)
    is_doji = np.abs(df['Open'] - df['Close']) < (df['High'] - df['Low']) * 0.1

    # 📈 Condizioni di Ingresso Long e Short
    # Queste sono le maschere booleane che identificano le candele che soddisfano le condizioni.
    long_entry_condition = is_doji & (df['RSI'] < 30)
    short_entry_condition = is_doji & (df['RSI'] > 70)

    # 🚪 Condizioni di Uscita
    # Le condizioni di uscita sono separate e si applicano a tutte le righe.
    exit_long_condition = df['RSI'] > 50
    exit_short_condition = df['RSI'] < 50

    # Inizializza una Series di segnali con un valore predefinito di 0 (Hold)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Applica i segnali tramite assegnazione vettoriale.
    # L'ordine è importante per gestire la priorità dei segnali.
    signals[exit_long_condition] = -2
    signals[exit_short_condition] = -1
    signals[long_entry_condition] = 2
    signals[short_entry_condition] = 1

    return signals


import pandas as pd
import numpy as np


def bollinger_bands_adx_simple_vectorized(df, adx_entry_threshold=25, adx_exit_threshold=20):
    """
    Generates trading signals for the entire DataFrame based on the
    Bollinger Bands and ADX strategy, now including DMP and DMN for
    trend direction confirmation.

    The strategy looks for a breakout of the Bollinger Bands confirmed by a
    strong ADX and the directional indicators (DMP/DMN). An exit is
    triggered when the trend weakens, signaled by both a price crossover
    of the middle band and a drop in ADX.

    Args:
        df (pd.DataFrame): DataFrame with pre-calculated prices and indicators.
                          It must contain the following columns:
                          'Close' (closing price),
                          'BB_Upper' (upper Bollinger band),
                          'BB_Middle' (middle Bollinger band),
                          'BB_Lower' (lower Bollinger band),
                          'ADX_14' (ADX value),
                          'DMP_14' (Positive Directional Indicator),
                          'DMN_14' (Negative Directional Indicator).
        adx_entry_threshold (int): The ADX value required to confirm a trend for entry signals.
        adx_exit_threshold (int): The ADX value at which a trend is considered to be weakening,
                                   triggering an exit signal.

    Returns:
        pd.Series: A Series of trading signals (2 for Buy, 1 for Sell, -1 for Exit Short, -2 for Exit Long, 0 for Hold).
    """
    # Create boolean masks for entry and exit conditions.
    # This vectorized logic is much more efficient than for loops.

    # Entry conditions: Price breaks out of a band with a strong ADX,
    # and the directional indicators confirm the trend's direction.
    buy_condition = (df['Close'] > df['BB_Upper']) & (df['ADX_14'] >= adx_entry_threshold) & (
                df['DMP_14'] > df['DMN_14'])
    sell_condition = (df['Close'] < df['BB_Lower']) & (df['ADX_14'] >= adx_entry_threshold) & (
                df['DMN_14'] > df['DMP_14'])

    # Exit conditions: Price crosses the middle band AND ADX drops below the exit threshold.
    # Exit long when price drops below the middle band AND ADX falls below the exit threshold.
    exit_long_condition = (df['Close'] < df['BB_Middle']) & (df['ADX_14'] < adx_exit_threshold)
    # Exit short when price rises above the middle band AND ADX falls below the exit threshold.
    exit_short_condition = (df['Close'] > df['BB_Middle']) & (df['ADX_14'] < adx_exit_threshold)

    # Initialize a signals Series with a default value of 0 (Hold).
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Apply signals using vectorized assignment.
    # We apply exits last to ensure they override entries if conditions overlap,
    # which is a safe approach.

    # Apply entry signals
    signals[buy_condition] = 2  # 2 for Buy
    signals[sell_condition] = 1  # 1 for Sell

    # Apply exit signals
    signals[exit_long_condition] = -2  # -2 to Exit Long
    signals[exit_short_condition] = -1  # -1 to Exit Short

    return signals

def bollinger_bands_adx_simple_25_20_vectorized(df):
    return bollinger_bands_adx_simple_vectorized(df, adx_entry_threshold=25, adx_exit_threshold=20)

def bollinger_bands_adx_simple_35_20_vectorized(df):
    return bollinger_bands_adx_simple_vectorized(df, adx_entry_threshold=35, adx_exit_threshold=20)

def bollinger_bands_adx_simple_45_25_vectorized(df):
    return bollinger_bands_adx_simple_vectorized(df, adx_entry_threshold=45, adx_exit_threshold=25)


indicators_strategy =[
    bollinger_bands_mean_reversion_sma_vectorized,
    sma_stoch_close_strategy_vectorized,
    stochastic_oscillator_signal_vectorized,
    moving_average_crossover_signal_vectorized,
    #rsi_bollinger_macd_total_signal_v5_vectorized,
    #rsi_bollinger_macd_total_signal_v1_vectorized,
    #rsi_bollinger_macd_total_signal_v2_vectorized,
    rsi_hammer_70_30_vectorized,
    rsi_hammer_65_35_vectorized,
    rsi_hammer_80_20_vectorized,
    mean_reversion_signal_v1_vectorized,
    mean_reversion_signal_v2_vectorized,
    mean_reversion_signal_v3_vectorized,
    rsi_engulfing_signals_vectorized,
    mixed_signal_strategy_vectorized,
    pinbar_macd_strategy_vectorized,
    doji_rsi_simplified_vectorized,
    bollinger_bands_adx_simple_35_20_vectorized,
    bollinger_bands_adx_simple_25_20_vectorized,
    bollinger_bands_adx_simple_45_25_vectorized,
    x_bollinger_macd_total_signal_5_vectorized,
    x_bollinger_macd_total_signal_10_vectorized,
    x_rsi_bollinger_total_signal_15_65_35_vectorized,
    x_rsi_bollinger_total_signal_10_70_30_vectorized,
    adx_sma_tf_20_50_25,
    adx_sma_tf_10_30_25,
    adx_sma_tf_15_40_35,
    adx_sma_tf_10_30_35,
    adx_trend_breakout_20_25,
    adx_trend_breakout_10_40,
    adx_trend_breakout_10_35,
    #keltner_channel_vectorized,
    #donchian_channel_vectorized,
    #weekly_breakout_vectorized,
    t_indicators_combined_signal_vectorized
]

