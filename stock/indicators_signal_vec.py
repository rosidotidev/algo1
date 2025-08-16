
import pandas as pd
import numpy as np

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

indicators_strategy =[
    bollinger_bands_mean_reversion_sma_vectorized,
    sma_stoch_close_strategy_vectorized,
    stochastic_oscillator_signal_vectorized,
    moving_average_crossover_signal_vectorized,
    rsi_bollinger_macd_total_signal_v5_vectorized,
    rsi_bollinger_macd_total_signal_v1_vectorized,
    rsi_bollinger_macd_total_signal_v2_vectorized,
    mean_reversion_signal_v1_vectorized,
    mean_reversion_signal_v2_vectorized,
    mean_reversion_signal_v3_vectorized,
    rsi_engulfing_signals_vectorized,
    mixed_signal_strategy_vectorized,
    pinbar_macd_strategy_vectorized,
    doji_rsi_simplified_vectorized,
    t_indicators_combined_signal_vectorized
]