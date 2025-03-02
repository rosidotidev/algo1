
import pandas as pd
import pandas_ta as ta  # Import pandas_ta

def sma_stoch_close_strategy(df, current_p):
    """Genera segnali meno selettivi."""

    current_pos = df.index.get_loc(current_p)

    # Buy Conditions (Rilassate)
    if (df['SMA_short'].iloc[current_pos] > df['SMA_long'].iloc[current_pos] and
        df['STOCHk_14_3_3'].iloc[current_pos] > df['STOCHd_14_3_3'].iloc[current_pos] and
        df['STOCHk_14_3_3'].iloc[current_pos] < 35 and  # Zona di ipervenduto rilassata
        df['Close'].iloc[current_pos] > df['BB_Middle'].iloc[current_pos]):
        return 2  # Buy

    # Sell Conditions (Rilassate)
    if (df['SMA_short'].iloc[current_pos] < df['SMA_long'].iloc[current_pos] and
        df['STOCHk_14_3_3'].iloc[current_pos] < df['STOCHd_14_3_3'].iloc[current_pos] and
        df['STOCHk_14_3_3'].iloc[current_pos] > 65 and  # Zona di ipercomprato rilassata
        df['Close'].iloc[current_pos] < df['BB_Middle'].iloc[current_pos]):
        return 1  # Sell

    return 0  # Hold

def stochastic_oscillator_signal(df, current_p):
    """
    Generates trading signals based on the Stochastic Oscillator.

    Strategy Explanation:
    The Stochastic Oscillator is a momentum indicator that compares a security's closing price
    to its range over a certain period. It is used to identify potential overbought and oversold
    conditions. The %K line represents the current close relative to the period's high/low range,
    and the %D line is a moving average of %K.

    Buy signals are generated when %K crosses above %D in the oversold zone (typically below 20),
    indicating potential upward momentum. Sell signals are generated when %K crosses below %D
    in the overbought zone (typically above 80), indicating potential downward momentum.

    It is assumed that the DataFrame `df` already contains the following columns:
    - 'STOCHk_14_3_3': %K line of the Stochastic Oscillator.
    - 'STOCHd_14_3_3': %D line of the Stochastic Oscillator.

    To calculate these columns using pandas_ta, you can use the following code:
    df.ta.stoch(high=df['High'], low=df['Low'], close=df['Close'], k=14, d=3, append=True)

    Args:
        df (pandas.DataFrame): DataFrame containing financial data and the Stochastic Oscillator.
        current_p (int): Index of the current data point.

    Returns:
        int: Trading signal (2 = Buy, 1 = Sell, 0 = Hold, -1 = Exit Short, -2 = Exit Long).
    """

    current_pos = df.index.get_loc(current_p)

    # Buy (Long Entry) Conditions
    if (df['STOCHk_14_3_3'].iloc[current_pos] > df['STOCHd_14_3_3'].iloc[current_pos] and
        df['STOCHk_14_3_3'].iloc[current_pos - 1] <= df['STOCHd_14_3_3'].iloc[current_pos - 1] and
        df['STOCHk_14_3_3'].iloc[current_pos] < 30):  # Oversold
        return 2  # Buy Signal (Long Entry)

    # Sell (Short Entry) Conditions
    if (df['STOCHk_14_3_3'].iloc[current_pos] < df['STOCHd_14_3_3'].iloc[current_pos] and
        df['STOCHk_14_3_3'].iloc[current_pos - 1] >= df['STOCHd_14_3_3'].iloc[current_pos - 1] and
        df['STOCHk_14_3_3'].iloc[current_pos] > 70):  # Overbought
        return 1  # Sell Signal (Short Entry)

    # Exit Long Conditions
    if (df['STOCHk_14_3_3'].iloc[current_pos] < df['STOCHd_14_3_3'].iloc[current_pos] and
        df['STOCHk_14_3_3'].iloc[current_pos - 1] >= df['STOCHd_14_3_3'].iloc[current_pos - 1]):
        return -2 #Exit Long

    # Exit Short Conditions
    if (df['STOCHk_14_3_3'].iloc[current_pos] > df['STOCHd_14_3_3'].iloc[current_pos] and
        df['STOCHk_14_3_3'].iloc[current_pos - 1] <= df['STOCHd_14_3_3'].iloc[current_pos - 1]):
        return -1 #Exit Short

    # Hold Condition (No Action)
    return 0  # Hold Signal (Do nothing)

def moving_average_crossover_signal(df, current_p):
    """
    Generates trading signals based on moving average crossover.

    Strategy Explanation:
    The Moving Average Crossover strategy is a trend-following strategy that uses two moving averages
    of different periods to generate trading signals. A shorter-period moving average is used to
    represent the short-term trend, and a longer-period moving average is used to represent the
    long-term trend.

    A buy signal is generated when the short-period moving average crosses above the long-period
    moving average, indicating a potential upward trend. A sell signal is generated when the
    short-period moving average crosses below the long-period moving average, indicating a
    potential downward trend.

    It is assumed that the DataFrame `df` already contains the following columns:
    - 'SMA_short': Short moving average.
    - 'SMA_long': Long moving average.

    To calculate these columns using pandas_ta, you can use the following code:
    df['SMA_short'] = ta.sma(df['Close'], length=short_window)
    df['SMA_long'] = ta.sma(df['Close'], length=long_window)
    (Replace 'short_window' and 'long_window' with your desired periods)

    Args:
        df (pandas.DataFrame): DataFrame containing financial data and moving averages.
        current_p (int): Index of the current data point.

    Returns:
        int: Trading signal (2 = Buy, 1 = Sell, 0 = Hold, -1 = Exit Short, -2 = Exit Long).
    """

    current_pos = df.index.get_loc(current_p)

    # Buy (Long Entry) Conditions
    if (df['SMA_short'].iloc[current_pos] > df['SMA_long'].iloc[current_pos] and
        df['SMA_short'].iloc[current_pos - 1] <= df['SMA_long'].iloc[current_pos - 1]):
        return 2  # Buy Signal (Long Entry)

    # Sell (Short Entry) Conditions
    if (df['SMA_short'].iloc[current_pos] < df['SMA_long'].iloc[current_pos] and
        df['SMA_short'].iloc[current_pos - 1] >= df['SMA_long'].iloc[current_pos - 1]):
        return 1  # Sell Signal (Short Entry)

    # Exit Long Conditions
    if (df['SMA_short'].iloc[current_pos] < df['SMA_long'].iloc[current_pos] and
        df['SMA_short'].iloc[current_pos - 1] >= df['SMA_long'].iloc[current_pos - 1]):
        return -2 #Exit Long

    # Exit Short Conditions
    if (df['SMA_short'].iloc[current_pos] > df['SMA_long'].iloc[current_pos] and
        df['SMA_short'].iloc[current_pos - 1] <= df['SMA_long'].iloc[current_pos - 1]):
        return -1 #Exit Short

    # Hold Condition (No Action)
    return 0  # Hold Signal (Do nothing)

def rsi_bollinger_macd_total_signal_v1(df, current_p):
    current_pos = df.index.get_loc(current_p)
    # **Buy (Long Entry) Conditions**
    # RSI_14 < 50, price above or near the lower Bollinger band, MACD > Signal
    """
     if self.rsi < 50 and self.data.close >= self.bollinger.lines.bot and self.macd.macd > self.macd.signal:
    """
    c1 = df['RSI'].iloc[current_pos] < 50  # Condition RSI_14 < 50
    c2 = df['Close'].iloc[current_pos] >= df['BB_Lower'].iloc[current_pos]  # Price above or near the lower Bollinger band
    c3 = df['MACD'].iloc[current_pos] > df['MACD_Signal'].iloc[current_pos]  # MACD > Signal

    # Complete Buy Condition
    if c1 and c2 and c3:
        return 2  # Buy Signal (Long Entry)

    # **Sell (Short Entry) Conditions**
    # RSI_14 > 70, price below or near the middle Bollinger band, MACD < Signal
    c4 = df['RSI'].iloc[current_pos] > 70  # Condition RSI_14 > 70
    c5 = df['Close'].iloc[current_pos] <= df['BB_Middle'].iloc[current_pos]  # Price below or near the middle Bollinger band
    c6 = df['MACD'].iloc[current_pos] < df['MACD_Signal'].iloc[current_pos]  # MACD < Signal

    # Complete Sell Condition
    if c4 and c5 and c6:
        return 1  # Sell Signal (Short Entry)
    """
    EXIT LONG CONDITION
    self.rsi > 70 or self.data.close >= self.bollinger.lines.top
    """
    c7 = df['RSI'].iloc[current_pos] > 70
    c8 = df['Close'].iloc[current_pos] >= df['BB_Upper'].iloc[current_pos]
    if c7 and c8:
        return -2
    """
    EXIT SHORT
    if self.rsi < 50 or self.data.close <= self.bollinger.lines.bot:
    """
    c9= df['RSI'].iloc[current_pos] < 50
    c10= df['Close'].iloc[current_pos] <= df['BB_Lower'].iloc[current_pos]
    if c9 and c10:
        return -1
    # **Hold Condition (No Action)**
    return 0  # Hold Signal (Do nothing)

def rsi_bollinger_macd_total_signal_v2(df, current_p):
    current_pos = df.index.get_loc(current_p)
    # **Buy (Long Entry) Conditions**
    # RSI_14 < 50, price above or near the lower Bollinger band, MACD > Signal
    """
     if self.rsi < 50 and self.data.close >= self.bollinger.lines.bot and self.macd.macd > self.macd.signal:
    """
    c1 = df['RSI'].iloc[current_pos] < 50  # Condition RSI_14 < 50
    c2 = df['Close'].iloc[current_pos] >= df['BB_Lower'].iloc[current_pos]  # Price above or near the lower Bollinger band
    c3 = df['MACD'].iloc[current_pos] > df['MACD_Signal'].iloc[current_pos]  # MACD > Signal

    # Complete Buy Condition
    if c1 and c2 and c3:
        return 2  # Buy Signal (Long Entry)

    # **Sell (Short Entry) Conditions**
    # RSI_14 > 70, price below or near the middle Bollinger band, MACD < Signal
    c4 = df['RSI'].iloc[current_pos] > 70  # Condition RSI_14 > 70
    c5 = df['Close'].iloc[current_pos] <= df['BB_Middle'].iloc[current_pos]  # Price below or near the middle Bollinger band
    c6 = df['MACD'].iloc[current_pos] < df['MACD_Signal'].iloc[current_pos]  # MACD < Signal

    # Complete Sell Condition
    if c4 and c5 and c6:
        return 1  # Sell Signal (Short Entry)
    """
    EXIT LONG CONDITION
    self.rsi > 70 or self.data.close >= self.bollinger.lines.top
    """
    c7 = df['RSI'].iloc[current_pos] > 70
    c8 = df['Close'].iloc[current_pos] >= df['BB_Upper'].iloc[current_pos]
    if c7 or c8:
        return -2
    """
    EXIT SHORT
    if self.rsi < 50 or self.data.close <= self.bollinger.lines.bot:
    """
    c9= df['RSI'].iloc[current_pos] < 50
    c10= df['Close'].iloc[current_pos] <= df['BB_Lower'].iloc[current_pos]
    if c9 or c10:
        return -1
    # **Hold Condition (No Action)**
    return 0  # Hold Signal (Do nothing)

def mean_reversion_signal_v1(df, current_p):
    current_pos = df.index.get_loc(current_p)
    """
    LONG SIGNAL
    if self.data.close[0] <= self.bollinger.lines.bot and self.rsi[0] < 30:
    """
    c2 = df['RSI'].iloc[current_pos] < 30  # Condition RSI_14 < 50
    c1 = df['Close'].iloc[current_pos] >= df['BB_Lower'].iloc[current_pos]  # Price above or near the lower Bollinger band
    if c1 and c2:
        return 2
    """
    SHORT
    elif self.data.close[0] >= self.bollinger.lines.top and self.rsi[0] > 70:
    """
    c3 = df['RSI'].iloc[current_pos] > 70
    c4 = df['Close'].iloc[current_pos] >= df['BB_Upper'].iloc[current_pos]
    if c3 and c4:
        return 1
    """
    EXIT LONG
    if self.rsi[0] > 50 or self.data.close[0] >= self.bollinger.lines.top
    """
    c4 = df['RSI'].iloc[current_pos] > 50
    c5 = df['Close'].iloc[current_pos] >= df['BB_Upper'].iloc[current_pos]
    if c4 or c5:
        return -2
    """
    EXIT SHORT
    if self.rsi[0] < 50 or self.data.close[0] <= self.bollinger.lines.bot:
    """
    c6 = df['RSI'].iloc[current_pos] < 50
    c7 = df['Close'].iloc[current_pos] <= df['BB_Lower'].iloc[current_pos]
    if c6 or c7:
        return -1
    return 0

def t_indicators_combined_signal(df, current_candle):
    """
    Combines signals from various candlestick strategies, excluding itself.

    Args:
        df (pd.DataFrame): DataFrame containing candlestick data.
        current_candle (index): Index of the current candle.


    Returns:
        int: 2 for buy signal, 1 for sell signal, 0 for no signal.
    """
    # Create a new list excluding combined_signal
    strategies = [func for func in indicators_strategy if func != t_indicators_combined_signal]

    signals = [func(df, current_candle) for func in strategies]
    buy_signals = signals.count(2)
    sell_signals = signals.count(1)

    # Example combination logic (customize as needed)
    if buy_signals > sell_signals and buy_signals > 2:
        return 2  # Strong buy signal
    elif sell_signals > buy_signals and sell_signals > 2:
        return 1  # Strong sell signal
    else:
        return 0  # No strong signal


indicators_strategy =[
    sma_stoch_close_strategy,
    stochastic_oscillator_signal,
    moving_average_crossover_signal,
    rsi_bollinger_macd_total_signal_v1,
    rsi_bollinger_macd_total_signal_v2,
    mean_reversion_signal_v1,
    t_indicators_combined_signal
]