
import pandas as pd

def bollinger_bands_mean_reversion_sma(df, current_p):
    """
    Generates trading signals based on the Bollinger Bands Mean Reversion strategy,
    strengthened with SMA_short and SMA_long confirmation.

    Strategy Description:
    This strategy aims to capitalize on the tendency of prices to revert to the mean
    when they deviate significantly from it, using Bollinger Bands, and reinforces
    signals with SMA_short and SMA_long confirmation.

    Entry Rules:
    - Long (Buy): When the price touches or falls below the lower Bollinger Band AND
                   SMA_short is above SMA_long.
    - Short (Sell): When the price touches or rises above the upper Bollinger Band AND
                    SMA_short is below SMA_long.

    Exit Rules:
    - Take Profit: When the price reaches the middle Bollinger Band (Simple Moving Average).
    - Stop Loss: Place a stop loss slightly outside the opposite Bollinger Band.

    Additional Considerations:
    - The middle Bollinger Band is a Simple Moving Average (SMA) of the closing prices.
    - This strategy is more effective in range-bound or sideways markets.
    - SMA_short and SMA_long are used to confirm the trend direction and filter false signals.
    - Ensure SMA_short and SMA_long are pre-calculated in the DataFrame.
    - Adjust Bollinger Band parameters based on timeframe.
    - Implement proper risk management (stop loss, position sizing).
    - Backtest the strategy on historical data.
    """

    current_pos = df.index.get_loc(current_p)

    # Buy signal with SMA confirmation
    if (df['Close'].iloc[current_pos] <= df['BB_Lower'].iloc[current_pos] and
        df['SMA_short'].iloc[current_pos] > df['SMA_long'].iloc[current_pos]):
        return 2  # Buy

    # Sell signal with SMA confirmation
    elif (df['Close'].iloc[current_pos] >= df['BB_Upper'].iloc[current_pos] and
          df['SMA_short'].iloc[current_pos] < df['SMA_long'].iloc[current_pos]):
        return 1  # Sell

    return 0  # Hold

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

def rsi_bollinger_macd_total_signal_v5(df, current_p, tolerance_percent=5):
    """
    Generates trading signals based on RSI, Bollinger Bands, and MACD,
    using a percentage range to define "near" the Bollinger Bands.

    Args:
        df (pandas.DataFrame): DataFrame containing OHLC data.
        current_p (int): Index of the current data point.
        tolerance_percent (int): Percentage tolerance for "near" (e.g., 5).

    Returns:
        int: Trading signal (2 = Buy, 1 = Sell, 0 = Hold, -1 = Exit Short, -2 = Exit Long).
    """

    current_pos = df.index.get_loc(current_p)

    # Buy (Long Entry) Conditions
    rsi_condition = df['RSI'].iloc[current_pos] < 50
    lower_band = df['BB_Lower'].iloc[current_pos]
    lower_tolerance_min = lower_band - (lower_band * tolerance_percent / 100)
    lower_tolerance_max = lower_band + (lower_band * tolerance_percent / 100)
    bollinger_lower_condition = lower_tolerance_min < df['Close'].iloc[current_pos] < lower_tolerance_max
    macd_condition = df['MACD'].iloc[current_pos] > df['MACD_Signal'].iloc[current_pos]

    # Complete Buy Condition
    if rsi_condition and bollinger_lower_condition and macd_condition:
        return 2  # Buy Signal (Long Entry)

    # Sell (Short Entry) Conditions
    rsi_sell_condition = df['RSI'].iloc[current_pos] > 70
    middle_band = df['BB_Middle'].iloc[current_pos]
    middle_tolerance_min = middle_band - (middle_band * tolerance_percent / 100)
    middle_tolerance_max = middle_band + (middle_band * tolerance_percent / 100)
    bollinger_middle_condition = middle_tolerance_min < df['Close'].iloc[current_pos] < middle_tolerance_max
    macd_sell_condition = df['MACD'].iloc[current_pos] < df['MACD_Signal'].iloc[current_pos]

    # Complete Sell Condition
    if rsi_sell_condition and bollinger_middle_condition and macd_sell_condition:
        return 1  # Sell Signal (Short Entry)

    # Exit Long Condition
    exit_long_rsi = df['RSI'].iloc[current_pos] > 70
    exit_long_bollinger = df['Close'].iloc[current_pos] >= df['BB_Upper'].iloc[current_pos]
    if exit_long_rsi or exit_long_bollinger:
        return -2

    # Exit Short Condition
    exit_short_rsi = df['RSI'].iloc[current_pos] < 50
    exit_short_bollinger = df['Close'].iloc[current_pos] <= df['BB_Lower'].iloc[current_pos]
    if exit_short_rsi or exit_short_bollinger:
        return -1

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
    c5 = df['Close'].iloc[current_pos] <= df['BB_Upper'].iloc[current_pos]  # Price below or near the middle Bollinger band
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

def mean_reversion_signal_v1(df, current_p, tolerance_percent=5):
    """
    Generates mean reversion trading signals based on RSI and Bollinger Bands,
    using a percentage range to define "near" the Bollinger Bands.

    Args:
        df (pandas.DataFrame): DataFrame containing OHLC data.
        current_p (int): Index of the current data point.
        tolerance_percent (int): Percentage tolerance for "near" (e.g., 5).

    Returns:
        int: Trading signal (2 = Buy, 1 = Sell, 0 = Hold, -1 = Exit Short, -2 = Exit Long).
    """

    current_pos = df.index.get_loc(current_p)

    # LONG SIGNAL
    # if self.data.close[0] <= self.bollinger.lines.bot and self.rsi[0] < 30:
    rsi_long_condition = df['RSI'].iloc[current_pos] < 30
    lower_band = df['BB_Lower'].iloc[current_pos]
    lower_tolerance_min = lower_band - (lower_band * tolerance_percent / 100)
    lower_tolerance_max = lower_band + (lower_band * tolerance_percent / 100)
    bollinger_lower_condition = lower_tolerance_min < df['Close'].iloc[current_pos] < lower_tolerance_max
    if bollinger_lower_condition and rsi_long_condition:
        return 2

    # SHORT SIGNAL
    # elif self.data.close[0] >= self.bollinger.lines.top and self.rsi[0] > 70:
    rsi_short_condition = df['RSI'].iloc[current_pos] > 70
    upper_band = df['BB_Upper'].iloc[current_pos]
    upper_tolerance_min = upper_band - (upper_band * tolerance_percent / 100)
    upper_tolerance_max = upper_band + (upper_band * tolerance_percent / 100)
    bollinger_upper_condition = upper_tolerance_min < df['Close'].iloc[current_pos] < upper_tolerance_max
    if bollinger_upper_condition and rsi_short_condition:
        return 1

    # EXIT LONG
    # if self.rsi[0] > 50 or self.data.close[0] >= self.bollinger.lines.top
    exit_long_rsi = df['RSI'].iloc[current_pos] > 50
    exit_long_bollinger = df['Close'].iloc[current_pos] >= df['BB_Upper'].iloc[current_pos]
    if exit_long_rsi or exit_long_bollinger:
        return -2

    # EXIT SHORT
    # if self.rsi[0] < 50 or self.data.close[0] <= self.bollinger.lines.bot:
    exit_short_rsi = df['RSI'].iloc[current_pos] < 50
    exit_short_bollinger = df['Close'].iloc[current_pos] <= df['BB_Lower'].iloc[current_pos]
    if exit_short_rsi or exit_short_bollinger:
        return -1

    return 0

def rsi_engulfing_signals(df, current_p):
    """
    Generates trading signals based on Engulfing candlestick patterns and RSI,
    confirmed by overbought/oversold conditions.

    Args:
        df (pandas.DataFrame): DataFrame containing OHLC data and RSI.
        current_p (int): Index of the current data point.

    Returns:
        int: Trading signal (2 = Buy, 1 = Sell Short, 0 = Hold,
             -1 = Exit Short, -2 = Exit Long).
    """

    current_pos = df.index.get_loc(current_p)

    # Ensure there is at least one previous candle for the analysis
    if current_pos < 1:
        return 0

    previous_candle = df.iloc[current_pos - 1]
    current_candle = df.iloc[current_pos]

    # Conditions for Bullish Engulfing
    is_bullish_engulfing = (
        current_candle['Close'] > previous_candle['Open'] and
        current_candle['Open'] < previous_candle['Close'] and
        current_candle['Low'] < previous_candle['Low']
    )

    # Conditions for Bearish Engulfing
    is_bearish_engulfing = (
        current_candle['Open'] > previous_candle['Close'] and
        current_candle['Close'] < previous_candle['Open'] and
        current_candle['High'] > previous_candle['High']
    )

    # 📈 LONG SIGNAL (2)
    # Enter long if there is a bullish engulfing in an oversold area (RSI < 30)
    rsi_long_condition = current_candle['RSI'] < 30
    if is_bullish_engulfing and rsi_long_condition:
        return 2

    # 📉 SHORT SIGNAL (1)
    # Enter short if there is a bearish engulfing in an overbought area (RSI > 70)
    rsi_short_condition = current_candle['RSI'] > 70
    if is_bearish_engulfing and rsi_short_condition:
        return 1

    # 🚪 EXIT LONG (-2)
    # Exit a long position if the RSI moves above 50
    exit_long_rsi_condition = current_candle['RSI'] > 50
    if exit_long_rsi_condition:
        return -2

    # 🚪 EXIT SHORT (-1)
    # Exit a short position if the RSI moves below 50
    exit_short_rsi_condition = current_candle['RSI'] < 50
    if exit_short_rsi_condition:
        return -1

    # ⏸️ HOLD (0)
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

def mixed_signal_strategy(df, current_p):
    """
    Generates trading signals by combining RSI, Stochastic, and Moving Averages.

    Args:
        df (pandas.DataFrame): DataFrame with the specified columns.
        current_p (int): Index of the current data point.

    Returns:
        int: Trading signal (2 = Buy, 1 = Sell Short, 0 = Hold, -1 = Exit Short, -2 = Exit Long).
    """

    current_pos = df.index.get_loc(current_p)

    # Ensure there is a previous data point for comparison
    if current_pos < 1:
        return 0

    close = df['Close'].iloc[current_pos]
    rsi = df['RSI'].iloc[current_pos]
    stoch_k = df['STOCHk_14_3_3'].iloc[current_pos]
    stoch_d = df['STOCHd_14_3_3'].iloc[current_pos]
    sma_short = df['SMA_short'].iloc[current_pos]
    sma_long = df['SMA_long'].iloc[current_pos]
    bb_lower = df['BB_Lower'].iloc[current_pos]
    bb_upper = df['BB_Upper'].iloc[current_pos]

    # Conditions to confirm the trend
    bullish_trend = sma_short > sma_long
    bearish_trend = sma_short < sma_long

    # 📈 LONG SIGNAL (2)
    # Enter if the trend is bullish AND momentum indicators are in an oversold zone
    if bullish_trend and (rsi < 30 or (stoch_k < 20 and stoch_d < 20)) and close < bb_lower:
        return 2

    # 📉 SHORT SIGNAL (1)
    # Enter if the trend is bearish AND momentum indicators are in an overbought zone
    if bearish_trend and (rsi > 70 or (stoch_k > 80 and stoch_d > 80)) and close > bb_upper:
        return 1

    # 🚪 EXIT LONG (-2)
    # Exit a long position if the trend reverses or momentum is overbought
    if bearish_trend or (rsi > 70 and close > bb_upper):
        return -2

    # 🚪 EXIT SHORT (-1)
    # Exit a short position if the trend reverses or momentum is oversold
    if bullish_trend or (rsi < 30 and close < bb_lower):
        return -1

    # ⏸️ HOLD (0)
    return 0


def pinbar_macd_strategy(df, current_p):
    """
    Generates trading signals based on Pin Bars, MACD, and Moving Averages.

    Args:
        df (pandas.DataFrame): DataFrame with the specified columns.
        current_p (int): Index of the current data point.

    Returns:
        int: Trading signal (2 = Buy, 1 = Sell Short, 0 = Hold, -1 = Exit Short, -2 = Exit Long).
    """

    current_pos = df.index.get_loc(current_p)

    # Ensure there is a previous data point for comparison
    if current_pos < 1:
        return 0

    close = df['Close'].iloc[current_pos]
    open_p = df['Open'].iloc[current_pos]
    high = df['High'].iloc[current_pos]
    low = df['Low'].iloc[current_pos]

    # Previous data for SMA and MACD crosses
    sma_short_prev = df['SMA_short'].iloc[current_pos - 1]
    sma_long_prev = df['SMA_long'].iloc[current_pos - 1]
    macd_prev = df['MACD'].iloc[current_pos - 1]
    macd_signal_prev = df['MACD_Signal'].iloc[current_pos - 1]

    sma_short = df['SMA_short'].iloc[current_pos]
    sma_long = df['SMA_long'].iloc[current_pos]
    macd = df['MACD'].iloc[current_pos]
    macd_signal = df['MACD_Signal'].iloc[current_pos]

    # Conditions to identify a Bullish Pin Bar
    body_size = abs(open_p - close)
    lower_shadow = min(open_p, close) - low
    upper_shadow = high - max(open_p, close)

    is_bullish_pinbar = (
            lower_shadow > 2 * body_size and
            upper_shadow < body_size
    )

    # Conditions to identify a Bearish Pin Bar
    is_bearish_pinbar = (
            upper_shadow > 2 * body_size and
            lower_shadow < body_size
    )

    # Bullish SMA cross (trend)
    bullish_sma_cross = (sma_short_prev < sma_long_prev) and (sma_short > sma_long)

    # Bearish SMA cross (trend)
    bearish_sma_cross = (sma_short_prev > sma_long_prev) and (sma_short < sma_long)

    # Bullish MACD cross (momentum)
    bullish_macd_cross = (macd_prev < macd_signal_prev) and (macd > macd_signal)

    # Bearish MACD cross (momentum)
    bearish_macd_cross = (macd_prev > macd_signal_prev) and (macd < macd_signal)

    # 📈 LONG SIGNAL (2)
    # Enter long if a bullish Pin Bar is present AND there's a bullish cross in trend or momentum
    if is_bullish_pinbar and (bullish_sma_cross or bullish_macd_cross):
        return 2

    # 📉 SHORT SIGNAL (1)
    # Enter short if a bearish Pin Bar is present AND there's a bearish cross in trend or momentum
    if is_bearish_pinbar and (bearish_sma_cross or bearish_macd_cross):
        return 1

    # 🚪 EXIT LONG (-2)
    # Exit a long position if the MACD turns bearish or there's a bearish SMA cross
    if bearish_sma_cross or bearish_macd_cross:
        return -2

    # 🚪 EXIT SHORT (-1)
    # Exit a short position if the MACD turns bullish or there's a bullish SMA cross
    if bullish_sma_cross or bullish_macd_cross:
        return -1

    # ⏸️ HOLD (0)
    return 0

def doji_rsi_simplified(df, current_p):
    """
    Generates trading signals based on a Doji and RSI.

    Args:
        df (pandas.DataFrame): DataFrame with OHLC and RSI data.
        current_p (int): Index of the current data point.

    Returns:
        int: Trading signal (2 = Buy, 1 = Sell Short, 0 = Hold, -1 = Exit Short, -2 = Exit Long).
    """
    current_pos = df.index.get_loc(current_p)

    if current_pos < 1:
        return 0

    current_candle = df.iloc[current_pos]
    rsi = current_candle['RSI']

    # Condition to identify a Doji (small body)
    is_doji = abs(current_candle['Open'] - current_candle['Close']) < (current_candle['High'] - current_candle['Low']) * 0.1

    # 📈 Long Entry
    if is_doji and rsi < 30:
        return 2

    # 📉 Short Entry
    if is_doji and rsi > 70:
        return 1

    # 🚪 Exit a Long position
    # If RSI moves above 50, it indicates momentum is now in your favor.
    if rsi > 50:
        return -2

    # 🚪 Exit a Short position
    # If RSI moves below 50, it indicates momentum is now in your favor.
    if rsi < 50:
        return -1

    # ⏸️ Hold
    return 0

indicators_strategy =[
    bollinger_bands_mean_reversion_sma,
    sma_stoch_close_strategy,
    stochastic_oscillator_signal,
    moving_average_crossover_signal,
    rsi_bollinger_macd_total_signal_v5,
    rsi_bollinger_macd_total_signal_v1,
    rsi_bollinger_macd_total_signal_v2,
    mean_reversion_signal_v1,
    rsi_engulfing_signals,
    mixed_signal_strategy,
    pinbar_macd_strategy,
    doji_rsi_simplified,
    t_indicators_combined_signal
]