

def dax_total_signal(df, current_candle):
    current_pos = df.index.get_loc(current_candle)

    # Buy condition
    c1 = df['High'].iloc[current_pos] > df['High'].iloc[current_pos - 1]
    c2 = df['High'].iloc[current_pos - 1] > df['Low'].iloc[current_pos]
    c3 = df['Low'].iloc[current_pos] > df['High'].iloc[current_pos - 2]
    c4 = df['High'].iloc[current_pos - 2] > df['Low'].iloc[current_pos - 1]
    c5 = df['Low'].iloc[current_pos - 1] > df['High'].iloc[current_pos - 3]
    c6 = df['High'].iloc[current_pos - 3] > df['Low'].iloc[current_pos - 2]
    c7 = df['Low'].iloc[current_pos - 2] > df['Low'].iloc[current_pos - 3]

    if c1 and c2 and c3 and c4 and c5 and c6 and c7:
        return 2

    # Symmetrical conditions for short (sell condition)
    c1 = df['Low'].iloc[current_pos] < df['Low'].iloc[current_pos - 1]
    c2 = df['Low'].iloc[current_pos - 1] < df['High'].iloc[current_pos]
    c3 = df['High'].iloc[current_pos] < df['Low'].iloc[current_pos - 2]
    c4 = df['Low'].iloc[current_pos - 2] < df['High'].iloc[current_pos - 1]
    c5 = df['High'].iloc[current_pos - 1] < df['Low'].iloc[current_pos - 3]
    c6 = df['Low'].iloc[current_pos - 3] < df['High'].iloc[current_pos - 2]
    c7 = df['High'].iloc[current_pos - 2] < df['High'].iloc[current_pos - 3]

    if c1 and c2 and c3 and c4 and c5 and c6 and c7:
        return 1
    return 0

def dax_momentum_signal(df, current_candle):
    """
    Determines if the current candle generates a BUY (2) or SELL (1) signal
    based on momentum-based price action rules.
    """
    current_pos = df.index.get_loc(current_candle)

    # Ensure we have enough candles for calculations
    if current_pos < 3:
        return 0  # Not enough historical data

    # Buy condition: Bullish momentum
    c1 = df['High'].iloc[current_pos] > df['High'].iloc[current_pos - 1]
    c2 = df['Low'].iloc[current_pos - 1] > df['Low'].iloc[current_pos - 2]
    c3 = df['Close'].iloc[current_pos] > df['Low'].iloc[current_pos] + (df['High'].iloc[current_pos] - df['Low'].iloc[current_pos]) * 0.7

    if c1 and c2 and c3:
        return 2  # Buy signal

    # Sell condition: Bearish momentum
    c1 = df['Low'].iloc[current_pos] < df['Low'].iloc[current_pos - 1]
    c2 = df['High'].iloc[current_pos - 1] < df['High'].iloc[current_pos - 2]
    c3 = df['Close'].iloc[current_pos] < df['High'].iloc[current_pos] - (df['High'].iloc[current_pos] - df['Low'].iloc[current_pos]) * 0.7

    if c1 and c2 and c3:
        return 1  # Sell signal

    return 0  # No signal


def inside_bar_breakout_signal(df, current_candle):
    """
    Determines if the current candle generates a BUY (2) or SELL (1) signal
    based on the Inside Bar Breakout strategy.
    """
    current_pos = df.index.get_loc(current_candle)

    # Ensure we have at least one previous candle
    if current_pos < 1:
        return 0  # Not enough historical data

    # Previous candle (Inside Bar reference)
    prev_high = df['High'].iloc[current_pos - 1]
    prev_low = df['Low'].iloc[current_pos - 1]

    # Current candle
    current_high = df['High'].iloc[current_pos]
    current_low = df['Low'].iloc[current_pos]

    # Check if the previous candle is an Inside Bar
    if prev_high > df['High'].iloc[current_pos - 2] and prev_low < df['Low'].iloc[current_pos - 2]:

        # Buy Signal → If price breaks above the Inside Bar high
        if current_high > prev_high:
            return 2

            # Sell Signal → If price breaks below the Inside Bar low
        if current_low < prev_low:
            return 1

    return 0  # No breakout

def three_bar_reversal_signal(df, current_candle):
    """
    Determines if the current candle generates a BUY (2) or SELL (1) signal
    based on the Three Bar Reversal strategy.
    """
    current_pos = df.index.get_loc(current_candle)

    # Ensure we have at least two previous candles
    if current_pos < 2:
        return 0  # Not enough historical data

    # Candle -2 (First Candle)
    first_open = df['Open'].iloc[current_pos - 2]
    first_close = df['Close'].iloc[current_pos - 2]
    first_high = df['High'].iloc[current_pos - 2]
    first_low = df['Low'].iloc[current_pos - 2]

    # Candle -1 (Second Candle - Indecision Bar)
    second_open = df['Open'].iloc[current_pos - 1]
    second_close = df['Close'].iloc[current_pos - 1]
    second_high = df['High'].iloc[current_pos - 1]
    second_low = df['Low'].iloc[current_pos - 1]

    # Candle 0 (Third Candle - Confirmation Bar)
    third_open = df['Open'].iloc[current_pos]
    third_close = df['Close'].iloc[current_pos]
    third_high = df['High'].iloc[current_pos]
    third_low = df['Low'].iloc[current_pos]

    # **Bullish Reversal Condition**
    # 1. First candle is bearish (Close < Open)
    # 2. Second candle makes a lower low but has a higher high (Indecision candle)
    # 3. Third candle is bullish (Close > Open) and closes above the first candle's open
    if first_close < first_open and \
       second_low < first_low and second_high > first_high and \
       third_close > third_open and third_close > first_open:
        return 2  # Buy Signal

    # **Bearish Reversal Condition**
    # 1. First candle is bullish (Close > Open)
    # 2. Second candle makes a higher high but has a lower low (Indecision candle)
    # 3. Third candle is bearish (Close < Open) and closes below the first candle's open
    if first_close > first_open and \
       second_high > first_high and second_low < first_low and \
       third_close < third_open and third_close < first_open:
        return 1  # Sell Signal

    return 0  # No signal

def engulfing_pattern_signal(df, current_candle):
    """
    Identifies Engulfing Pattern for buy (2) or sell (1) signals.
    """
    current_pos = df.index.get_loc(current_candle)

    if current_pos < 1:
        return 0  # Not enough historical data

    prev_open = df['Open'].iloc[current_pos - 1]
    prev_close = df['Close'].iloc[current_pos - 1]
    current_open = df['Open'].iloc[current_pos]
    current_close = df['Close'].iloc[current_pos]

    # Bullish Engulfing
    if prev_close < prev_open and current_close > current_open and current_close > prev_open and current_open < prev_close:
        return 2  # Buy Signal

    # Bearish Engulfing
    if prev_close > prev_open and current_close < current_open and current_close < prev_open and current_open > prev_close:
        return 1  # Sell Signal

    return 0

def pin_bar_signal(df, current_candle):
    """
    Identifies a Pin Bar (long wick candle) for potential buy (2) or sell (1) signals.
    Returns 0 if no valid Pin Bar is found.
    """
    if current_candle not in df.index:
        return 0  # If the index does not exist, return 0

    open_price = df.loc[current_candle, 'Open']
    close_price = df.loc[current_candle, 'Close']
    high_price = df.loc[current_candle, 'High']
    low_price = df.loc[current_candle, 'Low']

    candle_body = abs(close_price - open_price)
    upper_wick = high_price - max(open_price, close_price)
    lower_wick = min(open_price, close_price) - low_price

    # Improved criteria:
    # 1. The long wick must be at least 2x the body
    # 2. The opposite wick must be much smaller
    # 3. The body should be small relative to the total range

    # Bullish Pin Bar (Hammer) - Long Lower Wick
    if lower_wick >= candle_body * 2 and upper_wick <= candle_body * 0.5 and close_price > open_price:
        return 2  # BUY Signal

    # Bearish Pin Bar (Shooting Star) - Long Upper Wick
    if upper_wick >= candle_body * 2 and lower_wick <= candle_body * 0.5 and close_price < open_price:
        return 1  # SELL Signal

    return 0  # No valid signal

def morning_evening_star_signal(df, current_candle):
    """
    Identifies Morning Star (Buy - 2) and Evening Star (Sell - 1).
    """
    current_pos = df.index.get_loc(current_candle)

    if current_pos < 2:
        return 0  # Not enough historical data

    # First Candle
    first_open = df['Open'].iloc[current_pos - 2]
    first_close = df['Close'].iloc[current_pos - 2]

    # Second Candle (Indecision)
    second_open = df['Open'].iloc[current_pos - 1]
    second_close = df['Close'].iloc[current_pos - 1]

    # Third Candle
    third_open = df['Open'].iloc[current_pos]
    third_close = df['Close'].iloc[current_pos]

    # Bullish Morning Star
    if first_close < first_open and third_close > third_open and third_close > (first_open + first_close) / 2:
        return 2  # Buy Signal

    # Bearish Evening Star
    if first_close > first_open and third_close < third_open and third_close < (first_open + first_close) / 2:
        return 1  # Sell Signal

    return 0
def shooting_star_hammer_signal(df, current_candle):
    """
    Identifies a Shooting Star (bearish reversal) or Hammer (bullish reversal).
    Returns:
        - 1 for a Shooting Star (SELL signal)
        - 2 for a Hammer (BUY signal)
        - 0 if no valid signal is found.
    """
    if current_candle not in df.index:
        return 0  # If the index does not exist, return 0

    open_price = df.loc[current_candle, 'Open']
    close_price = df.loc[current_candle, 'Close']
    high_price = df.loc[current_candle, 'High']
    low_price = df.loc[current_candle, 'Low']

    candle_body = abs(close_price - open_price)
    upper_wick = high_price - max(open_price, close_price)
    lower_wick = min(open_price, close_price) - low_price

    # Improved Shooting Star (Bearish)
    if (
        upper_wick >= candle_body * 2 and  # Long upper wick (>= 2x body)
        lower_wick <= candle_body * 0.3 and  # Small lower wick (<= 30% of body)
        close_price < open_price  # Closes lower than it opened (bearish)
    ):
        return 1  # SELL Signal

    # Improved Hammer (Bullish)
    if (
        lower_wick >= candle_body * 2 and  # Long lower wick (>= 2x body)
        upper_wick <= candle_body * 0.3 and  # Small upper wick (<= 30% of body)
        close_price > open_price  # Closes higher than it opened (bullish)
    ):
        return 2  # BUY Signal

    return 0  # No valid signal

def hammer_signal(df, current_candle):
    """
    Identifies Hammer pattern for buy (2) or sell (1) signals.
    """
    current_pos = df.index.get_loc(current_candle)

    if current_pos < 1:
        return 0  # Not enough historical data

    open_price = df['Open'].iloc[current_pos]
    close_price = df['Close'].iloc[current_pos]
    high_price = df['High'].iloc[current_pos]
    low_price = df['Low'].iloc[current_pos]

    body = abs(close_price - open_price)
    lower_wick = open_price - low_price if close_price > open_price else close_price - low_price
    upper_wick = high_price - close_price if close_price > open_price else high_price - open_price

    # Bullish Hammer: Small body, long lower wick, almost no upper wick
    if lower_wick > 2 * body and upper_wick < 0.3 * body:
        return 2  # Buy Signal

    # Bearish Exit: If price drops below the Hammer low
    if close_price < low_price:
        return 1  # Sell Signal

    return 0



def inverted_hammer_signal(df, current_candle):
    """
    Identifies Inverted Hammer (Buy - 2).
    """
    current_pos = df.index.get_loc(current_candle)

    if current_pos < 1:
        return 0  # Not enough historical data

    # Current Candle
    open_price = df['Open'].iloc[current_pos]
    close_price = df['Close'].iloc[current_pos]
    high_price = df['High'].iloc[current_pos]
    low_price = df['Low'].iloc[current_pos]

    body = abs(close_price - open_price)
    upper_wick = high_price - close_price if close_price > open_price else high_price - open_price
    lower_wick = open_price - low_price if close_price > open_price else close_price - low_price

    # Inverted Hammer Condition: Small body, long upper wick, almost no lower wick
    if upper_wick > 2 * body and lower_wick < 0.3 * body:
        return 2  # Buy Signal

    return 0


def doji_signal_strategy_v1(df, current_p, trend_period=5):
    """
    Generates trading signals based on a Doji and the preceding trend,
    using the difference between closing prices.

    Args:
        df (pandas.DataFrame): DataFrame with OHLC data.
        current_p (int): Index of the current data point.
        trend_period (int): Number of candles to consider for the trend.

    Returns:
        int: Trading signal (2 = Buy, 1 = Sell Short, 0 = Hold, -1 = Exit Short, -2 = Exit Long).
    """
    current_pos = df.index.get_loc(current_p)

    if current_pos < trend_period:
        return 0

    current_candle = df.iloc[current_pos]

    # Closing prices for the period
    first_close = df.iloc[current_pos - trend_period]['Close']
    last_close = df.iloc[current_pos - 1]['Close']  # Use the close of the candle before the Doji

    # Trend definition based on the difference between closing prices
    is_uptrend = last_close > first_close
    is_downtrend = last_close < first_close

    # Condition to identify a Doji
    is_doji = abs(current_candle['Open'] - current_candle['Close']) < (
                current_candle['High'] - current_candle['Low']) * 0.1

    # 📈 Long Entry Signal (Buy)
    if is_doji and is_downtrend:
        return 2

    # 📉 Short Entry Signal (Sell)
    if is_doji and is_uptrend:
        return 1

    # 🚪 Exit Signals
    # Exit logic can be more complex, but for this basic strategy,
    # we exit if the trend reverses.
    if is_uptrend:
        return -2  # Exit a short position

    if is_downtrend:
        return -1  # Exit a long position

    return 0  # Hold

def combined_signal(df, current_candle):
    """
    Combines signals from various candlestick strategies, excluding itself.

    Args:
        df (pd.DataFrame): DataFrame containing candlestick data.
        current_candle (index): Index of the current candle.
        candlestick_strategies (list): List of candlestick strategy functions.

    Returns:
        int: 2 for buy signal, 1 for sell signal, 0 for no signal.
    """
    # Create a new list excluding combined_signal
    strategies = [func for func in candlestick_strategies if func != combined_signal]

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


def aaa():
    """

    hammer_signal,
    inverted_hammer_signal
    """
    q=1


"""
dax_total_signal,
    dax_momentum_signal,
    inside_bar_breakout_signal,
    three_bar_reversal_signal,
    engulfing_pattern_signal,
    pin_bar_signal,
    morning_evening_star_signal,
    shooting_star_hammer_signal,
    hammer_signal,
"""

# List of all signal functions
candlestick_strategies = [
    dax_momentum_signal,
    inside_bar_breakout_signal,
    three_bar_reversal_signal,
    engulfing_pattern_signal,
    pin_bar_signal,
    morning_evening_star_signal,
    shooting_star_hammer_signal,
    hammer_signal,
    inverted_hammer_signal,
    doji_signal_strategy_v1,
    combined_signal
]
