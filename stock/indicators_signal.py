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
indicators_strategy =[
    rsi_bollinger_macd_total_signal_v1,
    rsi_bollinger_macd_total_signal_v2,
    mean_reversion_signal_v1
]