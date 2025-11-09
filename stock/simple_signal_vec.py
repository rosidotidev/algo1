import pandas as pd
import numpy as np
import stock.strategy_util as stru

def ema_pullback_strategy_b1(df: pd.DataFrame,
                             short_ema: int = 8,
                             long_ema: int = 13,
                             min_move: float = 0.001,
                             min_trend_gap: float = 0.06,
                             cooloff: int = -3) -> pd.Series:
    """
    EMA Pullback Strategy - Vectorized Pure Version
    -----------------------------------------------
    Migliorie:
    - Cross EMA8 validi solo con movimento significativo
    - Filtro trend: gap minimo 0.5–1% tra EMA8 e EMA21
    - Cooloff vettoriale (nessun for)
    - Nessun indicatore extra (solo prezzo + EMA)
    """

    close = df['Close']
    open_ = df['Open']

    ema_short = close.ewm(span=short_ema, adjust=False).mean()
    ema_long = close.ewm(span=long_ema, adjust=False).mean()

    # Movimento minimo intraday per considerare un cross valido
    price_move = abs(close - open_) / open_

    cross_ema_short = (
        (((open_ > ema_short) & (close < ema_short)) |
         ((open_ < ema_short) & (close > ema_short)))
        & (price_move > min_move)
    )

    # Filtri trend con gap minimo
    trend_up = ema_short > ema_long * (1 + min_trend_gap)
    trend_down = ema_long > ema_short * (1 + min_trend_gap)

    # Condizioni di ingresso
    long_condition = (
        trend_up &
        cross_ema_short &
        (open_ > ema_long) &
        (close > ema_long) &
        (close.shift(1) > ema_short.shift(1))
    )

    short_condition = (
        trend_down &
        cross_ema_short &
        (open_ < ema_long) &
        (close < ema_long) &
        (close.shift(1) < ema_short.shift(1))
    )

    # Condizioni di uscita
    exit_long_condition = (close < ema_short)
    exit_short_condition = (close > ema_short)

    # Creazione serie segnali base
    signals = pd.Series(0, index=df.index, dtype='int8')
    signals[exit_long_condition] = -2
    signals[exit_short_condition] = -1
    signals[long_condition] = 2
    signals[short_condition] = 1

    # COOL-OFF vettoriale: ignora segnali entro N barre dal precedente
    if cooloff > 0:
        entry_mask = signals.isin([1, 2])
        last_entry = entry_mask.astype(float).replace(0.0, np.nan)
        last_entry = last_entry * np.arange(len(entry_mask))  # indice progressivo
        last_valid = last_entry.ffill()
        dist = np.arange(len(entry_mask)) - last_valid
        # Cancella segnali troppo vicini
        signals[(entry_mask) & (dist <= cooloff)] = 0
    df['signals']=signals
    df['ema_short']=ema_short
    df['ema_long']=ema_long
    return signals

def ema_pullback_strategy(df: pd.DataFrame,
                          short_ema: int = 8,
                          long_ema: int = 21,
                          min_move: float = 0.02,
                          min_trend_gap: float = 0.06,
                          lookback_trend: int = 5) -> pd.Series:
    """
    EMA Pullback Strategy - Trend Persistence Version
    ------------------------------------------------
    Logica aggiornata:
    - Conferma trend tramite EMA short rispetto al valore di lookback_trend barre fa
    - Cross valido solo se la candela corrente 'sfonda' la EMA short
    - Filtri su gap di trend e movimento minimo intraday
    """

    close = df['Close']
    open_ = df['Open']
    high = df['High']
    low = df['Low']

    ema_short = close.ewm(span=short_ema, adjust=False).mean()
    ema_long = close.ewm(span=long_ema, adjust=False).mean()

    # Movimento minimo intraday per considerare un cross valido
    #price_move = abs(close - open_) / open_

    # Trend filter con persistenza nel tempo
    ema_short_prev = ema_short.shift(lookback_trend)
    persistent_up = ema_short > ema_short_prev * (1 + min_trend_gap)
    persistent_down = ema_short < ema_short_prev * (1 + min_trend_gap)

    # Filtri trend generali con gap minimo
    trend_up = (ema_short > ema_long * (1 + min_trend_gap)) & persistent_up
    #trend_up = persistent_up
    trend_down = (ema_long > ema_short * (1 + min_trend_gap)) & persistent_down
    #trend_down = persistent_down

    # Cross EMA short (candela che "sfonda" la EMA)
    cross = ((low < ema_short) & (high > ema_short)) |((low > ema_short) & (high < ema_short))#& (price_move > min_move)

    # Condizioni di ingresso
    long_condition = trend_up & cross & (close > ema_long)
    short_condition = trend_down & cross & (close < ema_long)

    # Serie segnali
    signals = pd.Series(0, index=df.index, dtype='int8')
    signals[long_condition] = 2
    signals[short_condition] = 1

    # Aggiunta colonne di debug utili per analisi
    df['ema_short'] = ema_short
    df['ema_long'] = ema_long
    df['signals'] = signals
    df['cross'] = cross

    df['trend_up'] = trend_up
    df['trend_down']=trend_down

    return signals



def ema_pullback_simple_strategy(
    df: pd.DataFrame,
    short_ema: int = 8,
    long_ema: int = 21
) -> pd.Series:
    """
    Strategy:
    - Trend filter: EMA8 vs EMA21
    - Entry when price crosses EMA8 without touching EMA21
    - Returns:
        2 = Buy
        1 = Sell/Short
        0 = Hold
    """

    close = df['Close']
    open_ = df['Open']

    ema_short = close.ewm(span=short_ema, adjust=False).mean()
    ema_long = close.ewm(span=long_ema, adjust=False).mean()


    # Crossing EMA8 → Open/Close on opposite sides
    cross_ema_short = (
        ((open_ > ema_short) & (close < ema_short)) |
        ((open_ < ema_short) & (close > ema_short))

    )

    # LONG conditions
    long_condition = (
        (ema_short > ema_long) &
        cross_ema_short &
        (open_ > ema_long) &
        (close > ema_long ) &
        close.shift(1) > ema_short.shift(1)
    )

    # SHORT conditions
    short_condition = (
        (ema_long > ema_short) &
        cross_ema_short &
        (open_ < ema_long) &
        (close < ema_long) &
        close.shift(1) < ema_short.shift(1)
    )
    exit_long_condition=ema_long > ema_short
    exit_short_condition = ema_long < ema_short
    signals = pd.Series(0, index=df.index, dtype='int8')
    signals[exit_long_condition] = -2
    signals[exit_short_condition] = -1
    signals[long_condition] = 2
    signals[short_condition] = 1

    return signals



def squeeze_ema_strategy(df: pd.DataFrame, short_ema: int = 10, squeez_back_period: int = 5) -> pd.Series:
    """
    Strategia semplice basata su:
    - TTM Squeeze (default params)
    - EMA short per confermare breakout
    """
    # --- Squeeze ---
    squeeze_df = stru.compute_squeeze(df)  # usa parametri di default

    # --- EMA ---
    ema = df['Close'].ewm(span=short_ema, adjust=False).mean()

    # --- Condizioni LONG ---
    long_condition = (
        squeeze_df['squeeze_on'].shift(squeez_back_period) &   # squeeze attivo qualche periodo fa
        squeeze_df['squeeze_off'].shift(squeez_back_period-1) & # squeeze rilasciato subito dopo
        (df['Close'] > ema)                                                   # conferma trend rialzista
    )


    # --- Condizioni SHORT ---
    short_condition = (
        squeeze_df['squeeze_on'].shift(squeez_back_period) &
        squeeze_df['squeeze_off'].shift(squeez_back_period-1) &
        (df['Close'] < ema)                                                   # conferma trend ribassista
    )
    df['squeeze_on'] = squeeze_df['squeeze_on']
    df['squeeze_off'] = squeeze_df['squeeze_off']
    df['ema'] = ema
    df['long_condition'] = long_condition
    df['short_condition'] = short_condition
    # --- Segnali ---
    signals = pd.Series(0, index=df.index, dtype='int8')
    signals[long_condition] = 2
    signals[short_condition] = 1

    return signals

def squeeze_ema_10_20(df: pd.DataFrame):
    return squeeze_ema_strategy(df,10,20)

def squeeze_ema_15_25(df: pd.DataFrame):
    return squeeze_ema_strategy(df,15,25)

def volume_trend_ema_threshold(df: pd.DataFrame,
                               lookback: int = 20,
                               ema_len: int = 8,
                               vol_threshold: float = 0.20) -> pd.Series:
    """
    Volume Breakout + EMA Filter + Max-Volume Threshold
    ---------------------------------------------------
    Logic:
    - Volume must also exceed (rolling_max_volume * (1 + vol_threshold))
    - EMA used only as directional trend filter
    - Returns only entry signals (2 long, 1 short)
    """

    close = df['Close']
    volume = df['Volume']

    # EMA for directional trend filter
    ema_short = close.ewm(span=ema_len, adjust=False).mean()

    # Rolling max volume of the period (shifted to exclude today)
    vol_max = volume.rolling(lookback).max().shift(1)

    # Threshold: volume must exceed max_volume * (1 + threshold)
    vol_threshold_cond = volume > (vol_max * (1 + vol_threshold))

    # Output signal structure
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Directional conditions
    long_condition  = vol_threshold_cond & (close > ema_short)
    short_condition = vol_threshold_cond & (close < ema_short)

    signals[long_condition] = 2
    signals[short_condition] = 1

    return signals

def volume_donchian_breakout(df: pd.DataFrame,
                             lookback_vol: int = 20,
                             lookback_price: int = 50,
                             vol_threshold: float = 0.20) -> pd.Series:
    """
    Volume Breakout + Donchian Channel Filter (Non-Intraday Strategy)
    -----------------------------------------------------------------
    Logic:
    - Volume must exceed (rolling_max_volume * (1 + vol_threshold))
    - Entry signal only if price closes beyond the high/low of the long-term
      Donchian Channel (representing significant S/R).
    - Returns entry signals (2 for long, 1 for short).
    """

    close = df['Close']
    high = df['High']
    low = df['Low']

    # --- 1. Define Key Price Levels (Long-Term Price Channel) ---
    # We use the Donchian Channel (Max High / Min Low)
    # lookback_price (e.g., 50 periods) defines the S/R window

    # Resistance (R): Highest high observed over the last lookback_price periods
    R_level = high.rolling(lookback_price).max().shift(1)

    # Support (S): Lowest low observed over the last lookback_price periods
    S_level = low.rolling(lookback_price).min().shift(1)

    # --- 2. Volume Breakout Condition ---
    volume = df['Volume']
    # Rolling max volume of the period (shifted to exclude today)
    vol_max = volume.rolling(lookback_vol).max().shift(1)

    # Threshold: volume must exceed max_volume * (1 + threshold)
    vol_threshold_cond = volume > (vol_max * (1 + vol_threshold))

    # --- 3. Signal Generation ---
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Directional Conditions (Volume Breakout + Price breaking the Key Level)

    # Long: Volume Spike AND Close is above the Resistance R_level (upside breakout)
    long_condition = vol_threshold_cond & (close > R_level)

    # Short: Volume Spike AND Close is below the Support S_level (downside breakout)
    short_condition = vol_threshold_cond & (close < S_level)

    signals[long_condition] = 2
    signals[short_condition] = 1

    return signals

def volume_donchian_breakout_10_25_20(df: pd.DataFrame) -> pd.Series:
    return volume_donchian_breakout(df,10,25,0.2)

def volume_donchian_breakout_15_30_20(df: pd.DataFrame) -> pd.Series:
    return volume_donchian_breakout(df,15,30,0.2)

def volume_trend_basic(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    close = df['Close']
    volume = df['Volume']

    # Calcolo della media mobile semplice
    sma = close.rolling(lookback).mean()
    vw = stru.vwap(df, lookback)
    # Condizione di "cross volume breakout"
    vol_cross = stru.cross_over_max(volume, lookback)

    signals = pd.Series(0, index=df.index, dtype='int8')

    # Entry SOLO se c'è stato CROSS del volume oltre il massimo precedente
    long_condition = vol_cross & (close > sma) & (close > vw) & (vw > sma)
    short_condition = vol_cross & (close < sma) & (close < vw) & (vw < sma)

    signals[long_condition] = 2
    signals[short_condition] = 1

    return signals



def volume_trend_10(df: pd.DataFrame):
    return volume_trend_basic(df, lookback=10)

def volume_trend_30(df: pd.DataFrame):
    return volume_trend_basic(df, lookback=30)


def momentum_vwap_vma_strategy(df: pd.DataFrame,
                                        momentum_length: int = 20,
                                        smooth_length: int = 5,
                                        vwap_length: int = 20,
                                        vma_length: int = 20) -> pd.Series:

    hist = stru.compute_momentum_hist(df, momentum_length, smooth_length)
    vwap_val = stru.vwap(df, vwap_length)
    vma_val = stru.vma(df, vma_length)

    close = df["Close"]
    volume = df["Volume"]

    # Momentum crescente = conferma trend
    momentum_up = hist > hist.shift(1)

    signals = pd.Series(0, index=df.index, dtype='int8')

    # ✅ LONG
    long_condition = (
        (hist > 0) &
        momentum_up &
        (stru.cross_over(close,vwap_val)) &
        (volume > vma_val) &
        (close > close.shift(1)
        )
    )
    signals[long_condition] = 2

    # ✅ SHORT
    short_condition = (
        (hist < 0) &
        (~momentum_up) &
        (stru.cross_under(close,  vwap_val)) &
        (volume > vma_val) &
        (close < close.shift(1)
        )
    )
    signals[short_condition] = 1

    return signals

def ema_vwap_vectorized(df: pd.DataFrame,
                            emaShortPeriod: int = 13,
                            emaLongPeriod: int = 18) -> pd.Series:
    """
    Vectorized EMA/VWAP/SMA crossover strategy.
    Generates trading signals based on entry conditions only:
    2 = Long entry
    1 = Short entry
    0 = No action (Hold/Flat/Exit handled by backtester)

    Args:
        df (pd.DataFrame): Must include 'High', 'Low', 'Close', 'Volume'
        emaShortPeriod (int): Fast EMA for entry
        emaLongPeriod (int): Slow EMA for entry
        smaPeriod (int): SMA filter for entry

    Returns:
        pd.Series: Trading signals (2 = Buy, 1 = Short, 0 = Hold)
    """

    close = df["Close"]
    # === Indicatori ===
    emaShort = close.ewm(span=emaShortPeriod, adjust=False).mean()
    emaLong = close.ewm(span=emaLongPeriod, adjust=False).mean()

    # VWAP vectorized
    vwap = stru.vwap(df)

    # === Segnali ===
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Entry Long (Buy)
    long_condition = (
        (emaShort > emaLong) &
        (emaShort.shift(1) <= emaLong.shift(1)) &
        (close > vwap)
    )
    signals[long_condition] = 2

    # Entry Short (Sell Short)
    short_condition = (
        (emaShort < emaLong) &
        (emaShort.shift(1) >= emaLong.shift(1)) &
        (close < vwap)
    )
    signals[short_condition] = 1

    return signals

def ema_vwap_sma_vectorized(df: pd.DataFrame,
                            emaShortPeriod: int = 17,
                            emaLongPeriod: int = 31,
                            smaPeriod: int = 69) -> pd.Series:
    """
    Vectorized EMA/VWAP/SMA crossover strategy.
    Generates trading signals based on entry conditions only:
    2 = Long entry
    1 = Short entry
    0 = No action (Hold/Flat/Exit handled by backtester)

    Args:
        df (pd.DataFrame): Must include 'High', 'Low', 'Close', 'Volume'
        emaShortPeriod (int): Fast EMA for entry
        emaLongPeriod (int): Slow EMA for entry
        smaPeriod (int): SMA filter for entry

    Returns:
        pd.Series: Trading signals (2 = Buy, 1 = Short, 0 = Hold)
    """

    close = df["Close"]
    # === Indicatori ===
    emaShort = close.ewm(span=emaShortPeriod, adjust=False).mean()
    emaLong = close.ewm(span=emaLongPeriod, adjust=False).mean()
    smaLong = close.rolling(smaPeriod).mean()

    # VWAP vectorized
    vwap = stru.vwap(df)

    # === Segnali ===
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Entry Long (Buy)
    long_condition = (
        (emaShort > emaLong) &
        (emaShort.shift(1) <= emaLong.shift(1)) &
        (close > vwap) &
        (close > smaLong)
    )
    signals[long_condition] = 2

    # Entry Short (Sell Short)
    short_condition = (
        (emaShort < emaLong) &
        (emaShort.shift(1) >= emaLong.shift(1)) &
        (close < vwap) &
        (close < smaLong)
    )
    signals[short_condition] = 1

    return signals

def ema_vwap_sma_15_30_50(df: pd.DataFrame):
    return ema_vwap_sma_vectorized(df,15,30,50)

def ema_vwap_sma_5_10_20(df: pd.DataFrame):
    return ema_vwap_sma_vectorized(df,5,10,20)

def ttm_squeeze_strategy(df: pd.DataFrame,
                         bb_length: int = 20, keltner_length: int = 20,
                         bb_mult: float = 2.0, keltner_mult: float = 1.5,
                         momentum_length: int = 20, smooth_length: int = 5) -> pd.Series:
    """
    Full TTM Squeeze strategy:
    - Entry: when squeeze releases (on→off)
    - Direction: histogram sign
    - Exit: when histogram changes sign
    """

    squeeze = stru.compute_squeeze(df, bb_length, keltner_length, bb_mult, keltner_mult)
    hist = stru.compute_momentum_hist(df, momentum_length, smooth_length)

    squeeze_release = squeeze['squeeze_on'].shift(1) & squeeze['squeeze_off']

    # Initialize signals
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Entries
    signals[squeeze_release & (hist > 0)] = 2   # Buy
    signals[squeeze_release & (hist < 0)] = 1   # Sell

    # Exits → momentum crosses zero (change of direction)
    hist_prev = hist.shift(1)
    cross_down = (hist_prev > 0) & (hist < 0)
    cross_up = (hist_prev < 0) & (hist > 0)

    signals[cross_down] = -2  # exit from Buy
    signals[cross_up] = -1    # exit from Sell

    return signals



def parab_rev_vectorized(df, short=5, medium=10, long=20):
    """
    parabol-reverse strategy (TF).

    Logic:
        - Long (2) when short MMA > medium MMA and short MMA < long MMA (uptrend pullback)
        - Short (1) when short MMA < medium MMA and short MMA < long MMA (downtrend pullback)
        - Hold (0) otherwise

    Args:
        df (pd.DataFrame): Must contain 'Close' column
        short (int): period for short moving average
        medium (int): period for medium moving average
        long (int): period for long moving average

    Returns:
        pd.Series: signals (2=Long, 1=Short, 0=Hold)
    """
    # Calculate moving averages
    df['mma_short'] = df['Close'].rolling(short).mean()
    df['mma_medium'] = df['Close'].rolling(medium).mean()
    df['mma_long'] = df['Close'].rolling(long).mean()

    # Define entry conditions
    long_signal = (df['mma_short'] > df['mma_medium']) & (df['mma_medium'] < df['mma_long'])
    short_signal = (df['mma_short'] < df['mma_medium']) & (df['mma_medium'] > df['mma_long'])

    # Create signals series
    signals = pd.Series(0, index=df.index, dtype='int8')
    signals[long_signal] = 2
    signals[short_signal] = 1

    # Clean temporary columns
    df.drop(columns=['mma_short', 'mma_medium', 'mma_long'], inplace=True)

    return signals

def parab_rev_8_16_32_vectorized(df):
    return parab_rev_vectorized(df, short=8, medium=16, long=32)

def parab_rev_3_6_15_vectorized(df):
    return parab_rev_vectorized(df, short=3, medium=6, long=15)


def vwap_trading_strategy_threshold(df: pd.DataFrame, lookback: int = 20, threshold: float = 0.005) -> pd.Series:
    """
    Rolling VWAP strategy with a threshold to reduce false signals.
    Threshold in fraction of price (0.005 = 0.5%).
    """
    vwap = stru.vwap(df,lookback)

    signals = pd.Series(0, index=df.index, dtype='int8')
    signals[df['Close'] > vwap * (1 + threshold)] = 1  # Sell
    signals[df['Close'] < vwap * (1 - threshold)] = 2  # Buy

    return signals

def vwap_trading_strategy_cross(df: pd.DataFrame, lookback: int = 20, threshold: float = 0.005) -> pd.Series:
    """
    VWAP strategy with cross detection and exit signals.
    Buy/Sell only when a cross occurs, exit when price returns to VWAP.
    """
    vwap = stru.vwap(df,lookback)

    upper_band = vwap * (1 + threshold)
    lower_band = vwap * (1 - threshold)

    # Previous day conditions
    prev_close = df['Close'].shift(1)
    prev_above_upper = prev_close > upper_band.shift(1)
    prev_below_lower = prev_close < lower_band.shift(1)

    # Current day conditions
    above_upper = df['Close'] > upper_band
    below_lower = df['Close'] < lower_band

    # Initialize signals
    signals = pd.Series(0, index=df.index, dtype='int8')

    # --- Entry signals (cross from opposite side) ---
    buy_cross = (~prev_below_lower) & below_lower        # just crossed below lower band
    sell_cross = (~prev_above_upper) & above_upper        # just crossed above upper band

    signals[buy_cross] = 2
    signals[sell_cross] = 1

    # --- Exit signals (return to VWAP center) ---
    prev_below_vwap = prev_close < vwap.shift(1)
    prev_above_vwap = prev_close > vwap.shift(1)

    cross_up_to_vwap = prev_below_vwap & (df['Close'] >= vwap)
    cross_down_to_vwap = prev_above_vwap & (df['Close'] <= vwap)

    signals[cross_up_to_vwap] = -2   # Exit Buy
    signals[cross_down_to_vwap] = -1 # Exit Sell

    return signals

def vwap_trading_strategy_cross_20_10(df: pd.DataFrame) -> pd.Series:
    return vwap_trading_strategy_cross(df, lookback= 20, threshold= 0.015)

def vwap_trading_strategy_cross_30_20(df: pd.DataFrame) -> pd.Series:
    return vwap_trading_strategy_cross(df, lookback= 30, threshold= 0.020)

def weekly_breakout_2_1(df):
    return weekly_breakout_vectorized(df,2,1)

def weekly_breakout_1_2(df):
    return weekly_breakout_vectorized(df,1,2)

def weekly_breakout_2_2(df):
    return weekly_breakout_vectorized(df,2,2)

def weekly_breakout_1_3(df):
    return weekly_breakout_vectorized(df,1,3)

def weekly_breakout_2_3(df):
    return weekly_breakout_vectorized(df,2,3)

def weekly_breakout_vectorized(df, lookback_weeks=1, entry_day=1):
    """
    Fully vectorized weekly breakout strategy with correct handling of long and short positions.

    Logic:
        - On entry_day (default Tuesday = 1):
            - 2 (Buy)  if previous day's close > previous N weeks' high
            - 1 (Sell) if previous day's close < previous N weeks' low
        - Exit signals:
            - -2 (Exit Buy)  if previously long and no new long
            - -1 (Exit Sell) if previously short and no new short
        - 0 otherwise
    """
    price_col = "Close"

    # 1. Compute weekly highs and lows
    weekly = df.groupby(df.index.to_period("W"))[price_col].agg(["max", "min"])
    weekly["high"] = weekly["max"].rolling(lookback_weeks).max().shift(1)
    weekly["low"] = weekly["min"].rolling(lookback_weeks).min().shift(1)

    # 2. Map weekly levels to daily index
    week_periods = df.index.to_period("W")
    prev_high = week_periods.map(weekly["high"].to_dict())
    prev_low = week_periods.map(weekly["low"].to_dict())

    # 3. Previous day's close for evaluating today's entry
    prev_close = df[price_col].shift(1)
    entry_flag = df.index.dayofweek == entry_day

    # 4. Vectorized entry signals
    long_entry  = entry_flag & (prev_close > prev_high)
    short_entry = entry_flag & (prev_close < prev_low)

    # 5. Initialize signal series
    signals = pd.Series(0, index=df.index, dtype="int8")
    signals[long_entry]  = 2
    signals[short_entry] = 1

    # 6. Track previous position state separately for long and short
    # Create boolean masks
    prev_long  = signals.shift(1) == 2
    prev_short = signals.shift(1) == 1

    # 7. Exit conditions (only where no new entry is present)
    exit_long  = entry_flag & prev_long & (~long_entry)
    exit_short = entry_flag & prev_short & (~short_entry)

    signals[exit_long & (signals == 0)]   = -2
    signals[exit_short & (signals == 0)]  = -1

    return signals


def keltner_rev_vectorized(df: pd.DataFrame, lookback_periods: int = 20, atr_multiplier: float = 2.0) -> pd.Series:
    original_signals = keltner_tf_vectorized(df.copy(),lookback_periods, atr_multiplier)
    # Invert the signals using np.where in a vectorized manner.
    # If the original signal is 2 (buy), the new signal is 1 (short).
    # If the original signal is 1 (sell), the new signal is 2 (buy to cover).
    # If the original signal is 0, it remains 0.
    inverted_signals = np.where(original_signals == 2, 1,
                                np.where(original_signals == 1, 2, 0))

    # Return a pandas Series with the correct index.
    return pd.Series(inverted_signals, index=df.index, dtype='int8')

def keltner_rev_15_1_5(df):
    return keltner_rev_vectorized(df,15,1.5)

def keltner_rev_50_2_5(df):
    return keltner_rev_vectorized(df,50,2.5)


def keltner_tf_vectorized(df: pd.DataFrame, lookback_periods: int = 20, atr_multiplier: float = 2.0) -> pd.Series:
    """:
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

def keltner_tf_15_1_5(df):
    return keltner_tf_vectorized(df,15,1.5)

def keltner_tf_50_2_5(df):
    return keltner_tf_vectorized(df,50,2.5)


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

def donchian_inv_with_ma_10_30(df):
    return donchian_inv_with_ma_filter(df,10,30)

def donchian_inv_with_ma_15_40(df):
    return donchian_inv_with_ma_filter(df,15,40)

def donchian_inv_with_ma_filter(df, lookback=20, ma_window=50):
    """
    Inverts the logic of the filled_bar_vectorized function to generate signals
    for a short strategy.

    A buy signal from the original function becomes a short entry signal (1).
    A sell signal from the original function becomes a short exit signal (2).

    Args:
        df (pd.DataFrame): The DataFrame with historical OHLC data.
        ratio (float): The threshold for the body-to-range ratio of the candle.

    Returns:
        pd.Series: A Series of inverted trading signals
                   (1 for short entry, 2 for short exit, 0 for no signal).
    """

    # Call the original function to get the base signals.
    # A copy of the DataFrame is passed to avoid modifying the original.
    original_signals = donchian_breakout_with_ma_filter(df.copy(),lookback, ma_window)

    # Invert the signals using np.where in a vectorized manner.
    # If the original signal is 2 (buy), the new signal is 1 (short).
    # If the original signal is 1 (sell), the new signal is 2 (buy to cover).
    # If the original signal is 0, it remains 0.
    inverted_signals = np.where(original_signals == 2, 1,
                                np.where(original_signals == 1, 2, 0))

    # Return a pandas Series with the correct index.
    return pd.Series(inverted_signals, index=df.index, dtype='int8')

def retracement_tf_vectorized(df, short=5, medium=10, long=20):
    """
    Trend-following retracement strategy (TF).

    Logic:
        - Long (2) when short MMA < medium MMA and short MMA > long MMA (uptrend pullback)
        - Short (1) when short MMA > medium MMA and short MMA < long MMA (downtrend pullback)
        - Hold (0) otherwise

    Args:
        df (pd.DataFrame): Must contain 'Close' column
        short (int): period for short moving average
        medium (int): period for medium moving average
        long (int): period for long moving average

    Returns:
        pd.Series: signals (2=Long, 1=Short, 0=Hold)
    """
    # Calculate moving averages
    df['mma_short'] = df['Close'].rolling(short).mean()
    df['mma_medium'] = df['Close'].rolling(medium).mean()
    df['mma_long'] = df['Close'].rolling(long).mean()

    # Define entry conditions
    long_signal = (df['mma_short'] < df['mma_medium']) & (df['mma_short'] > df['mma_long'])
    short_signal = (df['mma_short'] > df['mma_medium']) & (df['mma_short'] < df['mma_long'])

    # Create signals series
    signals = pd.Series(0, index=df.index, dtype='int8')
    signals[long_signal] = 2
    signals[short_signal] = 1

    # Clean temporary columns
    df.drop(columns=['mma_short', 'mma_medium', 'mma_long'], inplace=True)

    return signals


def retracement_rev_vectorized(df, short=5, medium=10, long=20):
    """
    Reversion retracement strategy (REVERSION).

    Logic:
        - Invert signals of the trend-following retracement strategy:
            - Long becomes Short
            - Short becomes Long
        - Hold remains 0

    Args:
        df (pd.DataFrame): Must contain 'Close' column
        short (int): period for short moving average
        medium (int): period for medium moving average
        long (int): period for long moving average

    Returns:
        pd.Series: signals (2=Long, 1=Short, 0=Hold)
    """
    # Get trend-following signals
    tf_signals = retracement_tf_vectorized(df.copy(), short, medium, long)

    # Invert the signals for reversion
    reversion_signals = tf_signals.map({2: 1, 1: 2, 0: 0}).astype('int8')

    return reversion_signals


def count_hammers(df):
    df['body'] = abs(df['Close'] - df['Open'])
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)

    hammer_condition = (df['lower_wick'] > 2 * df['body']) & \
                       (df['upper_wick'] < 0.3 * df['body'])
    return  hammer_condition.sum()

def count_inverted_hammers(df):
    # Vectorized calculation of candle body and wicks
    df['body'] = abs(df['Close'] - df['Open'])
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']

    # --- Inverted Hammer conditions ---
    inverted_hammer_condition = (df['upper_wick'] > 2 * df['body']) & \
                                (df['lower_wick'] < 0.3 * df['body'])
    return inverted_hammer_condition.sum()

def count_hammers_in_ticker(ticker):
    path_file=f"../../data/{ticker}.csv"
    df = pd.read_csv(path_file)
    return count_hammers(df)

def count_inverted_hammers_in_ticker(ticker):
    path_file=f"../../data/{ticker}.csv"
    df = pd.read_csv(path_file)
    return count_inverted_hammers(df)

def dax_total_signal_vectorized(df):
    """
    Generates a Series of trading signals for the entire DataFrame using a
    vectorized approach.

    Args:
        df (pd.DataFrame): The DataFrame with historical data.

    Returns:
        pd.Series: A Series of trading signals (2 for Buy, 1 for Sell, 0 for Hold).
    """
    # Create temporary columns for previous values using .shift()
    df['High_prev1'] = df['High'].shift(1)
    df['High_prev2'] = df['High'].shift(2)
    df['High_prev3'] = df['High'].shift(3)
    df['Low_prev1'] = df['Low'].shift(1)
    df['Low_prev2'] = df['Low'].shift(2)
    df['Low_prev3'] = df['Low'].shift(3)

    # Buy signal conditions
    buy_condition = (df['High'] > df['High_prev1']) & \
                    (df['High_prev1'] > df['Low_prev1']) & \
                    (df['Low_prev1'] > df['High_prev2']) & \
                    (df['High_prev2'] > df['Low_prev2']) & \
                    (df['Low_prev2'] > df['High_prev3']) & \
                    (df['High_prev3'] > df['Low_prev3']) & \
                    (df['Low_prev3'] > df['Low_prev2'])

    # Sell signal conditions (symmetrical to buy)
    sell_condition = (df['Low'] < df['Low_prev1']) & \
                     (df['Low_prev1'] < df['High_prev1']) & \
                     (df['High_prev1'] < df['Low_prev2']) & \
                     (df['Low_prev2'] < df['High_prev2']) & \
                     (df['High_prev2'] < df['Low_prev3']) & \
                     (df['Low_prev3'] < df['High_prev3']) & \
                     (df['High_prev3'] < df['High_prev2'])

    # Initialize a signal Series with a default value of 0 (Hold)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Apply signals based on the boolean masks
    signals[buy_condition] = 2  # Buy signal
    signals[sell_condition] = 1  # Sell signal

    # Remove temporary columns to keep the DataFrame clean
    df.drop(columns=['High_prev1', 'High_prev2', 'High_prev3',
                     'Low_prev1', 'Low_prev2', 'Low_prev3'],
            inplace=True)

    return signals

def dax_momentum_signal_vectorized(df):
    """
    Generates a Series of trading signals for the entire DataFrame
    using a vectorized approach based on momentum rules.

    Returns:
        A pandas Series of trading signals (2 for Buy, 1 for Sell, 0 for Hold).
    """
    # Create temporary columns for previous values using .shift()
    df['High_prev1'] = df['High'].shift(1)
    df['Low_prev1'] = df['Low'].shift(1)
    df['High_prev2'] = df['High'].shift(2)
    df['Low_prev2'] = df['Low'].shift(2)

    # Calculate range-based conditions
    df['Range'] = df['High'] - df['Low']
    buy_threshold = df['Low'] + df['Range'] * 0.7
    sell_threshold = df['High'] - df['Range'] * 0.7

    # Create boolean masks for each signal
    # Buy signal conditions
    buy_condition = (df['High'] > df['High_prev1']) & \
                    (df['Low_prev1'] > df['Low_prev2']) & \
                    (df['Close'] > buy_threshold)

    # Sell signal conditions
    sell_condition = (df['Low'] < df['Low_prev1']) & \
                     (df['High_prev1'] < df['High_prev2']) & \
                     (df['Close'] < sell_threshold)

    # Initialize a signal Series with a default value of 0 (Hold)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Apply signals using vectorized assignment
    signals[buy_condition] = 2  # Buy signal
    signals[sell_condition] = 1  # Sell signal

    # Cleanup: remove temporary columns to keep DataFrame clean
    df.drop(columns=['High_prev1', 'Low_prev1', 'High_prev2', 'Low_prev2', 'Range'], inplace=True)

    return signals


def inside_bar_breakout_signal_vectorized(df):
    """
    Generates a Series of trading signals for the entire DataFrame based on a
    correctly implemented vectorized Inside Bar Breakout strategy.

    Args:
        df (pd.DataFrame): The DataFrame with historical data.

    Returns:
        pd.Series: A Series of trading signals (2 for Buy, 1 for Sell, 0 for Hold).
    """
    # 1. Creazione di colonne temporanee per i dati passati.
    #    Questo sostituisce le chiamate .iloc[current_pos - 1] e .iloc[current_pos - 2]
    #    della versione non vettoriale.
    df['High_prev1'] = df['High'].shift(1)
    df['Low_prev1'] = df['Low'].shift(1)
    df['High_prev2'] = df['High'].shift(2)
    df['Low_prev2'] = df['Low'].shift(2)

    # 2. Condizione per identificare l'Inside Bar.
    #    La candela precedente (prev1) deve avere un massimo inferiore e un minimo superiore
    #    rispetto alla candela che la precede (prev2).
    #    Corrisponde a "if prev_high > df['High'].iloc[current_pos - 2] and prev_low < df['Low'].iloc[current_pos - 2]"
    #    nella versione originale.
    is_inside_bar = (df['High_prev1'] > df['High_prev2']) & (df['Low_prev1'] < df['Low_prev2'])

    # 3. Condizioni per il breakout sulla candela attuale.
    #    Queste condizioni corrispondono a "if current_high > prev_high" e "if current_low < prev_low".
    #    Ora usiamo i valori vettorializzati invece di quelli a singola riga.
    buy_breakout = (df['High'] > df['High_prev1'])
    sell_breakout = (df['Low'] < df['Low_prev1'])

    # 4. Combinazione delle condizioni per i segnali finali.
    #    Il segnale di Buy è generato solo se la candela precedente era un'Inside Bar E se il
    #    breakout rialzista si verifica sulla candela attuale. Questo replica la logica
    #    del tuo codice originale ("if is_inside_bar: ... if current_high > prev_high:").
    buy_signal_condition = is_inside_bar & buy_breakout
    sell_signal_condition = is_inside_bar & sell_breakout

    # 5. Inizializzazione e assegnazione dei segnali.
    #    Si crea una Series di default a 0 (Hold) e si usano le maschere booleane
    #    per assegnare i valori 2 (Buy) o 1 (Sell). Questo è il cuore dell'ottimizzazione
    #    rispetto ai return condizionali del tuo codice.
    signals = pd.Series(0, index=df.index, dtype='int8')
    signals[sell_signal_condition] = 1  # Segnale di Sell
    signals[buy_signal_condition] = 2  # Segnale di Buy


    # 6. Pulizia delle colonne temporanee.
    #    Questo passo non esiste nella versione originale perché le variabili sono locali
    #    alla funzione. Qui, invece, dobbiamo rimuovere le colonne che abbiamo aggiunto
    #    al DataFrame.
    df.drop(columns=['High_prev1', 'Low_prev1', 'High_prev2', 'Low_prev2'], inplace=True)

    return signals

def three_bar_reversal_signal_vectorized(df):
    """
    Generates a Series of trading signals for the entire DataFrame based on a
    vectorized Three Bar Reversal strategy.

    Args:
        df (pd.DataFrame): The DataFrame with historical data.

    Returns:
        pd.Series: A Series of trading signals (2 for Buy, 1 for Sell, 0 for Hold).
    """
    # Create temporary columns for previous values using .shift()
    df['open_prev1'] = df['Open'].shift(1)
    df['close_prev1'] = df['Close'].shift(1)
    df['high_prev1'] = df['High'].shift(1)
    df['low_prev1'] = df['Low'].shift(1)

    df['open_prev2'] = df['Open'].shift(2)
    df['close_prev2'] = df['Close'].shift(2)
    df['high_prev2'] = df['High'].shift(2)
    df['low_prev2'] = df['Low'].shift(2)

    # --- Bullish Reversal Conditions ---
    buy_condition = (df['close_prev2'] < df['open_prev2']) & \
                    (df['low_prev1'] < df['low_prev2']) & \
                    (df['high_prev1'] > df['high_prev2']) & \
                    (df['Close'] > df['Open']) & \
                    (df['Close'] > df['open_prev2'])

    # --- Bearish Reversal Conditions ---
    sell_condition = (df['close_prev2'] > df['open_prev2']) & \
                     (df['high_prev1'] > df['high_prev2']) & \
                     (df['low_prev1'] < df['low_prev2']) & \
                     (df['Close'] < df['Open']) & \
                     (df['Close'] < df['open_prev2'])

    # Initialize a signal Series with a default value of 0 (Hold)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Apply signals using vectorized assignment
    signals[buy_condition] = 2  # Buy signal
    signals[sell_condition] = 1  # Sell signal

    # Remove temporary columns to keep the DataFrame clean
    df.drop(columns=['open_prev1', 'close_prev1', 'high_prev1', 'low_prev1',
                     'open_prev2', 'close_prev2', 'high_prev2', 'low_prev2'],
            inplace=True)

    return signals

def engulfing_pattern_signal_vectorized(df):
    """
    Identifies Engulfing Patterns for buy (2) or sell (1) signals
    using a vectorized approach.

    Args:
        df (pd.DataFrame): The DataFrame with historical data.

    Returns:
        pd.Series: A Series of trading signals (2 for Buy, 1 for Sell, 0 for Hold).
    """

    # Create temporary columns for previous values using .shift()
    df['prev_open'] = df['Open'].shift(1)
    df['prev_close'] = df['Close'].shift(1)

    # Bullish Engulfing conditions
    bullish_engulfing_condition = (df['prev_close'] < df['prev_open']) & \
                                  (df['Close'] > df['Open']) & \
                                  (df['Close'] > df['prev_open']) & \
                                  (df['Open'] < df['prev_close'])

    # Bearish Engulfing conditions
    bearish_engulfing_condition = (df['prev_close'] > df['prev_open']) & \
                                  (df['Close'] < df['Open']) & \
                                  (df['Close'] < df['prev_open']) & \
                                  (df['Open'] > df['prev_close'])

    # Initialize a signal Series with a default value of 0 (Hold)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Apply signals using vectorized assignment
    signals[bullish_engulfing_condition] = 2  # Buy Signal
    signals[bearish_engulfing_condition] = 1  # Sell Signal

    # Remove temporary columns to keep the DataFrame clean
    df.drop(columns=['prev_open', 'prev_close'], inplace=True)

    return signals

def pin_bar_signal_vectorized(df):
    """
    Identifies Pin Bars using a vectorized approach to generate buy (2) or sell (1) signals.

    Args:
        df (pd.DataFrame): The DataFrame with historical data.

    Returns:
        pd.Series: A Series of trading signals (2 for Buy, 1 for Sell, 0 for Hold).
    """
    # Vectorized calculation of candle body and wicks
    df['candle_body'] = abs(df['Close'] - df['Open'])
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']

    # Vectorized criteria for Pin Bars
    # Bullish Pin Bar (Hammer) - Long Lower Wick
    bullish_pin_bar_condition = (df['lower_wick'] >= df['candle_body'] * 2) & \
                                (df['upper_wick'] <= df['candle_body'] * 0.5) & \
                                (df['Close'] > df['Open'])

    # Bearish Pin Bar (Shooting Star) - Long Upper Wick
    bearish_pin_bar_condition = (df['upper_wick'] >= df['candle_body'] * 2) & \
                                (df['lower_wick'] <= df['candle_body'] * 0.5) & \
                                (df['Close'] < df['Open'])

    # Initialize a signal Series with a default value of 0 (Hold)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Apply signals using vectorized assignment
    signals[bullish_pin_bar_condition] = 2  # Buy Signal
    signals[bearish_pin_bar_condition] = 1  # Sell Signal

    # Remove temporary columns to keep the DataFrame clean
    df.drop(columns=['candle_body', 'upper_wick', 'lower_wick'], inplace=True)

    return signals

def morning_evening_star_signal_vectorized(df):
    """
    Identifies Morning Star (Buy - 2) and Evening Star (Sell - 1) patterns
    using a vectorized approach.

    Args:
        df (pd.DataFrame): The DataFrame with historical data.

    Returns:
        pd.Series: A Series of trading signals (2 for Buy, 1 for Sell, 0 for Hold).
    """

    # Create temporary columns for previous values using .shift()
    df['first_open'] = df['Open'].shift(2)
    df['first_close'] = df['Close'].shift(2)
    df['second_open'] = df['Open'].shift(1)
    df['second_close'] = df['Close'].shift(1)

    # Bullish Morning Star conditions
    morning_star_condition = (df['first_close'] < df['first_open']) & \
                             (df['Close'] > df['Open']) & \
                             (df['Close'] > (df['first_open'] + df['first_close']) / 2)

    # Bearish Evening Star conditions
    evening_star_condition = (df['first_close'] > df['first_open']) & \
                             (df['Close'] < df['Open']) & \
                             (df['Close'] < (df['first_open'] + df['first_close']) / 2)

    # Initialize a signal Series with a default value of 0 (Hold)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Apply signals using vectorized assignment
    signals[morning_star_condition] = 2  # Buy Signal
    signals[evening_star_condition] = 1  # Sell Signal

    # Remove temporary columns to keep the DataFrame clean
    df.drop(columns=['first_open', 'first_close', 'second_open', 'second_close'], inplace=True)

    return signals

def shooting_star_hammer_signal_vectorized(df):
    """
    Identifies Shooting Star and Hammer patterns using a vectorized approach

    Args:
        df (pd.DataFrame): The DataFrame with historical data.

    Returns:
        pd.Series: A Series of trading signals (1 for sell, 2 for buy, 0 for no signal).
    """

    # Vectorized calculation of candle body and wicks
    df['candle_body'] = abs(df['Close'] - df['Open'])
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']

    # --- Shooting Star (Bearish) conditions ---
    shooting_star_condition = (df['upper_wick'] >= df['candle_body'] * 2) & \
                              (df['lower_wick'] <= df['candle_body'] * 0.3) & \
                              (df['Close'] < df['Open'])

    # --- Hammer (Bullish) conditions ---
    hammer_condition = (df['lower_wick'] >= df['candle_body'] * 2) & \
                       (df['upper_wick'] <= df['candle_body'] * 0.3) & \
                       (df['Close'] > df['Open'])

    # Initialize a signal Series with a default value of 0 (no signal)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Apply signals using vectorized assignment
    signals[shooting_star_condition] = 1  # SELL signal
    signals[hammer_condition] = 2  # BUY signal

    # Remove temporary columns to keep the DataFrame clean
    df.drop(columns=['candle_body', 'upper_wick', 'lower_wick'], inplace=True)

    return signals

def hammer_signal_vectorized(df):
    """
    Identifies a Hammer pattern using a vectorized approach to generate buy (2) or sell (1) signals.

    Args:
        df (pd.DataFrame): The DataFrame with historical data.

    Returns:
        pd.Series: A Series of trading signals (1 for sell, 2 for buy, 0 for no signal).
    """

    # Vectorized calculation of candle body and wicks
    df['body'] = abs(df['Close'] - df['Open'])
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)

    # --- Bullish Hammer conditions ---
    hammer_condition = (df['lower_wick'] > 2 * df['body']) & \
                       (df['upper_wick'] < 0.3 * df['body'])

    # --- Bearish Exit (Sell) condition ---
    # This condition is typically handled by backtesting logic, not a signal function.
    # We'll vectorize it here as a direct translation of your code.
    bearish_exit_condition = (df['Close'] < df['Low'].shift(1))

    # Initialize a signal Series with a default value of 0 (no signal)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Apply signals using vectorized assignment
    signals[hammer_condition] = 2  # BUY signal
    signals[bearish_exit_condition] = 1  # SELL signal

    # Remove temporary columns to keep the DataFrame clean
    df.drop(columns=['body', 'lower_wick', 'upper_wick'], inplace=True)

    return signals

def inverted_hammer_signal_vectorized(df):
    """
    Identifies Inverted Hammer patterns using a vectorized approach to generate buy (2) signals.

    Args:
        df (pd.DataFrame): The DataFrame with historical data.

    Returns:
        pd.Series: A Series of trading signals (2 for Buy, 0 for no signal).
    """

    # Vectorized calculation of candle body and wicks
    df['body'] = abs(df['Close'] - df['Open'])
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']

    # --- Inverted Hammer conditions ---
    inverted_hammer_condition = (df['upper_wick'] > 2 * df['body']) & \
                                (df['lower_wick'] < 0.3 * df['body'])
    bearish_exit_condition = (df['Close'] < df['Low'].shift(1))
    # Initialize a signal Series with a default value of 0 (no signal)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Apply signals using vectorized assignment
    signals[inverted_hammer_condition] = 2  # BUY signal
    signals[bearish_exit_condition] = 1  # BUY signal

    # Remove temporary columns to keep the DataFrame clean
    df.drop(columns=['body', 'upper_wick', 'lower_wick'], inplace=True)

    return signals

def filled_bar_vectorized(df,ratio=0.9):
    df["ratio"]=(df['Close']-df['Open'])/(df['High']-df['Low'])
    df['prev_low']=df['Low'].shift(1)
    bullish_signal = df["ratio"] > ratio
    bearish_signal = df['Close'] < df['prev_low']
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Apply entry signals using vectorized assignment
    signals[bullish_signal] = 2  # Buy signal
    signals[bearish_signal] = 1  # Sell signal

    # Clean up temporary columns
    df.drop(columns=['ratio', 'prev_low'], inplace=True)

    return signals



def inverted_filled_bar_strategy(df, ratio=0.9):
    """
    Inverts the logic of the filled_bar_vectorized function to generate signals
    for a short strategy.

    A buy signal from the original function becomes a short entry signal (1).
    A sell signal from the original function becomes a short exit signal (2).

    Args:
        df (pd.DataFrame): The DataFrame with historical OHLC data.
        ratio (float): The threshold for the body-to-range ratio of the candle.

    Returns:
        pd.Series: A Series of inverted trading signals
                   (1 for short entry, 2 for short exit, 0 for no signal).
    """

    # Call the original function to get the base signals.
    # A copy of the DataFrame is passed to avoid modifying the original.
    original_signals = filled_bar_vectorized(df.copy(), ratio)

    # Invert the signals using np.where in a vectorized manner.
    # If the original signal is 2 (buy), the new signal is 1 (short).
    # If the original signal is 1 (sell), the new signal is 2 (buy to cover).
    # If the original signal is 0, it remains 0.
    inverted_signals = np.where(original_signals == 2, 1,
                                np.where(original_signals == 1, 2, 0))

    # Return a pandas Series with the correct index.
    return pd.Series(inverted_signals, index=df.index, dtype='int8')



def doji_signal_strategy_v1_vectorized(df, trend_period=5):
    """
    Generates trading signals based on a Doji and the preceding trend
    using a vectorized approach.

    Args:
        df (pd.DataFrame): DataFrame with OHLC data.
        trend_period (int): Number of candles to consider for the trend.

    Returns:
        pd.Series: Trading signal Series (2 = Buy, 1 = Sell, -1 = Exit Short, -2 = Exit Long, 0 = Hold).
    """

    # Calculate trend using .shift() to compare closing prices
    # with the closing price from 'trend_period' candles ago.
    df['first_close'] = df['Close'].shift(trend_period)
    df['prev_close'] = df['Close'].shift(1)

    is_uptrend = df['prev_close'] > df['first_close']
    is_downtrend = df['prev_close'] < df['first_close']

    # Identify Doji candles using a vectorized calculation
    is_doji = abs(df['Open'] - df['Close']) < (df['High'] - df['Low']) * 0.1

    # Initialize a signal Series with a default value of 0 (Hold)
    signals = pd.Series(0, index=df.index, dtype='int8')

    # Apply entry signals using vectorized assignment
    signals[is_doji & is_downtrend] = 2  # Buy signal
    signals[is_doji & is_uptrend] = 1  # Sell signal

    # Apply exit signals based on trend.
    # NOTE: In backtesting, this logic is usually more complex
    # and managed by the backtest engine, not the signal function.
    # This is a direct translation of your original code's logic.
    signals[is_uptrend] = -2  # Exit a short position
    signals[is_downtrend] = -1  # Exit a long position

    # Override any conflicting signals:
    # A Doji entry signal takes priority over an exit signal
    signals[is_doji & is_downtrend] = 2
    signals[is_doji & is_uptrend] = 1

    # Clean up temporary columns
    df.drop(columns=['first_close', 'prev_close'], inplace=True)

    return signals

def combined_signal_vectorized(df):
    """
    Combines signals from multiple vectorized candlestick strategies.

    Args:
        df (pd.DataFrame): DataFrame containing candlestick data.
        candlestick_strategies (list): List of vectorized candlestick strategy functions.

    Returns:
        pd.Series: A Series with the combined signal (2 for buy, 1 for sell, 0 for no signal).
    """

    # 1. Run each vectorized strategy function and store the signals.
    # We pass a copy of the DataFrame to each function to avoid conflicts
    # if they add temporary columns.
    signal_series_list = [func(df.copy()) for func in candlestick_strategies if func != combined_signal_vectorized]

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

def liquidity_grab_rev_strategy(df: pd.DataFrame, lookback: int = 20):
    """
    Liquidity Grab Reversal (vectorized, one signal per grab).

    Args:
        df: DataFrame with columns ['High','Low'] (Close optional).
        lookback: number of candles for breakout detection.


    Returns:
        signals (pd.Series): 2 = Buy, 1 = Sell, 0 = Hold

    """

    highs = df['High']
    lows  = df['Low']

    # <-- 1) breakout detection (exclude current candle from the reference window) -->
    breakout_high = highs > highs.shift(1).rolling(lookback).max()
    breakout_low  = lows  < lows.shift(1).rolling(lookback).min()

    # <-- 2) produce raw grab levels at breakout candles and propagate forward -->
    grab_low_raw  = np.where(breakout_high, lows, np.nan)   # when price broke a previous high -> save that candle's low
    grab_high_raw = np.where(breakout_low, highs, np.nan)   # when price broke a previous low  -> save that candle's high

    grab_low  = pd.Series(grab_low_raw,  index=df.index).ffill()
    grab_high = pd.Series(grab_high_raw, index=df.index).ffill()

    # <-- 3) group ids: each grab sequence has an id (0 = no grab yet) -->
    gid_low  = breakout_high.cumsum()
    gid_high = breakout_low.cumsum()

    # <-- 4) detect breaking of current grab level -->
    broken_low  = (lows  < grab_low)  & grab_low.notna()   # candidate short
    broken_high = (highs > grab_high) & grab_high.notna()  # candidate long

    # <-- 5) keep ONLY the FIRST break inside each grab-group (vectorized) -->
    # cumsum inside each group: first time broken -> cum == 1
    broken_low_cum  = broken_low.groupby(gid_low).cumsum()
    first_broken_low = broken_low & (broken_low_cum == 1)

    broken_high_cum  = broken_high.groupby(gid_high).cumsum()
    first_broken_high = broken_high & (broken_high_cum == 1)

    # <-- 6) create signals; resolve rare conflicts (both true same bar) by zeroing them -->
    signals = pd.Series(0, index=df.index, dtype='int8')
    signals[first_broken_low]  = 1
    signals[first_broken_high] = 2

    # If both happened same bar (extremely rare), remove ambiguity:
    conflict = first_broken_low & first_broken_high
    if conflict.any():
        signals[conflict] = 0
    return signals


def liquidity_grab_tf_strategy(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    # Chiama la strategia originale
    signals = liquidity_grab_rev_strategy(df, lookback)

    # Inverte i segnali
    inverse_signals = signals.copy()
    inverse_signals[signals == 2] = 1  # Long → Short
    inverse_signals[signals == 1] = 2  # Short → Long
    # Hold rimane 0

    return inverse_signals



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
    dax_momentum_signal_vectorized,
    inside_bar_breakout_signal_vectorized,
    #three_bar_reversal_signal_vectorized,
    engulfing_pattern_signal_vectorized,
    #pin_bar_signal_vectorized,
    #morning_evening_star_signal_vectorized,
    #shooting_star_hammer_signal_vectorized,
    #hammer_signal_vectorized,
    #inverted_hammer_signal_vectorized,
    #doji_signal_strategy_v1_vectorized,
    #filled_bar_vectorized,
    #inverted_filled_bar_strategy,
    retracement_rev_vectorized,
    retracement_tf_vectorized,
    liquidity_grab_rev_strategy,
    liquidity_grab_tf_strategy,
    donchian_breakout_with_ma_filter,
    donchian_breakout_with_ma_10_30,
    donchian_breakout_with_ma_15_40,
    donchian_inv_with_ma_filter,
    donchian_inv_with_ma_15_40,
    donchian_inv_with_ma_10_30,
    #keltner_rev_vectorized,
    #keltner_rev_15_1_5,
    #keltner_rev_50_2_5,
    #keltner_tf_vectorized,
    #keltner_tf_15_1_5,
    #keltner_tf_50_2_5,
    weekly_breakout_vectorized,
    weekly_breakout_1_2,
    weekly_breakout_2_1,
    weekly_breakout_2_2,
    weekly_breakout_1_3,
    weekly_breakout_2_3,
    vwap_trading_strategy_threshold,
    vwap_trading_strategy_cross,
    vwap_trading_strategy_cross_20_10,
    vwap_trading_strategy_cross_30_20,
    parab_rev_vectorized,
    parab_rev_8_16_32_vectorized,
    parab_rev_3_6_15_vectorized,
    #ttm_squeeze_strategy,
    ema_vwap_sma_vectorized,
    #momentum_vwap_vma_strategy,
    ema_vwap_sma_15_30_50,
    ema_vwap_sma_5_10_20,
    ema_vwap_vectorized,
    volume_trend_basic,
    volume_trend_10,
    volume_trend_30,
    squeeze_ema_strategy,
    squeeze_ema_10_20,
    squeeze_ema_15_25,
    ema_pullback_strategy,
    volume_trend_ema_threshold,
    volume_donchian_breakout,
    volume_donchian_breakout_10_25_20,
    volume_donchian_breakout_15_30_20
    #combined_signal_vectorized
]
if __name__ == "__main__":
    print(count_hammers_in_ticker("GE"))
    print(count_inverted_hammers_in_ticker("GE"))