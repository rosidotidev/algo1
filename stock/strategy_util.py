import pandas as pd

def detect_ttm_squeeze(df: pd.DataFrame, bb_length: int = 20, kc_length: int = 20, mult: float = 1.5) -> pd.Series:
    """
    Detects TTM Squeeze condition (Bollinger vs Keltner).
    Returns a Series:
        1 = Squeeze ON (compression)
        0 = Squeeze OFF (normal / expansion)

    Args:
        df (pd.DataFrame): Must contain ['Close', 'High', 'Low'].
        bb_length (int): Bollinger Bands lookback period.
        kc_length (int): Keltner Channel lookback period.
        mult (float): Multiplier for Keltner Channel width.
    """

    # --- Bollinger Bands ---
    mid_bb = df['Close'].rolling(bb_length).mean()
    std_bb = df['Close'].rolling(bb_length).std()
    upper_bb = mid_bb + 2 * std_bb
    lower_bb = mid_bb - 2 * std_bb

    # --- Keltner Channel ---
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(kc_length).mean()
    mid_kc = df['Close'].rolling(kc_length).mean()
    upper_kc = mid_kc + atr * mult
    lower_kc = mid_kc - atr * mult

    # --- Squeeze condition ---
    squeeze_on = (upper_bb < upper_kc) & (lower_bb > lower_kc)

    return squeeze_on.astype(int)
