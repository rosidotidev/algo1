import pandas as pd

def compute_squeeze(df: pd.DataFrame, bb_length: int = 20, keltner_length: int = 20,
                    bb_mult: float = 2.0, keltner_mult: float = 1.5):
    """
    Compute TTM Squeeze conditions (squeeze on/off).
    """
    basis = df['Close'].rolling(bb_length).mean()
    dev = df['Close'].rolling(bb_length).std()

    upper_bb = basis + bb_mult * dev
    lower_bb = basis - bb_mult * dev

    tr0 = df['High'] - df['Low']
    tr1 = (df['High'] - df['Close'].shift()).abs()
    tr2 = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
    atr = tr.rolling(keltner_length).mean()

    upper_kc = basis + keltner_mult * atr
    lower_kc = basis - keltner_mult * atr

    squeeze_on = (upper_bb < upper_kc) & (lower_bb > lower_kc)
    squeeze_off = (upper_bb > upper_kc) & (lower_bb < lower_kc)

    return pd.DataFrame({
        'squeeze_on': squeeze_on,
        'squeeze_off': squeeze_off
    })


def compute_ttm_hist(df: pd.DataFrame, momentum_length: int = 20, smooth_length: int = 5) -> pd.Series:
    """
    Compute simplified TTM Squeeze histogram (momentum measure).
    """
    price = df['Close']
    momentum = price - price.rolling(momentum_length).mean()
    hist = momentum - momentum.rolling(smooth_length).mean()
    return hist


import pandas as pd


def debug_print_on_date(df: pd.DataFrame,
                        signals: pd.Series,
                        date_str: str = "2025-10-17",
                        tail: int = 20) -> bool:
    """
    Stampa le ultime righe di df e signals se l'ultima data del DataFrame
    (colonna 'Date' o indice datetime) coincide con date_str.

    Args:
        df: DataFrame (indice datetime o colonna 'Date').
        signals: Series di segnali allineata a df.
        date_str: data target in formato 'YYYY-MM-DD' (default '2025-10-17').
        tail: quante righe stampare (default 20).

    Returns:
        True se la stampa è avvenuta, False altrimenti.
    """
    if df.empty:
        print("⚠️ DataFrame vuoto.")
        return False

    target_date = pd.to_datetime(date_str).date()

    # Recupera l'ultima data dal DataFrame: prima prova la colonna 'Date', poi l'indice
    if 'Date' in df.columns:
        last = pd.to_datetime(df['Date'].iloc[-1])
    else:
        # usa l'indice; assicuriamoci che sia datetime
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            try:
                df = df.copy()
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                print("⚠️ L'indice non è datetime e non può essere convertito:", e)
                return False
        last = pd.to_datetime(df.index[-1])

    last_date = last.date()

    if last_date == target_date:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        print(f"📅 Last date is {target_date} — DataFrame tail:")
        print(df.tail(tail))
        print("\n📈 Signals tail:")
        print(signals.tail(tail))
        return True
    else:
        print(last_date)
        return False
