
import pandas_ta as ta
from stock import ticker as t
def main():
    print('hello pandas-ta')
    print(dir(ta))
    stock = 'KO'
    df = t.read_pandas_ticker(stock, '3y')
    df['SMA_10'] = ta.sma(df['Close'])
    df['RSI'] = ta.rsi(df['Close'])
    df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
    print(df)
    vp = df.ta.vp(price='close', volume='volume', bins=100)

    print(vp)
if __name__ == "__main__":
    main()