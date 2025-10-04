from stock import ticker as t
from stock import plot
from dotenv import load_dotenv
from stock.x_trades_bt import run_backtest_for_all_tickers
import shutil
from biz import biz_logic as biz

import shutil


def run_with_tickers(base_path="../data", tickers=["AMZN"]):
    """
    1. Copy {base_path}/tickers.txt -> {base_path}/tickers_tmp.txt
       (overwrite if exists, create empty backup if tickers.txt missing).
    2. Create a new tickers.txt with the specified tickers.
    3. Run mock logic.
    4. Restore tickers.txt from tickers_tmp.txt.

    Args:
        base_path (str): Path to the data folder.
        tickers (list[str]): List of ticker symbols to use temporarily.
    """
    tickers_file = f"{base_path}/tickers.txt"
    backup_file = f"{base_path}/tickers_tmp.txt"

    # Step 1: Backup
    try:
        shutil.copyfile(tickers_file, backup_file)
    except FileNotFoundError:
        with open(backup_file, "w") as f:
            f.write("")  # empty backup

    # Step 2: Write new tickers.txt
    with open(tickers_file, "w") as f:
        f.write("\n".join(tickers))  # write each ticker on a new line

    # Step 3: Run the process
    biz.run_long_process(optimize=True,parallel=False)

    # Step 4: Restore
    shutil.copyfile(backup_file, tickers_file)
    print("tickers.txt restored from tickers_tmp.txt")


def main1():
    load_dotenv()
    print("Hello world")
    stock='KO'
    df=t.read_pandas_ticker(stock,'3y')
    df=t.add_BB(df,21)
    df = t.add_BB(df, 100)
    t.save_to_file(df,f'{stock}.csv')

    plot.plot_columns(df,['Close','BB_21_UP','BB_21_LO','BB_100_UP','BB_100_LO'])
def main():
    run_backtest_for_all_tickers('../data/tickers.txt','../data/')

def mainx():
    run_with_tickers(tickers=["AMZN","INTC","WDC"])

def test_liquidity_grab():
    import matplotlib.pyplot as plt

    def plot_liquidity_grab(df, grab_low, grab_high, signals):
        plt.figure(figsize=(12, 6))

        # Plot candlestick-like: High-Low line
        plt.vlines(df.index, df['Low'], df['High'], color='black', alpha=0.6)

        # Plot grab levels
        plt.plot(df.index, grab_low, color='red', linestyle='--', label='Grab Low')
        plt.plot(df.index, grab_high, color='green', linestyle='--', label='Grab High')

        # Plot signals
        plt.scatter(df.index[signals == 2], df['High'][signals == 2], color='green', marker='^', s=100,
                    label='Long Signal')
        plt.scatter(df.index[signals == 1], df['Low'][signals == 1], color='red', marker='v', s=100,
                    label='Short Signal')

        plt.title("Liquidity Grab Strategy Signals")
        plt.xlabel("Candles")
        plt.ylabel("Price")
        plt.legend()
        #n.pyplt.savefig("plot.html", bbox_inches='tight')  # salva su file
        plt.show()
    import stock.simpe_signal_vec as can
    import stock.ticker as ti
    import pandas as pd
    import numpy as np

    df=ti.read_from_csv("../data/GLD.csv")
    signals = can.liquidity_grab_strategy(df, lookback=20)
    # grab_low e grab_high li puoi calcolare come nella funzione
    highs = df['High']
    lows = df['Low']
    breakout_high = highs > highs.shift(1).rolling(20).max()
    breakout_low = lows < lows.shift(1).rolling(20).min()
    grab_low = pd.Series(np.where(breakout_high, lows, np.nan), index=df.index).ffill()
    grab_high = pd.Series(np.where(breakout_low, highs, np.nan), index=df.index).ffill()

    plot_liquidity_grab(df, grab_low, grab_high, signals)


if __name__ == "__main__":
    test_liquidity_grab()