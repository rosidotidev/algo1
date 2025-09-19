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

if __name__ == "__main__":
    mainx()