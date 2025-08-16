from stock import ticker as t
from stock import plot
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from stock.dax_pattern_bt import run_backtest_for_all_tickers
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
if __name__ == "__main__":
    main()