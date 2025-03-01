from backtesting import Backtest
import stock.ticker as ti
from backtrader_util import bu
import stock.candle_signal as cs
import stock.indicators_signal as ins
import data.data_enricher as de
import pandas as pd
import traceback
import concurrent.futures
import time

from stock.base_bt import BaseStrategyBT


class DaxPatternBT(BaseStrategyBT):
    slperc= 0.05
    def init(self):
        self.long_stop_price = None  # Stop Loss
        self.short_stop_price = None

    def stop_loss_long_check(self):
        stop_l = self.data.Close[-1] < self.long_stop_price
        return stop_l

    def stop_loss_short_check(self):
        stop_l = self.data.Close[-1] > self.short_stop_price
        return stop_l

    def next(self):
        # Get the current date in the format of the DataFrame index
        #total_signal=self.get_total_signal()
        total_signal=self.current_df()["TotalSignal"]

        #size = self.calculate_size()  # Calculate size dynamically based on capital allocation
        self.track_no_action()
        if not (self.position.is_short or self.position.is_long):

            if total_signal == 2:
                self.buy()
                self.track_enter_long()
                self.long_stop_price=self.data.Close[-1] * (1 - self.slperc)
            elif total_signal == 1:
                self.sell()
                self.track_enter_short()
                self.short_stop_price = self.data.Close[-1] * (1 + self.slperc)
        else:
            if self.position.is_long and ((total_signal==1 or total_signal==-2)
                    or self.stop_loss_long_check()):
                self.position.close()
                self.track_close_long()
            elif self.position.is_short and ((total_signal==2 or total_signal==-1)
                    or self.stop_loss_short_check()):
                self.position.close()
                self.track_close_short()





def run_backtest_DaxPattern(data_path,slperc=0.04,tpperc=0.02,capital_allocation=1,show_plot=False,target_strategy=cs.dax_total_signal,add_indicators=True):
    df=ti.read_from_csv(data_path)
    df=bu.norm_date_time(df)
    if add_indicators:
        df=de.add_rsi_macd_bb(df)
    df = ti.add_total_signal(df,target_strategy)
    #print(df[df["TotalSignal"]>0])
    bt = Backtest(df, DaxPatternBT,
                  cash=10000,
                  finalize_trades=True,
                  exclusive_orders=True,
                  commission=.002)

    # Esegui il backtest
    ctx={}
    results = bt.run(slperc=0.20,df=df,ctx=ctx)
    ctx.update(results.to_dict())
    #print(f" ctx {results}")
    if show_plot:
        bt.plot(filename=None)
    for key, value in ctx.items():
        results[key] = value
    #return cerebro.broker.getvalue()
    return results

def run_backtest_for_all_tickers(tickers_file, data_directory,candle_strategy=cs.dax_momentum_signal,add_indicators=False):
    """Runs backtests for all tickers in tickers.txt and determines the best performer."""
    best_report=None
    # Read tickers from file
    with open(tickers_file, "r") as f:
        tickers = [line.strip() for line in f.readlines()]

    # Dictionary to store final portfolio values
    results = {}
    reports = {}
    field_selected='Equity Final [$]'
    #field_selected='Win Rate (%)'
    print(f"""
        {candle_strategy.__name__}
    """)
    for ticker in tickers:
        #print(ticker)
        data_path = f"{data_directory}/{ticker}.csv"  # Path to CSV file
        try:
            report = run_backtest_DaxPattern(data_path, slperc=0.15,tpperc=0.02,capital_allocation=1,target_strategy=candle_strategy,add_indicators=add_indicators)
            report["strategy"]=candle_strategy.__name__
            results[ticker] = report[field_selected]
            reports[ticker] = report
        except Exception as e:
            print(f"Error running backtest for {ticker}: {repr(e)}")
            traceback.print_exc()


    # Find the best-performing ticker
    if results:
        df=bu.save_reports_to_df(reports)
        best_ticker = max(results, key=results.get)
        print("\nBest performing ticker:", best_ticker)
        print(f"strategy {candle_strategy}")
        print(f"{field_selected}:", results[best_ticker])
        print(results)
        return df

def exec_analysis(base_path="../"):
    df = None
    for strategy in cs.candlestick_strategies:
        df1 = run_backtest_for_all_tickers(f'{base_path}../data/tickers.txt', f'{base_path}../data/', candle_strategy=strategy)
        df = bu.append_df(df, df1)
    for strategy in ins.indicators_strategy:
        df1 = run_backtest_for_all_tickers(f'{base_path}../data/tickers.txt', f'{base_path}../data/', candle_strategy=strategy,
                                           add_indicators=True)
        df = bu.append_df(df, df1)
    return df

def exec_analysis_parallel():
    """
    Executes backtesting for all tickers using both candlestick and indicator strategies in parallel.

    Returns:
        DataFrame: The combined results of all backtests.
    """
    df = None
    futures = []

    # Use ThreadPoolExecutor to run the loops in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # First loop: Run backtests for candlestick strategies
        for strategy in cs.candlestick_strategies:
            futures.append(
                executor.submit(run_backtest_for_all_tickers, '../../data/tickers.txt', '../../data/', strategy))

        # Second loop: Run backtests for indicator strategies
        for strategy in ins.indicators_strategy:
            futures.append(
                executor.submit(run_backtest_for_all_tickers, '../../data/tickers.txt', '../../data/', strategy, True))

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                df1 = future.result()
                df = bu.append_df(df, df1)
            except Exception as e:
                print(f"Error during backtest execution: {repr(e)}")

    return df
def exec_analysis_and_save_results(base_path='../'):
    df=exec_analysis(base_path)
    df.to_csv(f"{base_path}../results/report.csv", index=False)

if __name__ == "__main__":
    #run_backtest_DaxPattern("../../data/GS.csv",slperc=0.15,tpperc=0.02,capital_allocation=1,show_plot=True)
    #run_backtest_DaxPattern("../../data/SBUX.csv", slperc=0.15, tpperc=0.02, capital_allocation=1, show_plot=True,
    #                        target_strategy=ins.mean_reversion_signal_v1, add_indicators=True)

    #run_backtest_DaxPattern("../../data/HON.csv", slperc=0.15, tpperc=0.02, capital_allocation=1, show_plot=True,
    #                        target_strategy=ins.mean_reversion_signal_v1, add_indicators=True)
    # Measure execution time
    start_time = time.time()
    df=exec_analysis("../")

    #df=bu.load_csv("../../results/report.csv")
    end_time = time.time()
    print(f""
          f"***** Execution time: {end_time - start_time:.4f} seconds"
          f"")
    df.to_csv("../../results/report.csv", index=False)
    pd.set_option("display.max_rows", None)  # Show all rows
    pd.set_option("display.max_columns", None)  # Show all columns
    print("RESULTS __________________________________________")
    res_filtered=df[(df["Win Rate [%]"] >= 50)
                     & (df["Last Action"]==1)
    #                 & (df["Equity Final [$]"] >80000)
                     & (df["# Trades"] >0 )
    ]
    res_filtered=res_filtered[["Ticker","Win Rate [%]","Equity Final [$]","# Trades","strategy","Last Action"]]
    print(res_filtered)