import backtrader as bt
from stock.base import BaseStrategy
import stock.ticker as ti
from backtrader_util import bu
import candle_signal as cs
import indicators_signal as ins
import stock.data_enricher as de
import pandas as pd
import traceback
import concurrent.futures
import time


class DaxPattern(BaseStrategy):
    params = (
        ('slperc', 0.05),  # Fast SMA period
        ('tpperc', 0.02),  # Slow SMA period
    )
    def __init__(self):
        super().__init__()
        self.long_stop_price = None  # Stop Loss
        self.short_stop_price = None

    def stop_loss_long_check(self):
        stop_l = self.data.close[0] < self.long_stop_price
        if (stop_l):
            print(f"***** long stop loss {self.data.close[0]} < {self.long_stop_price}")
        return stop_l

    def stop_loss_short_check(self):
        stop_l = self.data.close[0] > self.short_stop_price
        if (stop_l):
            print(f"***** short stop loss {self.data.close[0]} > {self.short_stop_price}")
        return stop_l

    def next(self):
        # Get the current date in the format of the DataFrame index
        #total_signal=self.get_total_signal()
        total_signal=self.current_df()["TotalSignal"]
        size = self.calculate_size()  # Calculate size dynamically based on capital allocation
        self.track_no_action()
        if not self.position:
            if size>0:
                if total_signal == 2:
                    self.buy(size=size)
                    self.track_enter_long()
                    self.long_stop_price=self.data.close[0] * (1 - self.params.slperc)
                elif total_signal == 1:
                    self.sell(size=size)
                    self.track_enter_short()
                    self.short_stop_price = self.data.close[0] * (1 + self.params.slperc)
        else:
            if self.position.size>0 and ((total_signal==1 or total_signal==-2)
                    or self.stop_loss_long_check()):
                self.close()
                self.track_close_long()
            elif self.position.size<0 and ((total_signal==2 or total_signal==-1)
                    or self.stop_loss_short_check()):
                self.close()
                self.track_close_short()





def run_backtest_DaxPattern(data_path,slperc=0.04,tpperc=0.02,capital_allocation=1,show_plot=False,target_strategy=cs.dax_total_signal,add_indicators=False):
    df=ti.read_from_csv(data_path)
    df=bu.norm_date_time(df)
    if add_indicators:
        df=de.add_rsi_macd_bb(df)
    df = ti.add_total_signal(df,target_strategy)
    #df = ti.add_total_signal(df, cs.shooting_star_hammer_signal)
    data_feed = bt.feeds.PandasData(dataname=df)
    # Initialize Cerebro
    cerebro = bt.Cerebro()
    cerebro.adddata(data_feed)
    bu.add_analyzers_to_cerebro(cerebro)
    # Add strategy
    cerebro.addstrategy(DaxPattern,
                        capital_allocation=capital_allocation,
                        df=df,
                        slperc=slperc)

    # Run the backtest
    # Set initial capital
    cerebro.broker.setcash(10000.0)
    results = cerebro.run()
    report=bu.get_backtest_report(results)
    #print(report)
    #cerebro.plot()
    if show_plot:
        cerebro.plot()
    #return cerebro.broker.getvalue()
    return report

def run_backtest_for_all_tickers(tickers_file, data_directory,candle_strategy=cs.dax_momentum_signal,add_indicators=False):
    """Runs backtests for all tickers in tickers.txt and determines the best performer."""
    best_report=None
    # Read tickers from file
    with open(tickers_file, "r") as f:
        tickers = [line.strip() for line in f.readlines()]

    # Dictionary to store final portfolio values
    results = {}
    reports = {}
    field_selected='Final Portfolio Value'
    #field_selected='Win Rate (%)'
    print(f"""
    
    {candle_strategy.__name__}
    
    """)
    for ticker in tickers:
        print(ticker)
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
        print("report:",reports[best_ticker] )
        print(results)
        return df

def exec_analysis():
    df = None
    for strategy in cs.candlestick_strategies:
        df1 = run_backtest_for_all_tickers('../../data/tickers.txt', '../../data/', candle_strategy=strategy)
        df = bu.append_df(df, df1)
    for strategy in ins.indicators_strategy:
        df1 = run_backtest_for_all_tickers('../../data/tickers.txt', '../../data/', candle_strategy=strategy,
                                           add_indicators=True)
        df = bu.append_df(df, df1)
    return df


import concurrent.futures


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


if __name__ == "__main__":
    #run_backtest_DaxPattern("../../data/GS.csv",slperc=0.15,tpperc=0.02,capital_allocation=1,show_plot=True)
    run_backtest_DaxPattern("../../data/SBUX.csv", slperc=0.15, tpperc=0.02, capital_allocation=1, show_plot=True,
                            target_strategy=ins.mean_reversion_signal_v1, add_indicators=True)
    # Measure execution time
    start_time = time.time()
    #df=exec_analysis_parallel()

    df=bu.load_csv("../../results/report.csv")
    end_time = time.time()
    print(f""
          f"***** Execution time: {end_time - start_time:.4f} seconds"
          f"")
    df.to_csv("../../results/report.csv", index=False)
    pd.set_option("display.max_rows", None)  # Show all rows
    pd.set_option("display.max_columns", None)  # Show all columns
    print("RESULTS __________________________________________")
    res_filtered=df[(df["Win Rate (%)"] >= 80)
    #                & (df["Last Action"]==2)
                     & (df["Total Trades"] > 10)
    ]
    print(res_filtered)
