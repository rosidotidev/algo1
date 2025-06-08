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
import datetime

from stock.base_bt import BaseStrategyBT


class DaxPatternBT(BaseStrategyBT):
    slperc= 0.05
    tpperc= 1.00
    def init(self):
        self.long_stop_price = None  # Stop Loss
        self.short_stop_price = None
        self.long_tp_price= None
        self.short_tp_price= None

    def stop_loss_long_check(self):
        stop_l = self.data.Close[-1] < self.long_stop_price
        return stop_l

    def take_profit_long_check(self):
        take_p = self.data.Close[-1] > self.long_tp_price
        return take_p

    def take_profit_short_check(self):
        take_p = self.data.Close[-1] < self.short_tp_price
        return take_p

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
                self.long_tp_price = self.data.Close[-1] * (1 + self.tpperc)
            elif total_signal == 1:
                self.sell()
                self.track_enter_short()
                self.short_stop_price = self.data.Close[-1] * (1 + self.slperc)
                self.short_tp_price = self.data.Close[-1] * (1 - self.tpperc)
        else:
            if self.position.is_long and ((total_signal==1 or total_signal==-2)
                    or self.stop_loss_long_check() or self.take_profit_long_check()):
                self.position.close()
                self.track_close_long()
            elif self.position.is_short and ((total_signal==2 or total_signal==-1)
                    or self.stop_loss_short_check() or self.take_profit_short_check()):
                self.position.close()
                self.track_close_short()





def run_backtest_DaxPattern(data_path,slperc=0.04,tpperc=0.02,capital_allocation=1,show_plot=False,target_strategy=cs.dax_total_signal,add_indicators=True):
    df=ti.read_from_csv(data_path)
    df=bu.norm_date_time(df)
    if add_indicators:
        df=de.add_rsi_macd_bb(df)
        df=de.add_smas_long_short(df)
        df=de.add_stoch(df)
    df = ti.add_total_signal(df,target_strategy)
    #print(df[df["TotalSignal"]>0])
    bt = Backtest(df, DaxPatternBT,
                  cash=10000,
                  finalize_trades=True,
                  exclusive_orders=True,
                  commission=.002)

    # Esegui il backtest
    ctx={}
    results = bt.run(slperc=slperc,tpperc=tpperc,df=df,ctx=ctx)
    ctx.update(results.to_dict())
    #print(f" ctx {results}")
    if show_plot:
        bt.plot(filename=None)
    for key, value in ctx.items():
        results[key] = bu.format_value(value)
    #return cerebro.broker.getvalue()
    return results

def run_backtest_for_all_tickers(tickers_file, data_directory,slperc=0.15,tpperc=0.15,candle_strategy=cs.dax_momentum_signal,add_indicators=False):
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
            report = run_backtest_DaxPattern(data_path, slperc=slperc,tpperc=tpperc,capital_allocation=1,target_strategy=candle_strategy,add_indicators=add_indicators)
            report["strategy"]=candle_strategy.__name__
            results[ticker] = report[field_selected]
            reports[ticker] = report
        except Exception as e:
            print(f"Error running backtest for {ticker}: {repr(e)}")
            traceback.print_exc()


    # Find the best-performing ticker
    if results:
        df=bu.save_reports_to_df(reports)
        #fbest_ticker = max(results, key=results.get)
        #print("\nBest performing ticker:", best_ticker)
        #print(f"strategy {candle_strategy}")
        #print(f"{field_selected}:", results[best_ticker])
        #print(results)
        return df

def exec_analysis(base_path="../",slperc=0.15, tpperc=1.0):
    df = None
    for strategy in cs.candlestick_strategies:
        df1 = run_backtest_for_all_tickers(f'{base_path}../data/tickers.txt', f'{base_path}../data/', slperc=slperc,tpperc=tpperc,candle_strategy=strategy)
        df = bu.append_df(df, df1)
    for strategy in ins.indicators_strategy:
        df1 = run_backtest_for_all_tickers(f'{base_path}../data/tickers.txt', f'{base_path}../data/', slperc=slperc,tpperc=tpperc,candle_strategy=strategy,
                                           add_indicators=True)
        df = bu.append_df(df, df1)
    return df

def exec_analysis_parallel(base_path="../", slperc=0.15, tpperc=1.0):
    """
    Executes backtesting for all tickers using both candlestick and indicator strategies in parallel.

    Args:
        base_path (str): Base path for data files and output.
        slperc (float): Stop loss percentage.
        tpperc (float): Take profit percentage.

    Returns:
        DataFrame: Combined results from all backtests.
    """
    df = None
    futures = []
    tickers_file = f'{base_path}../data/tickers.txt'
    data_dir = f'{base_path}../data/'

    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        # Candlestick strategies
        for strategy in cs.candlestick_strategies:
            futures.append(executor.submit(
                run_backtest_for_all_tickers,
                tickers_file, data_dir, slperc, tpperc, strategy, False
            ))

        # Indicator strategies
        for strategy in ins.indicators_strategy:
            futures.append(executor.submit(
                run_backtest_for_all_tickers,
                tickers_file, data_dir, slperc, tpperc, strategy, True
            ))

        for future in concurrent.futures.as_completed(futures):
            try:
                df1 = future.result()
                df = bu.append_df(df, df1)
            except Exception as e:
                print(f"Error during backtest execution: {repr(e)}")

    return df
def exec_analysis_and_save_results(base_path='../', slperc=0.15, tpperc=1.0, parallel=True):
    start_time = time.time()  # ⏱️ Start timer

    # Choose execution mode
    if parallel:
        df = exec_analysis_parallel(base_path, slperc=slperc, tpperc=tpperc)
    else:
        df = exec_analysis(base_path, slperc=slperc, tpperc=tpperc)

    # Save results
    today = datetime.datetime.now().strftime("%Y%m%d")
    df.to_csv(f"{base_path}../results/report_{today}.csv", index=False)
    df.to_csv(f"{base_path}../results/report.csv", index=False)

    # ⏱️ Print decorated summary
    elapsed_time = time.time() - start_time
    summary = (
            "\n" + "*" * 60 + "\n" +
            "*           Backtest Execution Completed            *\n" +
            "*" * 60 + "\n" +
            f"* Mode:       {'Parallel' if parallel else 'Sequential':<42}*\n" +
            f"* Stop Loss:  {slperc:<42}*\n" +
            f"* Take Profit:{tpperc:<42}*\n" +
            f"* Duration:   {elapsed_time:.2f} seconds{'':<29}*\n" +
            "*" * 60 + "\n"
    )
    print(summary)  # Optional: print to console
    return summary

def test0():
    run_backtest_DaxPattern("../../data/GS.csv",slperc=0.15,tpperc=0.02,capital_allocation=1000000,show_plot=True)


def test1():
    # run_backtest_DaxPattern("../../data/GS.csv",slperc=0.15,tpperc=0.02,capital_allocation=1,show_plot=True)
    # run_backtest_DaxPattern("../../data/SBUX.csv", slperc=0.15, tpperc=0.02, capital_allocation=1, show_plot=True,
    #                        target_strategy=ins.mean_reversion_signal_v1, add_indicators=True)

    # run_backtest_DaxPattern("../../data/AAPL.csv", slperc=0.15, tpperc=0.02, capital_allocation=1, show_plot=True,
    #                        target_strategy=ins.moving_average_crossover_signal, add_indicators=True)
    # Measure execution time
    start_time = time.time()
    df = exec_analysis("../")

    df = bu.load_csv("../../results/report.csv")
    end_time = time.time()
    print(f""
          f"***** Execution time: {end_time - start_time:.4f} seconds"
          f"")
    df.to_csv("../../results/report.csv", index=False)
    pd.set_option("display.max_rows", None)  # Show all rows
    pd.set_option("display.max_columns", None)  # Show all columns
    print("RESULTS __________________________________________")
    res_filtered = df[(df["Win Rate [%]"] >= 50)
                      & (df["Last Action"] == 1)
                      #                 & (df["Equity Final [$]"] >80000)
                      & (df["# Trades"] > 0)
                      ]
    res_filtered = res_filtered[["Ticker", "Win Rate [%]", "Equity Final [$]", "# Trades", "strategy", "Last Action"]]
    print(res_filtered)


if __name__ == "__main__":
    test0()