from backtesting import Backtest
import stock.ticker as ti
from backtrader_util import bu
import stock.candle_signal_vec as cs_vec
import stock.indicators_signal_vec as ins_vec
import data.data_enricher as de
import pandas as pd
import traceback
import concurrent.futures
import time
import datetime
import os
import threading
from stock.strategy_repo import StrategyRepo
from stock.x_backtesting_bt import XBacktestingBT

lock = threading.Lock()

def load_best_matrix(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"Error reading file {path}: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

def is_valid_strategy_in_repo(strategy_name: str, strategies_df: pd.DataFrame) -> bool:
    """
    Check if a strategy exists in the repository and is enabled.

    Args:
        strategy_name (str): Name of the strategy to check.
        strategies_df (pd.DataFrame): DataFrame returned by repo.get_all_strategies().

    Returns:
        bool: True if the strategy exists and is enabled, False otherwise.
    """
    if strategies_df.empty:
        return False

    # Filtra per il nome della strategia
    match = strategies_df[
        (strategies_df['strategy_name'] == strategy_name) &
        (strategies_df['enabled'] == True)
    ]

    return not match.empty


def is_valid_strategy(ticker: str, strategy: str, best_matrix: pd.DataFrame) -> bool:
    #print(f"[TRACE] Called is_valid_strategy with:\n  Ticker: {ticker}\n  Strategy: {strategy}")

    # Filtra tutte le righe per quel ticker
    ticker_rows = best_matrix[best_matrix['Ticker'] == ticker]
    #print(f"[TRACE] Found {len(ticker_rows)} rows for ticker '{ticker}'.")

    if ticker_rows.empty:
        #print(f"[TRACE] No entries found for ticker '{ticker}'. Returning: True")
        return True

    # Verifica se esiste almeno una riga con strategy uguale e diversa da NO_STRATEGY
    match = ticker_rows[
        (ticker_rows['strategy'] == strategy) &
        (ticker_rows['strategy'] != 'NO_STRATEGY') &
        (ticker_rows['Skip'] != True)
    ]

    result = not match.empty
    #print(f"[TRACE] Match found for strategy '{strategy}' and not NO_STRATEGY: {result}")
    return result



def run_backtest_DaxPattern_vec(data_path,slperc=0.04,tpperc=0.02,capital_allocation=1,show_plot=False,target_strategy=cs_vec.dax_total_signal_vectorized,add_indicators=True):
    '''
    Commenting this code because this kind of approach with caching doesn't work as expected
    May be increase memory drops performance.


    dtype=""
    if add_indicators:
        dtype="enriched"
    else:
        dtype="base"
    _ticker=ti.get_ticker_from_file_path(data_path)
    df=None
    #with(lock):

        if(bu.ticker_exists(_ticker,dtype)):
            df=bu.get_df_from_cache(_ticker,dtype)
            #print(f"[{threading.current_thread().name}] [{target_strategy}] using {_ticker}_{dtype} from cache")
        else:
            df=ti.read_from_csv(data_path)
            df=bu.norm_date_time(df)
            if add_indicators:
                df=de.add_rsi_macd_bb(df)
                df=de.add_smas_long_short(df)
                df=de.add_stoch(df)
            bu.add_df_to_cache(_ticker,df,dtype)
            #print(f"[{threading.current_thread().name}] [{target_strategy}] added {_ticker}_{dtype} to cache")
    df = ti.add_total_signal(df,target_strategy)
    '''
    df=ti.read_from_csv(data_path)
    df=bu.norm_date_time(df)
    if add_indicators:
        df=de.add_rsi_macd_bb(df)
        df=de.add_smas_long_short(df)
        df=de.add_stoch(df)
        df=de.add_adx_column(df)
    df = ti.add_total_signal_vec(df,target_strategy)

    bt = Backtest(df.dropna(), XBacktestingBT,
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
    #Used to solve issue on single backtest
    if False:
        bu.debug_if_contains("DRS",data_path,ctx,results);
    #return cerebro.broker.getvalue()
    return results



def run_backtest_for_all_tickers(tickers_file, data_directory,slperc=0.15,tpperc=0.15,candle_strategy=cs_vec.three_bar_reversal_signal_vectorized,add_indicators=False,optimize=False):
    """Runs backtests for all tickers in tickers.txt and determines the best performer."""
    best_report=None
    # Read tickers from file
    with open(tickers_file, "r") as f:
        tickers = [line.strip() for line in f.readlines()]
    best_matrix=None
    if optimize:
        best_path=f"{data_directory}/best_matrix.csv"
        best_matrix=load_best_matrix(best_path)
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
        if (not optimize or is_valid_strategy(ticker,candle_strategy.__name__,best_matrix)):
            data_path = f"{data_directory}/{ticker}.csv"  # Path to CSV file
            try:
                report = run_backtest_DaxPattern_vec(data_path, slperc=slperc,tpperc=tpperc,capital_allocation=1,target_strategy=candle_strategy,add_indicators=add_indicators)
                report["strategy"]=candle_strategy.__name__
                results[ticker] = report[field_selected]
                reports[ticker] = report
            except Exception as e:
                print(f"Error running backtest for {ticker}: {repr(e)}")
                traceback.print_exc()

        #else:
            #print(f"skip {ticker} {candle_strategy.__name__}")
    # Find the best-performing ticker
    if results:
        df=bu.save_reports_to_df(reports)
        #fbest_ticker = max(results, key=results.get)
        #print("\nBest performing ticker:", best_ticker)
        #print(f"strategy {candle_strategy}")
        #print(f"{field_selected}:", results[best_ticker])
        #print(results)
        return df
    return None

def exec_analysis(base_path="../",slperc=0.15, tpperc=1.0, optimize=False):
    df = None
    for strategy in cs_vec.candlestick_strategies:
        df1 = run_backtest_for_all_tickers(f'{base_path}../data/tickers.txt', f'{base_path}../data/', slperc=slperc,tpperc=tpperc,candle_strategy=strategy,optimize=optimize)
        df = bu.append_df(df, df1)
    for strategy in ins_vec.indicators_strategy:
        df1 = run_backtest_for_all_tickers(f'{base_path}../data/tickers.txt', f'{base_path}../data/', slperc=slperc,tpperc=tpperc,candle_strategy=strategy,
                                           add_indicators=True,optimize=optimize)
        df = bu.append_df(df, df1)
    return df

def exec_analysis_parallel(base_path="../", slperc=0.15, tpperc=1.0, optimize=False):
    """
    Executes backtesting for all tickers using both candlestick and indicator strategies in parallel.

    Args:
        base_path (str): Base path for data files and output.
        slperc (float): Stop loss percentage.
        tpperc (float): Take profit percentage.

    Returns:
        DataFrame: Combined results from all backtests.
    """
    df = pd.DataFrame()
    futures = []
    tickers_file = f'{base_path}../data/tickers.txt'
    data_dir = f'{base_path}../data/'

    # Instantiate repo
    repo = StrategyRepo()
    strategies_df = repo.get_all_strategies()  # recupera tutte le strategie attuali

    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures = []

        # --- Candlestick strategies ---
        for strategy in cs_vec.candlestick_strategies:
            strategy_name = strategy.__name__.replace("_", " ")
            if is_valid_strategy_in_repo(strategy_name, strategies_df):
                futures.append(executor.submit(
                    run_backtest_for_all_tickers,
                    tickers_file, data_dir, slperc, tpperc, strategy, False, optimize
                ))
            else:
                print(f"Skipping disabled strategy: {strategy_name}")

        # --- Indicator strategies ---
        for strategy in ins_vec.indicators_strategy:
            strategy_name = strategy.__name__.replace("_", " ")
            if is_valid_strategy_in_repo(strategy_name, strategies_df):
                futures.append(executor.submit(
                    run_backtest_for_all_tickers,
                    tickers_file, data_dir, slperc, tpperc, strategy, True, optimize
                ))
            else:
                print(f"Skipping disabled strategy: {strategy_name}")

        # --- Collect results ---
        for future in concurrent.futures.as_completed(futures):
            try:
                df1 = future.result()
                df = bu.append_df(df, df1)
            except Exception as e:
                print(f"Error during backtest execution: {repr(e)}")

    return df

def exec_analysis_parallel_threads(base_path="../", slperc=0.15, tpperc=1.0, optimize=False):
    """
    Executes backtesting for all tickers using both candlestick and indicator strategies in parallel (threads).

    Args:
        base_path (str): Base path for data files and output.
        slperc (float): Stop loss percentage.
        tpperc (float): Take profit percentage.
        optimize (bool): Whether to optimize parameters.

    Returns:
        DataFrame: Combined results from all backtests.
    """
    df = pd.DataFrame()
    futures = []
    tickers_file = f'{base_path}../data/tickers.txt'
    data_dir = f'{base_path}../data/'

    # Use threads instead of processes
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        # Candlestick strategies
        for strategy in cs_vec.candlestick_strategies:
            futures.append(executor.submit(
                run_backtest_for_all_tickers,
                tickers_file, data_dir, slperc, tpperc, strategy, False, optimize
            ))

        # Indicator strategies
        for strategy in ins_vec.indicators_strategy:
            futures.append(executor.submit(
                run_backtest_for_all_tickers,
                tickers_file, data_dir, slperc, tpperc, strategy, True, optimize
            ))

        # Collect results
        for future in concurrent.futures.as_completed(futures):
            try:
                df1 = future.result()
                df = bu.append_df(df, df1)
            except Exception as e:
                print(f"Error during backtest execution: {repr(e)}")

    return df
def exec_analysis_and_save_results(base_path='../', slperc=0.15, tpperc=1.0, parallel=True,optimize=False):
    start_time = time.time()  # ⏱️ Start timer

    # Choose execution mode
    if parallel:
        df = exec_analysis_parallel(base_path, slperc=slperc, tpperc=tpperc,optimize=optimize)
    else:
        df = exec_analysis(base_path, slperc=slperc, tpperc=tpperc,optimize=optimize)

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
            f"* Mode:       {'Parallel' if parallel else 'Sequential':<42}\n" +
            f"* Stop Loss:  {slperc:<42}\n" +
            f"* Take Profit:{tpperc:<42}\n" +
            f"* Duration:   {elapsed_time:.2f} seconds{'':<29}\n" +
            "*" * 60 + "\n"
    )
    print(summary)  # Optional: print to console
    return summary

def test0():
    s2 = run_backtest_DaxPattern_vec("../../data/DRS.csv", slperc=0.15, tpperc=0.40, target_strategy=ins_vec.weekly_breakout_vectorized,
                                 capital_allocation=10000, show_plot=True,add_indicators=True)
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

def test2():
    exec_analysis_and_save_results(parallel=True,optimize=False,base_path="../")
if __name__ == "__main__":
    test0()