from backtesting import Backtest
import stock.ticker as ti
from backtrader_util import bu
import stock.simple_signal_vec as simple
import stock.enriched_signal_vec as enriched
import data.data_enricher as de
import time
import datetime
import os
import shutil
from stock.strategy_repo import StrategyRepo
from stock.x_backtesting_bt import XBacktestingBT
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import traceback
from strategy.ticker_stategy_repo import TickerStrategyRepo
from strategy.ticker_strategy import TickerStrategy
from strategy.x_strategy_bt import XStrategyBT
import stock.plot as my_plot


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
    if  best_matrix is None:
        return True
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



def run_backtest_DaxPattern_vec(data_path,slperc=0.04,tpperc=0.02,capital_allocation=1,show_plot=False,target_strategy=simple.dax_total_signal_vectorized,add_indicators=True):

    df=ti.read_from_csv(data_path)
    return run_backtest_DaxPattern_vec_df(df,slperc=slperc,tpperc=tpperc,capital_allocation=capital_allocation,show_plot=show_plot,target_strategy=target_strategy,add_indicators=add_indicators)

def run_backtest_DaxPattern_vec_df(df, slperc=0.04, tpperc=0.02, capital_allocation=1, show_plot=False,
                                    target_strategy=simple.dax_total_signal_vectorized, add_indicators=True):
    df = bu.norm_date_time(df.copy())
    if add_indicators:
        df = de.add_rsi_macd_bb(df)
        df = de.add_smas_long_short(df)
        df = de.add_stoch(df)
        df = de.add_adx_column(df)
    df = ti.add_total_signal_vec(df, target_strategy)

    bt = Backtest(df.dropna(), XBacktestingBT,
                  cash=10000,
                  finalize_trades=True,
                  exclusive_orders=True,
                  commission=.002)

    # Esegui il backtest
    ctx = {}
    results = bt.run(slperc=slperc, tpperc=tpperc, df=df, ctx=ctx)
    ctx.update(results.to_dict())
    # print(f" ctx {results}")
    if show_plot:
        bt.plot(filename=None)
    for key, value in ctx.items():
        results[key] = bu.format_value(value)
    # Used to solve issue on single backtest
    if False:
        bu.debug_if_contains("DRS", data_path, ctx, results);
    # return cerebro.broker.getvalue()
    return results

def run_x_backtest_DaxPattern_vec(data_path, slperc=0.04, tpperc=0.02, capital_allocation=1, show_plot=False,
                                    target_strategy=simple.dax_total_signal_vectorized, add_indicators=True, open_browser=True):
    df = ti.read_from_csv(data_path)
    df = bu.norm_date_time(df.copy())
    if add_indicators:
        df = de.add_rsi_macd_bb(df)
        df = de.add_smas_long_short(df)
        df = de.add_stoch(df)
        df = de.add_adx_column(df)

    ticker=ti.get_ticker_from_file_path(data_path)
    dir_path=ti.get_data_path_from_file_path(data_path)
    tsr=bu.cache["context"]["TickerStrategyRepo"]

    tsd=tsr.get_by_ticker_and_strategy(ticker,target_strategy.__name__)

    if tsd:
        ts = TickerStrategy(
            ticker=tsd["ticker"],
            strategy_func=tsd["strategy_func"],
            params=tsd["params"]
        )
    else:
        ts = None

    bt = Backtest(df.dropna(), XStrategyBT,
                  cash=10000,
                  finalize_trades=True,
                  exclusive_orders=True,
                  commission=.002)

    # Esegui il backtest
    ctx = {}
    results = bt.run(slperc=slperc, tpperc=tpperc, ts=ts,df=df, ctx=ctx)
    ctx.update(results.to_dict())
    # print(f" ctx {results}")

    for key, value in ctx.items():
        results[key] = bu.format_value(value)

    if show_plot:
        fig=bt.plot(filename=None,open_browser=open_browser)
        bu.cache["context"]["backtest_plot"]=fig
    # Used to solve issue on single backtest
    if False:
        bu.debug_if_contains("DRS", data_path, ctx, results);
    # return cerebro.broker.getvalue()
    return results


def run_backtest_for_all_tickers(tickers_file, data_directory,slperc=0.15,tpperc=0.15,candle_strategy=simple.three_bar_reversal_signal_vectorized,add_indicators=False,optimize=False):
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

def exec_analysis_sequential_new(base_path="../", slperc=0.15, tpperc=1.0, optimize=False):
    """
    Executes backtesting for all tickers using both candlestick and indicator strategies in parallel.

    Args:
        base_path (str): Base path for data files and output.
        slperc (float): Stop loss percentage.
        tpperc (float): Take profit percentage.

    Returns:
        DataFrame: Combined results from all backtests.
    """
    df = pd.DataFrame()  # Initialize empty DataFrame for final results
    futures = []  # Placeholder for parallel execution (not used here)
    tickers_file = f'{base_path}../data/tickers.txt'  # Path to tickers file
    data_dir = f'{base_path}../data/'  # Base data directory

    # Read tickers from file
    with open(tickers_file, "r") as f:
        tickers = [line.strip() for line in f.readlines()]

    best_matrix = None
    if optimize:
        best_path = f"{data_dir}/best_matrix.csv"  # Path to best matrix for optimization
        best_matrix = load_best_matrix(best_path)  # Load best matrix if optimization is enabled

    # Instantiate the strategy repository
    repo = StrategyRepo(f'{base_path}../data/strategy_repo.csv')
    # Retrieve all strategies from the repository
    strategies_df = repo.get_all_strategies()
    # Get all available strategy functions (callable objects)
    all_functions = StrategyRepo.get_all_available_strategy_functions()

    # Loop over each ticker
    for ticker in tickers:

        # Loop over each strategy in the repository
        df = execute_all_strategies_for_single_ticker(all_functions, best_matrix, data_dir, df, optimize, slperc,
                                                      strategies_df, ticker, tpperc)

    return df


def exec_analysis_parallel_new(base_path="../", slperc=0.15, tpperc=1.0, optimize=False, max_workers=6):
    """
    Executes backtesting for all tickers using both candlestick and indicator strategies in parallel.
    Each ticker is processed in a separate process.

    Args:
        base_path (str): Base path for data files and output.
        slperc (float): Stop loss percentage.
        tpperc (float): Take profit percentage.
        optimize (bool): Whether to use optimization filtering.
        max_workers (int or None): Max number of parallel workers. Defaults to # of CPUs.

    Returns:
        DataFrame: Combined results from all backtests.
    """
    df = pd.DataFrame()
    tickers_file = f'{base_path}../data/tickers.txt'
    data_dir = f'{base_path}../data/'

    # Read tickers from file
    with open(tickers_file, "r") as f:
        tickers = [line.strip() for line in f.readlines()]

    best_matrix = None
    if optimize:
        print(f" optimize: {optimize}")
        best_path = f"{data_dir}/best_matrix.csv"
        best_matrix = load_best_matrix(best_path)

    # Instantiate the strategy repository
    repo = StrategyRepo(f'{base_path}../data/strategy_repo.csv')
    strategies_df = repo.get_all_strategies()
    all_functions = StrategyRepo.get_all_available_strategy_functions()

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for ticker in tickers:
            futures.append(
                executor.submit(
                    execute_all_strategies_for_single_ticker,
                    all_functions,
                    best_matrix,
                    data_dir,
                    pd.DataFrame(),   # start empty df per ticker
                    optimize,
                    slperc,
                    strategies_df,
                    ticker,
                    tpperc
                )
            )

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                df_part = future.result()
                df = bu.append_df(df, df_part)  # merge partial results
            except Exception as e:
                print(f"Error in parallel execution: {repr(e)}")
                traceback.print_exc()

    return df



def execute_all_strategies_for_single_ticker(all_functions, best_matrix, data_dir, df, optimize, slperc, strategies_df,
                                             ticker, tpperc):
    print(f"Working on ticker {ticker} ")
    bu.cache["context"]["TickerStrategyRepo"] = TickerStrategyRepo("../data/")
    for ind, row in strategies_df.iterrows():
        func_name = row["strategy_name"]  # Strategy name as string
        func_ref = func_name.replace(" ", "_")  # Convert name to function-like format
        enrich_flag = row["enrich"]  # Boolean flag indicating if additional indicators should be added

        # Check if the strategy is enabled in the repository
        if is_valid_strategy_in_repo(func_name, strategies_df):
            # Find the actual function object in the list of available functions
            strategy_func = next((f for f in all_functions if f.__name__ == func_ref), None)

            # Skip if the function is not found
            if strategy_func is None:
                print(f"Funzione {func_name} non trovata tra le available strategy functions.")
                continue

            # If optimization is off or the strategy is valid according to the best_matrix
            if (not optimize or is_valid_strategy(ticker, strategy_func.__name__, best_matrix)):
                data_path = f"{data_dir}/{ticker}.csv"  # Path to CSV file for ticker
                try:
                    # print(f"Running {ticker} and strategy {func_ref}")
                    # ticker_history_data = ti.read_from_csv(data_path)
                    # Run the backtest

                    #report = run_backtest_DaxPattern_vec(
                    report = run_x_backtest_DaxPattern_vec(
                        data_path,
                        slperc=slperc,
                        tpperc=tpperc,
                        capital_allocation=1,
                        target_strategy=strategy_func,
                        add_indicators=enrich_flag
                    )
                    report["strategy"] = strategy_func.__name__  # Add strategy name to report
                    reports = {}
                    reports[ticker] = report
                    df1 = bu.save_reports_to_df(reports)
                    df = bu.append_df(df, df1)

                except Exception as e:
                    # Print error traceback if backtest fails
                    print(f"Error running backtest for {ticker}: {repr(e)}")
                    traceback.print_exc()
            #else:
                # Skip disabled strategies
                #print(f"Skipping disabled strategy: {func_name} for ticker {ticker}")
    return df



def exec_analysis(base_path="../",slperc=0.15, tpperc=1.0, optimize=False):
    df = None
    for strategy in simple.candlestick_strategies:
        df1 = run_backtest_for_all_tickers(f'{base_path}../data/tickers.txt', f'{base_path}../data/', slperc=slperc,tpperc=tpperc,candle_strategy=strategy,optimize=optimize)
        df = bu.append_df(df, df1)
    for strategy in enriched.indicators_strategy:
        df1 = run_backtest_for_all_tickers(f'{base_path}../data/tickers.txt', f'{base_path}../data/', slperc=slperc,tpperc=tpperc,candle_strategy=strategy,
                                           add_indicators=True,optimize=optimize)
        df = bu.append_df(df, df1)
    return df

def exec_analysis_and_save_results(base_path='../', slperc=0.15, tpperc=1.0, parallel=True,optimize=False):
    start_time = time.time()  # ⏱️ Start timer

    # Choose execution mode
    if parallel:
        df = exec_analysis_parallel_new(base_path, slperc=slperc, tpperc=tpperc,optimize=optimize)
    else:
        df = exec_analysis_sequential_new(base_path, slperc=slperc, tpperc=tpperc, optimize=optimize)
        #df = exec_analysis(base_path, slperc=slperc, tpperc=tpperc,optimize=optimize)

    # Save results
    today = datetime.datetime.now().strftime("%Y%m%d")
    file_today = f"{base_path}../results/report_{today}.csv"
    file_latest = f"{base_path}../results/report.csv"

    df.to_csv(file_today, index=False)
    shutil.copy(file_today, file_latest)

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
    s2 = run_backtest_DaxPattern_vec("../../data/DRS.csv", slperc=0.15, tpperc=0.40, target_strategy=simple.filled_bar_vectorized,
                                 capital_allocation=10000, show_plot=True,add_indicators=False)

def test2():
    exec_analysis_and_save_results(parallel=True,optimize=False,base_path="../")

def test3():
    import time

    start_time = time.perf_counter()
    df = exec_analysis_sequential_new()
    # df=exec_analysis_parallel()
    exec_time = time.perf_counter()

    print(df)
    print(f"Data exec time: {exec_time - start_time:.2f} seconds")

def test4():
    bu.cache["context"]["TickerStrategyRepo"]=TickerStrategyRepo("../../data/")
    func_strategy=simple.ema_pullback_strategy
    #func_strategy=enriched.adx_trend_breakout_10_35
    add_ind=StrategyRepo.get_add_indicators_flag(func_strategy)
    s2 = run_x_backtest_DaxPattern_vec("../../data/MGNI.csv", slperc=0.05, tpperc=0.08, target_strategy=func_strategy,
                                 capital_allocation=10000, show_plot=True,add_indicators=add_ind,open_browser=False)

    fig=bu.cache["context"]["backtest_plot"]
    my_plot.plot_x(fig)
    strategy=s2._strategy
    print(strategy.df)
    print(strategy.closed_trades)
    df=strategy.df
    df['trades']=0
    print(strategy.closed_trades[0])
    for trade in strategy.closed_trades:
        entry_time = trade.entry_time
        if entry_time in df.index:
            if trade.is_long:
                df.at[entry_time, "trades"] = 2
            elif trade.is_short:
                df.at[entry_time, "trades"] = 1
    my_plot.plot_numeric_flags_signals(strategy.df,['Close','ema_short','ema_long'],flag_cols=['cross','trend_up','trend_down'],signals=df['trades'])
    a=1
if __name__ == "__main__":
    test4()