import stock.indicators_signal_vec as ins_vec
import stock.candle_signal_vec as cs_vec
import backtrader_util.bu as bu
import pandas as pd
from typing import List

def get_strategy_names() -> List[str]:
    """
    Combines two predefined lists of strategy functions and returns their names
    in a readable format by replacing underscores with spaces.

    Returns:
        List[str]: A list of strategy names as strings, formatted for display.
    """
    # Combine the two predefined lists of strategy functions
    backtesting_functions = ins_vec.indicators_strategy + cs_vec.candlestick_strategies

    # Convert function names to readable strings by replacing underscores with spaces
    algorithm_names = [f.__name__.replace("_", " ") for f in backtesting_functions]

    # Return the list of readable strategy names
    return algorithm_names


def save_cache(stop_loss, take_profit):
    bu.cache["stop_loss"] = stop_loss
    bu.cache["take_profit"] = take_profit
    return "Values saved!"

def generate_best_matrix(win_rate, ret, trades, strategies):
    """
    Generate the best strategy matrix based on filters and selected strategies.

    Args:
        win_rate (float): Minimum win rate threshold.
        ret (float): Minimum return threshold.
        trades (int): Minimum number of trades.
        strategies (list[str]): List of selected strategies to consider.

    Returns:
        pd.DataFrame: Filtered and annotated strategy matrix with Skip column.
    """
    # Replace spaces with underscores in strategy names
    strategies = [s.replace(" ", "_") for s in strategies]

    # Load the report
    df = pd.read_csv("../results/report.csv")
    print("Selected strategies:", strategies)

    # Filter based on thresholds
    query = (
        f"df[(df['Win Rate [%]'] > {win_rate}) & "
        f"(df['Return [%]'] > {ret}) & "
        f"(df['# Trades'] >= {trades})]"
    )

    try:
        filtered = eval(query, {"df": df, "pd": pd})
        selected = filtered[['Ticker', 'strategy', 'Win Rate [%]', 'Return [%]', '# Trades']].copy()
    except Exception as e:
        selected = pd.DataFrame({"Error": [str(e)]})

    # Identify tickers missing from filtered results
    all_tickers = set(df['Ticker'].unique())
    selected_tickers = set(selected['Ticker'].unique())
    missing = all_tickers - selected_tickers

    # Create placeholder rows with NO_STRATEGY
    missing_df = pd.DataFrame([{'Ticker': t, 'strategy': 'NO_STRATEGY'} for t in missing])

    # Merge results
    result_df = pd.concat([selected, missing_df], ignore_index=True).sort_values('Ticker')

    # --- NEW: Add Skip column ---
    # Initialize Skip = False
    result_df['Skip'] = False
    # Set Skip = True for rows whose strategy is not in the selected list and is not NO_STRATEGY
    mask = (~result_df['strategy'].isin(strategies)) & (result_df['strategy'] != 'NO_STRATEGY')
    result_df.loc[mask, 'Skip'] = True

    # Save the final matrix
    result_df.to_csv("../data/best_matrix.csv", index=False)
    return result_df

def run_backtest(ticker, function_name):
    function_name = function_name.replace(" ", "_")

    merged_functions_list = cs_vec.candlestick_strategies + ins_vec.indicators_strategy
    functions_dict = {func.__name__: func for func in merged_functions_list}
    func = functions_dict[function_name]

    # True se la funzione appartiene agli indicator strategy
    add_indicators = func in ins_vec.indicators_strategy

    res = trades.run_backtest_DaxPattern_vec(
        f"../data/{ticker}.csv",
        slperc=bu.cache["stop_loss"],
        tpperc=bu.cache["take_profit"],
        capital_allocation=1,
        show_plot=True,
        target_strategy=func,
        add_indicators=add_indicators
    )
    return res

