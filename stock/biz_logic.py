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

def generate_best_matrix(win_rate, ret, trades):
    df = pd.read_csv("../results/report.csv")

    query = (
        f"df[(df['Win Rate [%]'] > {win_rate}) & "
        f"(df['Return [%]'] > {ret}) & "
        f"(df['# Trades'] >= {trades})]"
    )

    try:
        filtered = eval(query, {"df": df, "pd": pd})
        selected = filtered[['Ticker', 'strategy','Win Rate [%]','Return [%]','# Trades']].copy()
    except Exception as e:
        selected = pd.DataFrame({"Error": [str(e)]})

    all_tickers = set(df['Ticker'].unique())
    selected_tickers = set(selected['Ticker'].unique())
    missing = all_tickers - selected_tickers
    missing_df = pd.DataFrame([{'Ticker': t, 'strategy': 'NO_STRATEGY'} for t in missing])
    result_df = pd.concat([selected, missing_df], ignore_index=True).sort_values('Ticker')
    result_df.to_csv("../data/best_matrix.csv", index=False)
    return result_df