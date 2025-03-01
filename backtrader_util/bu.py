import backtrader.analyzers as btanalyzers
import pandas as pd
from datetime import datetime

def get_timestamp():
    return datetime.now().strftime("%Y%m%d%H%M")

def append_df(df1,df2):
    if df1 is None:
        return df2
    if df2 is None:
        return df1
    df_final = pd.concat([df1, df2], ignore_index=True)
    return df_final

def load_csv(input_file="backtest_results.csv"):
    """
    Loads a CSV file into a Pandas DataFrame and displays it in a scrollable view.
    """
    df = pd.read_csv(input_file)
    pd.set_option("display.max_rows", None)  # Show all rows
    pd.set_option("display.max_columns", None)  # Show all columns
    #print(df)
    return df


def save_reports_to_csv(reports, output_file="backtest_results.csv"):
    """
    Converts the reports dictionary into a Pandas DataFrame and saves it to a CSV file.
    Each row corresponds to a ticker, with columns for each report key, the ticker name, and a timestamp.
    """

    # List to store data for each ticker
    df=save_reports_to_df(reports)

    # Save to CSV
    df.to_csv(output_file, index=False)
    return df

def save_reports_to_df(reports):
    """
    Converts the reports dictionary into a Pandas DataFrame.
    Each row corresponds to a ticker, with columns for each report key, the ticker name, and a timestamp.
    """

    # List to store data for each ticker
    data = []

    # Current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for ticker, report in reports.items():
        row = {"Ticker": ticker, "Timestamp": timestamp}  # Start with Ticker and Timestamp

        # Ensure report is a dictionary
        if isinstance(report, dict):
            row.update(report)
        elif isinstance(report,pd.Series):
            row.update(report.to_dict())
        else:
            row["Value"] = report  # Store float or other non-dict values with a generic key

        data.append(row)

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(data)
    return df

def print_backtest_report(results):
    """
    Prints a detailed backtest report with aggregated performance metrics.
    """
    strategy_results = results[0]  # First instance of the strategy

    # Extract analyzers
    returns = strategy_results.analyzers.returns.get_analysis()
    drawdown = strategy_results.analyzers.drawdown.get_analysis()
    trade_analysis = strategy_results.analyzers.trade.get_analysis()
    sharpe_ratio = strategy_results.analyzers.sharpe.get_analysis()

    # Portfolio Value
    final_portfolio_value = strategy_results.broker.get_value()

    # Aggregated returns
    agg_returns = returns['rtot'] * 100  # Convert to percentage

    # Drawdown metrics
    max_drawdown = drawdown['max']['drawdown']  # Maximum drawdown
    avg_drawdown = drawdown['drawdown']  # Average drawdown

    # Trade statistics
    num_trades = trade_analysis.total.closed  # Number of closed trades
    win_rate = (trade_analysis.won.total / num_trades * 100) if num_trades > 0 else 0

    # Convert AutoOrderedDict to lists before summing
    won_pnl = list(trade_analysis.won.pnl.values()) if trade_analysis.won.total > 0 else [0]
    lost_pnl = list(trade_analysis.lost.pnl.values()) if trade_analysis.lost.total > 0 else [0]

    # Best and Worst Trade (percentage returns)
    best_trade = max(won_pnl) if won_pnl else 0
    worst_trade = min(lost_pnl) if lost_pnl else 0
    avg_trade = (sum(won_pnl) + sum(lost_pnl)) / num_trades if num_trades > 0 else 0

    # Average return per trade
    avg_return_per_trade = agg_returns / num_trades if num_trades > 0 else 0

    # Print the report
    print("\n===== BACKTEST REPORT =====")
    print(f"Sharpe Ratio: {sharpe_ratio.get('sharperatio', 'N/A')}")
    print(f"Aggregated Returns: {agg_returns:.2f}%")
    print(f"Final Portfolio Value: {final_portfolio_value:.2f}")
    print(f"Total Trades: {num_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Maximum Drawdown: {max_drawdown:.2f}")
    print(f"Average Drawdown: {avg_drawdown:.2f}%")
    print(f"Average Return per Trade: {avg_return_per_trade:.2f}%")
    print(f"Best Trade: {best_trade:.2f}%")
    print(f"Worst Trade: {worst_trade:.2f}%")
    print(f"Average Trade: {avg_trade:.2f}%")

def at_least_a_trade(total):
    at_least=True
    if "total" in total and "open" in total and "closed" not in total:
        at_least=False
    elif (total.total==1 and total.closed==0) or total.total==0:
        at_least=False
    return at_least

def get_backtest_report(results):
    """
    Returns a detailed backtest report as a dictionary with aggregated performance metrics.
    """
    strategy_results = results[0]  # First instance of the strategy
    last_action=strategy_results.last_position
    # Extract analyzers
    returns = strategy_results.analyzers.returns.get_analysis()
    drawdown = strategy_results.analyzers.drawdown.get_analysis()
    trade_analysis = strategy_results.analyzers.trade.get_analysis()
    sharpe_ratio = strategy_results.analyzers.sharpe.get_analysis()

    # Portfolio Value
    final_portfolio_value = strategy_results.broker.get_value()

    # Aggregated returns
    agg_returns = returns['rtot'] * 100  # Convert to percentage

    # Drawdown metrics
    max_drawdown = drawdown['max']['drawdown']  # Maximum drawdown
    avg_drawdown = drawdown['drawdown']  # Average drawdown

    # Trade statistics
    if at_least_a_trade(trade_analysis.total):
        num_trades = trade_analysis.total.closed  # Number of closed trades
        win_rate = (trade_analysis.won.total / num_trades * 100) if num_trades > 0 else 0

        # Convert AutoOrderedDict to lists before summing
        won_pnl = list(trade_analysis.won.pnl.values()) if trade_analysis.won.total > 0 else [0]
        lost_pnl = list(trade_analysis.lost.pnl.values()) if trade_analysis.lost.total > 0 else [0]

        # Best and Worst Trade (percentage returns)
        best_trade = max(won_pnl) if won_pnl else 0
        worst_trade = min(lost_pnl) if lost_pnl else 0
        avg_trade = (sum(won_pnl) + sum(lost_pnl)) / num_trades if num_trades > 0 else 0

        # Average return per trade
        avg_return_per_trade = agg_returns / num_trades if num_trades > 0 else 0

        # Create and return a dictionary
        report = {
            "Sharpe Ratio": round(sharpe_ratio.get('sharperatio', 'N/A'),2),
            "Aggregated Returns (%)": round(agg_returns, 2),
            "Final Portfolio Value": round(final_portfolio_value, 2),
            "Total Trades": num_trades,
            "Win Rate (%)": round(win_rate, 2),
            "Maximum Drawdown (%)": round(max_drawdown, 2),
            "Average Drawdown (%)": round(avg_drawdown, 2),
            "Average Return per Trade (%)": round(avg_return_per_trade, 2),
            "Best Trade": round(best_trade, 2),
            "Worst Trade": round(worst_trade, 2),
            "Average Trade ": round(avg_trade, 2)
        }
        report["Last Action"] = last_action
        return report
    else:
        # Create and return a dictionary
        report = {
            "Sharpe Ratio": 0,
            "Aggregated Returns (%)": 0,
            "Final Portfolio Value": round(0, 2),
            "Total Trades": 0,
            "Win Rate (%)": round(0, 2),
            "Maximum Drawdown (%)": round(0, 2),
            "Average Drawdown (%)": round(0, 2),
            "Average Return per Trade (%)": round(0, 2),
            "Best Trade": round(0, 2),
            "Worst Trade": round(0, 2),
            "Average Trade ": round(0, 2)
        }
        report["Last Action"]=last_action
        return report

def norm_date_time(df):
    # Sample DataFrame with a datetime column including a timezone offset
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S%z',utc=True)
    df['Date'] = df['Date'].dt.tz_localize(None)
    # Ensure the datetime column is the index
    df.set_index('Date', inplace=True)
    return df

def format_value(x):
    """
    Formats a value to two decimal places if numeric, or to 'YYYY-MM-DD' if it's a date.

    Args:
        x: The value to format.

    Returns:
        str: The formatted value as a string.
    """
    if isinstance(x, (int, float)):
        return "{:.2f}".format(x)  # Format to two decimal places
    elif isinstance(x, pd.Timestamp):
        return x.strftime('%Y-%m-%d')  # Format date to 'YYYY-MM-DD'
    return x

cache={"stop_loss":0.15,"take_profit":0.15}



if __name__ == "__main__":
    print(format_value(2.333))

