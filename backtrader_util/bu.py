import os
import pandas as pd
from datetime import datetime
import backtrader_util.bu as bu


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

def get_csv_files(directory):
    """list of csv file in a directory"""
    try:
        files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        return files
    except FileNotFoundError:
        return ["Directory not found"]


cache = {
    "stop_loss": 0.10,
    "take_profit": 0.15,
    "df": {}
}


def reset_df_cache():
    """Reset the 'df' entry in cache to an empty dict."""
    bu.cache["df"] = {}


def ticker_exists(ticker: str, dtype: str) -> bool:
    """
    Check if ticker with exact dtype key exists in cache.
    If dtype is None, check if ticker_base or ticker_enriched exists.
    """
    if "df" not in bu.cache:
        return False

    key = f"{ticker}_{dtype}"
    return key in bu.cache["df"]



def add_df_to_cache(ticker: str, df: pd.DataFrame, dtype: str='base'):
    """
    Add a DataFrame to cache with key ticker_dtype.
    Overwrites if key already exists. check on max length
    """
    if "df" not in bu.cache:
        bu.cache["df"] = {}
    key = f"{ticker}_{dtype}"
    if len(bu.cache["df"]) <= 30:
        bu.cache["df"][key] = df.copy()



def get_df_from_cache(ticker: str, dtype: str) -> pd.DataFrame | None:
    """
    Get DataFrame from cache by ticker and dtype key.
    Returns None if not found.
    """
    key = f"{ticker}_{dtype}"
    return bu.cache["df"].get(key)

def debug_if_contains(target:str,data: str, ctx: dict, results: pd.Series):
    if target in data.upper():  # case-insensitive
        print("=" * 40)
        print(f"data path {data}")
        print("=" * 40)
        print("CTX:")
        for k, v in ctx.items():
            print(f"  {k}: {v}")
        print("-" * 40)
        print("RESULTS:")
        print(results.to_string())  # stampa la Series in formato leggibile
        print("=" * 40)

if __name__ == "__main__":
    print(format_value(2.333))

