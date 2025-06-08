import os
import pandas as pd

def analyze_top_strategies(results_dir='../results/', sort_by_metrics=None):
    if sort_by_metrics is None:
        sort_by_metrics = ['Return [%]']  # default fallback

    # List all CSV files, excluding 'report.csv'
    csv_files = [
        f for f in os.listdir(results_dir)
        if f.endswith('.csv') and f != 'report.csv'
    ]

    if not csv_files:
        print("No CSV files found.")
        return pd.DataFrame()

    # Load and concatenate all CSVs
    all_df = pd.concat([
        pd.read_csv(os.path.join(results_dir, f)) for f in csv_files
    ], ignore_index=True)

    # Drop rows missing key columns
    required_cols = ['Ticker', 'strategy', 'Return [%]', 'Win Rate [%]', '# Trades']
    all_df = all_df.dropna(subset=required_cols)

    # Group and calculate mean metrics
    grouped = all_df.groupby(['Ticker', 'strategy']).agg({
        'Return [%]': 'mean',
        'Win Rate [%]': 'mean',
        '# Trades': 'mean'
    }).reset_index()

    # Rename for clarity
    grouped.rename(columns={'# Trades': 'Avg Trades'}, inplace=True)

    # Validate sort_by_metrics
    missing = [m for m in sort_by_metrics if m not in grouped.columns]
    if missing:
        print(f"Error: These sort metrics are missing: {missing}")
        return grouped

    # Sort by provided metrics
    top_performers = grouped.sort_values(
        by=sort_by_metrics,
        ascending=[False] * len(sort_by_metrics)
    )

    # Define columns to display (ensure no duplicates)
    display_cols = ['Ticker', 'strategy'] + sort_by_metrics + ['Return [%]', 'Win Rate [%]', 'Avg Trades']
    display_cols = list(dict.fromkeys(display_cols))

    # Display settings for full output
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.precision", 2)

    print(f"\n******** Top Ticker+Strategy by {sort_by_metrics} ********\n")
    print(top_performers[display_cols].head(40))
    return top_performers


if __name__ == "__main__":
    # Choose the metric(s) to rank by
    selected_metrics = [ 'Return [%]','Win Rate [%]']  # Can be one or more
    analyze_top_strategies(sort_by_metrics=selected_metrics)
