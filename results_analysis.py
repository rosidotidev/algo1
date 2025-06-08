import os
import pandas as pd
import numpy as np

def load_all_results(results_dir='../results/'):
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

    return all_df


def analyze_top_strategies(results_dir='../results/', sort_by_metrics=None):
    if sort_by_metrics is None:
        sort_by_metrics = ['Return [%]']  # default fallback

    all_df = load_all_results(results_dir)

    required_cols = ['Ticker', 'strategy', 'Return [%]', 'Win Rate [%]', '# Trades']
    all_df = all_df.dropna(subset=required_cols)

    grouped = all_df.groupby(['Ticker', 'strategy']).agg({
        'Return [%]': 'mean',
        'Win Rate [%]': 'mean',
        '# Trades': 'mean'
    }).reset_index()

    grouped.rename(columns={'# Trades': 'Avg Trades'}, inplace=True)

    missing = [m for m in sort_by_metrics if m not in grouped.columns]
    if missing:
        print(f"Error: These sort metrics are missing: {missing}")
        return grouped

    top_performers = grouped.sort_values(
        by=sort_by_metrics,
        ascending=[False] * len(sort_by_metrics)
    )

    display_cols = ['Ticker', 'strategy'] + sort_by_metrics + ['Return [%]', 'Win Rate [%]', 'Avg Trades']
    display_cols = list(dict.fromkeys(display_cols))

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.precision", 2)

    print(f"\n******** Top Ticker+Strategy by {sort_by_metrics} ********\n")
    print(top_performers[display_cols].head(10))
    return top_performers


def analyze_top_strategies_by_product(results_dir='../results/', product_metrics=None):
    if product_metrics is None:
        product_metrics = ['Return [%]', 'Win Rate [%]']  # default

    all_df = load_all_results(results_dir)

    required_cols = ['Ticker', 'strategy'] + product_metrics
    all_df = all_df.dropna(subset=required_cols)

    grouped = all_df.groupby(['Ticker', 'strategy']).agg({col: 'mean' for col in product_metrics}).reset_index()

    # Create new column: product of selected metrics
    grouped['Metric Product'] = grouped[product_metrics].prod(axis=1)

    top_performers = grouped.sort_values(by='Metric Product', ascending=False)

    display_cols = ['Ticker', 'strategy', 'Metric Product'] + product_metrics

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.precision", 2)

    print(f"\n******** Top Ticker+Strategy by Product of {product_metrics} ********\n")
    print(top_performers[display_cols].head(10))
    return top_performers


if __name__ == "__main__":
    selected_metrics = ['Win Rate [%]', 'Return [%]']

    # First logic: sort by individual metrics in order
    analyze_top_strategies(sort_by_metrics=selected_metrics)

    # Second logic: sort by product of metrics
    analyze_top_strategies_by_product(product_metrics=selected_metrics)
