import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_columns(df, columns, title="Custom Plot", xlabel="Date", ylabel="Values", figsize=(12, 6)):
    """
    Plots specified columns from a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - columns (list): A list of column names to plot.
    - title (str, optional): The title of the plot. Default is "Custom Plot".
    - xlabel (str, optional): Label for the x-axis. Default is "Date".
    - ylabel (str, optional): Label for the y-axis. Default is "Values".
    - figsize (tuple, optional): Figure size for the plot. Default is (12, 6).

    Returns:
    - None: Displays the plot.
    """
    # Create the figure with the specified size
    plt.figure(figsize=figsize)

    # Plot each column in the list
    for column in columns:
        plt.plot(df.index, df[column], label=column, linewidth=2)

    # Customize the plot
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid()

    # Show the plot
    plt.show()

def plot_signals(df: pd.DataFrame, signals: pd.Series, title: str = "Strategy Signals"):
    """
    Generic visualization for trading signals.
    - 2 = Buy
    - 1 = Short Sell
    - -2 = Exit Long
    - -1 = Exit Short
    """
    close = df['Close']
    high = df['High']
    low = df['Low']

    plt.figure(figsize=(14, 7))

    # Price
    plt.plot(close.index, close.values, linewidth=1, label="Close Price")

    # Signals
    buys = signals == 2
    sells = signals == 1
    exit_longs = signals == -2
    exit_shorts = signals == -1

    plt.scatter(close.index[buys], close[buys], marker="^", s=100, label="Buy",  zorder=5)
    plt.scatter(close.index[sells], close[sells], marker="v", s=100, label="Short Sell", zorder=5)
    plt.scatter(close.index[exit_longs], close[exit_longs], marker="o", s=100, label="Exit Long", zorder=5)
    plt.scatter(close.index[exit_shorts], close[exit_shorts], marker="x", s=100, label="Exit Short", zorder=5)

    # High/Low lines per meglio comprendere le candele (non troppo invasivo)
    plt.vlines(close.index, low.values, high.values, color='lightgray', alpha=0.4, linewidth=0.6)

    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




def plot_numeric_flags_signals(
        df,
        num_cols,
        flag_cols=None,
        signals=None,
        price_col="Close",
        title="Strategy Plot"
):
    idx = df.index

    # Determine number of panels
    num_panels = 1
    if flag_cols:
        num_panels += len(flag_cols)

    fig, axes = plt.subplots(num_panels, 1, figsize=(18, 4 * num_panels), sharex=True)
    if num_panels == 1:
        axes = [axes]

    # === PANEL 1: Numeric indicators ===
    ax_main = axes[0]
    for col in num_cols:
        if col in df.columns:
            ax_main.plot(idx, df[col], linewidth=1.4, label=col)
        else:
            print(f"⚠ WARNING: column '{col}' not found in df")

    # ✅ Overlay signals
    if signals is not None:
        # assume price_col is in df, else use first num_col
        price = df[price_col] if price_col in df else df[num_cols[0]]

        # BUY
        ax_main.scatter(idx[signals == 2],
                        price[signals == 2],
                        marker="^", s=120, label="BUY")
        # SHORT
        ax_main.scatter(idx[signals == 1],
                        price[signals == 1],
                        marker="v", s=120, label="SHORT")
        # EXIT LONG
        ax_main.scatter(idx[signals == -2],
                        price[signals == -2],
                        marker="x", s=80, label="EXIT LONG")
        # EXIT SHORT
        ax_main.scatter(idx[signals == -1],
                        price[signals == -1],
                        marker="X", s=80, label="EXIT SHORT")

    ax_main.set_title(title)
    ax_main.grid(True)
    ax_main.legend(loc="upper left")

    # === FLAG PANELS ===
    if flag_cols:
        for i, col in enumerate(flag_cols, start=1):
            ax = axes[i]
            if col not in df.columns:
                print(f"⚠ WARNING: flag column '{col}' not found in df")
                continue

            # Convert boolean to int
            series = df[col].astype(int)

            ax.fill_between(idx, 0, series, where=series == 1, alpha=0.4)
            ax.set_ylim(-0.2, 1.2)
            ax.set_yticks([])
            ax.set_ylabel(col, rotation=0, labelpad=50, fontsize=10, va="center")
            ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_x(figure=None,open_browser=True):
    if figure is None:
        return
