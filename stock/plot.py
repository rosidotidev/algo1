import matplotlib.pyplot as plt

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
