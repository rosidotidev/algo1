import pandas as pd
import stock.indicators_signal_vec as ins_vec
import stock.candle_signal_vec as cs_vec
from typing import List

def init_strategy_db(filename: str = "../data/strategy_db.csv"):
    """
    Initializes a DataFrame for a predefined set of trading strategies and
    saves it to a CSV file. The strategies are loaded internally from
    'ins_vec.indicators_strategy' and 'cs_vec.candlestick_strategies'.

    Args:
        filename (str): The name of the CSV file to save the database.
    """
    # Load the predefined strategy functions from the specified modules
    backtesting_functions = ins_vec.indicators_strategy + cs_vec.candlestick_strategies

    # Create a list of dictionaries for each strategy's data
    strategy_data = []
    for i, f in enumerate(backtesting_functions):
        strategy_data.append({
            'strategy_id': i,
            'strategy_name': f.__name__.replace("_", " "),
            'function_ref': f.__name__,
            'enabled': True
        })

    # Create a pandas DataFrame from the list of dictionaries
    df = pd.DataFrame(strategy_data)

    # Save the DataFrame to a CSV file without the index
    df.to_csv(filename, index=False)
    print(f"Strategy database initialized and saved to '{filename}'.")


def enable_strategy_by_name(strategy_name: str, filename: str = "../data/strategy_db.csv"):
    """
    Abilita una strategia nel database CSV impostando il suo flag 'enabled' su True.

    Args:
        strategy_name (str): Il nome della strategia da abilitare.
        filename (str): Il nome del file CSV del database delle strategie.
    """
    try:
        # Load the strategy database from the CSV file
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Errore: Il file '{filename}' non è stato trovato.")
        return

    # Check if the strategy name exists in the database
    if strategy_name not in df['strategy_name'].values:
        print(f"Errore: La strategia '{strategy_name}' non è stata trovata nel database.")
        return

    # Update the 'enabled' column for the specified strategy
    df.loc[df['strategy_name'] == strategy_name, 'enabled'] = True

    # Save the updated DataFrame back to the CSV file
    df.to_csv(filename, index=False)

    print(f"Strategia '{strategy_name}' abilitata con successo.")


def get_strategy_names(filename: str = "../data/strategy_db.csv", enabled: bool = True) -> List[str]:
    """
    Returns a list of strategy names from the CSV database that match the specified 'enabled' value.

    Args:
        filename (str): The name of the strategy database CSV file.
        enabled (bool): The 'enabled' value to filter by. Defaults to True.

    Returns:
        List[str]: A list of the filtered strategy names.
    """
    try:
        # Load the strategy database from the CSV file
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return []

    # Filter the DataFrame based on the 'enabled' column value
    filtered_df = df[df['enabled'] == enabled]

    # Extract the 'strategy_name' column as a list
    strategy_names = filtered_df['strategy_name'].tolist()

    return strategy_names

