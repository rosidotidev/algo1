import pandas as pd
from typing import Optional, List
import json
from stock.strategy_repo import StrategyRepo
import stock.ticker as ti

class TickerStrategyRepo:
    def __init__(self, base_path: str = "../data"):
        self.base_path = base_path
        self.filename = f'{base_path}/ticker_strategies.csv'
        self._df: Optional[pd.DataFrame] = None
        self._load()

    def _load(self):
        """Loads strategies from the CSV file, or initializes a new repo if not found."""
        try:
            self._df = pd.read_csv(self.filename)
            # Converts the string representation of the dictionary into an actual dictionary
            #self._df['params'] = self._df['params'].apply(eval)
            self._df['params'] = self._df['params'].apply(json.loads)
        except FileNotFoundError:
            print(f"File '{self.filename}' not found. Initializing a new repository...")
            self._df = pd.DataFrame(columns=['ticker', 'strategy_func', 'params'])

    def save(self):
        """Saves the repository to CSV."""
        #self._df.to_csv(self.filename, index=False)
        df_copy = self._df.copy()
        df_copy['params'] = df_copy['params'].apply(json.dumps)
        df_copy.to_csv(self.filename, index=False)
        #print(f"Ticker strategy repository saved to '{self.filename}'.")

    def add_strategy(self, ticker: str, strategy_name: str, params: dict):
        """Adds a new strategy to the repository."""
        new_row = {'ticker': ticker, 'strategy_func': strategy_name, 'params': params}
        self._df = pd.concat([self._df, pd.DataFrame([new_row])], ignore_index=True)
        self.save()

    def get_by_ticker_and_strategy(self, ticker: str, strategy_name: str) -> Optional[dict]:
        """
        Returns a single dictionary for a specific ticker and strategy name.
        Returns None if no match is found.
        """
        if self._df is None or self._df.empty:
            return None

        # Filters the DataFrame based on the ticker and strategy function name
        filtered_row = self._df[
            (self._df['ticker'].str.lower() == ticker.lower()) &
            (self._df['strategy_func'].str.lower() == strategy_name.lower())
        ]

        if filtered_row.empty:
            return None

        # We expect at most one match, so we take the first row
        return filtered_row.iloc[0].to_dict()

    def init_repo(self):
        """
        Initializes the repository with mock tickers and strategies.
        Creates all combinations and persists them to file.
        """
        # Mocked tickers
        tickers = ti.read_tickers_from_file(f'{self.base_path}tickers.txt')

        # Mocked strategies with parameters
        strategies = StrategyRepo.get_all_available_strategy_functions()

        # Build all combinations
        rows = []
        for ticker in tickers:
            for strategy in strategies:
                rows.append({
                    "ticker": ticker,
                    "strategy_func": strategy.__name__,
                    "params": {}
                })

        # Replace internal dataframe
        self._df = pd.DataFrame(rows)

        # Persist to file
        self.save()

    def update_ticker_strategy(self, ticker: str, strategy_name: str, params: dict) -> bool:
        """
        Updates the params of an existing strategy for a given ticker and strategy.
        Returns True if updated, False if not found.
        """
        if self._df is None or self._df.empty:
            return False

        # Individua la riga corrispondente
        mask = (
            (self._df['ticker'].str.lower() == ticker.lower()) &
            (self._df['strategy_func'].str.lower() == strategy_name.lower())
        )

        if not mask.any():
            # Nessuna riga trovata
            return False

        # Aggiorna i parametri in memoria
        self._df.loc[mask, 'params'] = self._df.loc[mask, 'params'].apply(lambda _: params)

        # Salva su file
        self.save()
        return True

    def merge_ticker_strategies(self,csv_source="ticker_strategies_old.csv", csv_target="ticker_strategies.csv", csv_output=None):
        """
        Merge ticker strategies from csv_source into csv_target.
        Overwrites params in target if source has non-empty params.

        Args:
            csv_source (str): Path to CSV with strategies to copy from.
            csv_target (str): Path to CSV with strategies to update.
            csv_output (str or None): Path to save merged result (if None, overwrite target).
        """
        # Carica i due CSV
        df_src = pd.read_csv(f"{self.base_path}{csv_source}")
        df_tgt = pd.read_csv(f"{self.base_path}{csv_target}")

        # Normalizza params -> dict
        def parse_params(x):
            if isinstance(x, str) and x.strip():
                try:
                    return json.loads(x)
                except Exception:
                    return {}
            return {}

        df_src["params"] = df_src["params"].apply(parse_params)
        df_tgt["params"] = df_tgt["params"].apply(parse_params)

        # Creiamo dizionario di lookup dalle strategie sorgente
        src_dict = {
            (row["ticker"], row["strategy_func"]): row["params"]
            for _, row in df_src.iterrows()
            if row["params"] != {}
        }

        # Aggiorna i valori nel target
        updated = 0
        for idx, row in df_tgt.iterrows():
            key = (row["ticker"], row["strategy_func"])
            if key in src_dict and src_dict[key] != {}:
                df_tgt.at[idx, "params"] = src_dict[key]
                updated += 1

        print(f"Updated {updated} strategies with non-empty params from source.")

        # Salva il risultato
        if csv_output is None:
            csv_output = f"{self.base_path}{csv_target}"

        # Riconverti params a JSON string prima di salvare
        df_tgt["params"] = df_tgt["params"].apply(json.dumps)
        df_tgt.to_csv(csv_output, index=False)

        return df_tgt

    def backup(self, backup_filename="ticker_strategies_old.csv"):
        """
        Save current repo as backup.
        By default creates 'ticker_strategies_old.csv' in the same folder.
        """
        df_to_save = self._df.copy()
        df_to_save["params"] = df_to_save["params"].apply(json.dumps)
        backup_path = f'{self.base_path}{backup_filename}'
        df_to_save.to_csv(backup_path, index=False)
        print(f"Backup created: {backup_path}")
        return backup_path

if __name__ == "__main__":
    repo = TickerStrategyRepo('../../data/')
    repo.init_repo()
    #repo.merge_ticker_strategies('ticker_strategies_old.csv','ticker_strategies.csv')
    if False:
        # Test retrieval
        test_result = repo.get_by_ticker_and_strategy("AAPL", "mean_reversion")
        print("\nTest: get_by_ticker_and_strategy('AAPL', 'mean_reversion')")
        print(test_result)

        # Test add_strategy
        repo.add_strategy("GOOG", "custom_strategy", {"param1": 123, "param2": "abc"})
        print("\nAdded custom strategy for GOOG")
        print(repo.get_by_ticker_and_strategy("GOOG", "custom_strategy"))
        repo.update_ticker_strategy("GOOG", "custom_strategy", {"param1": 124, "param2": "def"})
        print(repo.get_by_ticker_and_strategy("GOOG", "custom_strategy"))