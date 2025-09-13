import pandas as pd
from typing import Optional, List
import stock.indicators_signal_vec as ins_vec
import stock.candle_signal_vec as cs_vec


class StrategyRepo:
    def __init__(self, filename: str = "../data/strategy_repo.csv"):
        self.filename = filename
        self._df: Optional[pd.DataFrame] = None
        self._load()

    def _load(self):
        """Load strategies from CSV file, or initialize if not found."""
        try:
            self._df = pd.read_csv(self.filename)
            if "enabled" in self._df.columns:
                self._df["enabled"] = self._df["enabled"].astype(bool)
        except FileNotFoundError:
            print(f"File '{self.filename}' not found. Initializing new repo...")
            self.init_repo()
    def enable_all(self) -> pd.DataFrame:
        """Enable all strategies and return updated DataFrame."""
        self._df["enabled"] = True
        self.save()
        print("All strategies have been enabled.")
        return self._df

    def disable_all(self) -> pd.DataFrame:
        """Disable all strategies and return updated DataFrame."""
        self._df["enabled"] = False
        self.save()
        print("All strategies have been disabled.")
        return self._df

    def save(self):
        """Save repository to CSV."""
        self._df.to_csv(self.filename, index=False)
    @staticmethod
    def get_all_available_strategy_functions():
            return ins_vec.indicators_strategy + cs_vec.candlestick_strategies

    def init_repo(self) -> pd.DataFrame:
        """Initialize the repository with all available strategies."""
        backtesting_functions = ins_vec.indicators_strategy + cs_vec.candlestick_strategies
        strategy_data = []

        for i, f in enumerate(backtesting_functions):
            # Determine enrich flag
            enrich_flag = f in ins_vec.indicators_strategy

            strategy_data.append({
                "id": i,
                "strategy_name": f.__name__.replace("_", " "),
                "function_ref": f.__name__,
                "enabled": True,
                "score": 10,  # optional field for performance tracking
                "enrich": enrich_flag
            })

        self._df = pd.DataFrame(strategy_data)
        self.save()
        print(f"Strategy repository initialized and saved to '{self.filename}'.")
        return self._df

    def enable(self, strategy_name: str) -> pd.DataFrame:
        """Enable a strategy by its name and return updated DataFrame."""
        if strategy_name not in self._df["strategy_name"].values:
            print(f"Error: strategy '{strategy_name}' not found.")
            return self._df
        self._df.loc[self._df["strategy_name"] == strategy_name, "enabled"] = True
        self.save()
        return self._df

    def disable(self, strategy_name: str) -> pd.DataFrame:
        """Disable a strategy by its name and return updated DataFrame."""
        if strategy_name not in self._df["strategy_name"].values:
            print(f"Error: strategy '{strategy_name}' not found.")
            return self._df
        self._df.loc[self._df["strategy_name"] == strategy_name, "enabled"] = False
        self.save()
        return self._df

    def get_strategies(self, enabled: bool = True) -> pd.DataFrame:
        """Return DataFrame of strategies filtered by enabled flag."""
        return self._df[self._df["enabled"] == enabled].copy()

    def get_all_strategies(self) -> pd.DataFrame:
        """Return DataFrame of all strategies."""
        return self._df.copy()

    def get_all_strategies_as_list(self) -> List[str]:
        """Return all strategy names as a list."""
        return self._df["strategy_name"].tolist()

    def set_score(self, strategy_name: str, score: float) -> pd.DataFrame:
        """Update the performance score of a strategy and return updated DataFrame."""
        if strategy_name not in self._df["strategy_name"].values:
            print(f"Error: strategy '{strategy_name}' not found.")
            return self._df
        self._df.loc[self._df["strategy_name"] == strategy_name, "score"] = score
        self.save()
        return self._df

    def top_strategies(self, n: int = 5) -> pd.DataFrame:
        """Return the top N strategies sorted by score."""
        return self._df.sort_values(by="score", ascending=False).head(n).copy()
