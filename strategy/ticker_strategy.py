import inspect
import pandas as pd

class TickerStrategy:
    def __init__(self, ticker: str, strategy_func, params: dict):
        self.ticker = ticker
        self.strategy_func = strategy_func
        self.params = params or {}

    def run(self, df: pd.DataFrame) -> pd.Series:
        """Runs the strategy on the given dataframe using the stored parameters"""
        sig = inspect.signature(self.strategy_func)
        valid_params = {
            k: v for k, v in self.params.items()
            if k in sig.parameters
        }
        return self.strategy_func(df, **valid_params)

    def to_dict(self) -> dict:
        """Converts the object into a dictionary (useful for saving in Pandas/CSV)"""
        return {
            "ticker": self.ticker,
            "strategy_func": self.strategy_func.__name__,
            "params": self.params
        }

    @classmethod
    def from_dict(cls, d: dict, strategy_registry: dict):
        """Recreates a TickerStrategy object from a dictionary"""
        func = strategy_registry[d["strategy_func"]]
        return cls(d["ticker"], func, d["params"])

    def __repr__(self):
        return f"TickerStrategy(ticker='{self.ticker}', strategy_func='{self.strategy_func.__name__}', params={self.params})"
