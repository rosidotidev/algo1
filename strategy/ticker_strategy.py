import inspect
import pandas as pd
from stock.strategy_repo import StrategyRepo

class TickerStrategy:
    def __init__(self, ticker: str, strategy_func, params: dict):
        self.ticker = ticker
        self.strategy_func = strategy_func
        self.params = params or {}

    def run(self, df: pd.DataFrame) -> pd.Series:
        """Runs the strategy on the given dataframe using the stored parameters"""
        func=StrategyRepo.get_strategy_function_by_name(self.strategy_func)
        sig = inspect.signature(func)
        valid_params = {
            k: v for k, v in self.params.items()
            if k in sig.parameters
        }
        return func(df, **valid_params)

    def __repr__(self):
        return f"TickerStrategy(ticker='{self.ticker}', strategy_func='{self.strategy_func}', params={self.params})"
