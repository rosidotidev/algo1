import stock.indicators_signal_vec as ins_vec
from strategy.ticker_strategy import TickerStrategy
import pandas as pd

strategy_registry = {
    "rsi_hammer_vectorized": ins_vec.rsi_hammer_vectorized,
    "bollinger_bands_adx_simple_vectorized": ins_vec.bollinger_bands_adx_simple_vectorized
}

def strategies_to_dataframe(strategies: list[TickerStrategy]) -> pd.DataFrame:
    return pd.DataFrame([s.to_dict() for s in strategies])

def dataframe_to_strategies(df: pd.DataFrame) -> list[TickerStrategy]:
    return [TickerStrategy.from_dict(row, strategy_registry) for row in df.to_dict(orient="records")]


if __name__ == "__main__":
    import stock.ticker as ti
    import data.data_enricher as de
    df=ti.read_from_csv("../../data/AAPL.csv")
    df=de.add_rsi_macd_bb(df)
    s1 = TickerStrategy("AAPL", strategy_registry["rsi_hammer_vectorized"], {"up_rsi_bound":65,"low_rsi_bound":25})
    s2 = TickerStrategy("AAPL", strategy_registry["rsi_hammer_vectorized"],{})
    result1 = s1.run(df)
    result2 = s2.run(df)
    print(result2)
    dataF=strategies_to_dataframe([s1,s2])
    print(dataF)
    strategies=dataframe_to_strategies(dataF)
    res=strategies[0].run(df)
    print(res)
