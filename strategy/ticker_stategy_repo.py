import stock.indicators_signal_vec as ins_vec
from strategy.ticker_strategy import TickerStrategy

strategy_registry = {
    "rsi_hammer_vectorized": ins_vec.rsi_hammer_vectorized,
    "bollinger_bands_adx_simple_vectorized": ins_vec.bollinger_bands_adx_simple_vectorized
}

if __name__ == "__main__":
    import stock.ticker as ti
    import data.data_enricher as de
    df=ti.read_from_csv("../../data/AAPL.csv")
    df=de.add_rsi_macd_bb(df)
    s1 = TickerStrategy("AAPL", strategy_registry["rsi_hammer_vectorized"], {"up_rsi_bound":65,"low_rsi_bound":25})
    result=s1.run(df)
    print(result)
