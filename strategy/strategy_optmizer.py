import time
import numpy as np
import itertools
import pandas as pd
from stock.my_yfinance import MyYFinance
import stock.candle_signal_vec as scs
import strategy.xx_trades_bt as trades
from strategy.ticker_stategy_repo import TickerStrategyRepo


# Importa la tua funzione di segnale
def filled_bar_vectorized(df, ratio=0.8):
    df["ratio"] = (df['Close'] - df['Open']) / (df['High'] - df['Low'])
    df['prev_low'] = df['Low'].shift(1)
    bullish_signal = df["ratio"] > ratio
    bearish_signal = df['Close'] < df['prev_low']
    signals = pd.Series(0, index=df.index, dtype='int8')
    signals[bullish_signal] = 2  # Buy signal
    signals[bearish_signal] = 1  # Sell signal
    df.drop(columns=['ratio', 'prev_low'], inplace=True)
    return signals


def main3():

    ticker="MS"
    func=scs.retracement_rev_vectorized
    price = MyYFinance.fetch_by_period(ticker, period='3y')

    # price.dropna(inplace=True)
    # Rimuovi le righe con prezzo di chiusura <= 0
    # price = price[price['Close'] > 0]

    s_window = np.arange(2, 10, 1)
    m_window = np.arange(5, 20, 2)
    l_window = np.arange(15, 40, 5)

    all_portfolios = {}
    tsp=TickerStrategyRepo("../../data/")

    strategy = tsp.get_by_ticker_and_strategy(ticker, func.__name__)
    # Iterate through all combinations of fast and slow windows
    for s_w, m_w, l_w in itertools.product(s_window, m_window, l_window):
        if s_w < m_w < l_w:
            #strategy=tsp.get_by_ticker_and_strategy(ticker,scs.retracement_tf_vectorized.__name__)
            tsp.update_ticker_strategy(strategy["ticker"],strategy["strategy_func"],{"short":int(s_w),"medium":int(m_w),"long":int(l_w)})
            sl_w = 0.05
            tp_w = 0.08
            res=trades.run_x_backtest_DaxPattern_vec(f"../../data/{ticker}.csv", slperc=sl_w, tpperc=tp_w, capital_allocation=1, show_plot=False,
                                    target_strategy=func, add_indicators=False)

            all_portfolios[(s_w, m_w, l_w)] = res
            win_rate=res["Win Rate [%]"]
            n_trades=res["# Trades"]
            ret=res["Return [%]"]

            print(f'[{s_w},{m_w},{l_w}] win rate {win_rate}, n trades {n_trades}, ret {ret}')

    best_return = -np.inf
    best_params = None
    for params, portfolio in all_portfolios.items():
        current_return = float(portfolio["Win Rate [%]"])
        if current_return > best_return and float(portfolio["# Trades"]) > 6:
            best_return = current_return
            best_params = params

    tsp.update_ticker_strategy(ticker, func.__name__,
                               {"short": int(best_params[0]), "medium": int(best_params[1]), "long": int(best_params[2])})

    best_portfolio = all_portfolios[best_params]
    print(f' best parameters {best_params}')
    print(best_portfolio)

if __name__ == "__main__":
    main3()

