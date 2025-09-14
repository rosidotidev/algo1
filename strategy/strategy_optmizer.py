import time
import numpy as np
import itertools
import pandas as pd
from stock.my_yfinance import MyYFinance
import strategy.xx_trades_bt as trades


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
    start_time = time.perf_counter()
    price = MyYFinance.fetch_by_period('AMZN', period='3y')

    # price.dropna(inplace=True)
    # Rimuovi le righe con prezzo di chiusura <= 0
    # price = price[price['Close'] > 0]

    download_time = time.perf_counter()
    print(f"Tempo di download dei dati: {download_time - start_time:.2f} secondi")

    ratio_window = np.arange(0.7, 0.95, 0.05)
    sl_window = np.arange(0.05, 0.20, 0.03)
    tp_window = np.arange(0.08, 0.25, 0.03)

    all_portfolios = {}

    # Iterate through all combinations of fast and slow windows
    for ratio_w, sl_w, tp_w in itertools.product(ratio_window, sl_window, tp_window):
        my_signals = filled_bar_vectorized(price, ratio=ratio_w)

        # 2. Convert numerical signals to boolean Series for VectorBT
        entries = (my_signals == 1)
        exits = (my_signals == 2)

        trades.run_backtest_DaxPattern_vec_df(price, slperc=0.04, tpperc=0.02, capital_allocation=1, show_plot=False,
                                    target_strategy=filled_bar_vectorized, add_indicators=True)


        # Store portfolio in a dictionary with the window pair as the key
        all_portfolios[(ratio_w, sl_w, tp_w)] = portfolio
        # ❗ Aggiungi queste righe per stampare i risultati parziali
        current_return = portfolio.total_return()
        current_trades = portfolio.trades.count()
        print(f"Combinazione ({ratio_w},{sl_w},{tp_w}) - Rendimento: {current_return:.2%} - trades: {current_trades}")

    backtest_time = time.perf_counter()
    print(f"Tempo di backtesting: {backtest_time - download_time:.2f} secondi")

    # Manually find the best portfolio by sorting returns
    best_return = -np.inf
    best_params = None

    for params, portfolio in all_portfolios.items():
        current_return = portfolio.total_return()
        if current_return > best_return:
            best_return = current_return
            best_params = params

    best_portfolio = all_portfolios[best_params]

    print("Migliore combinazione di parametri: ", best_params)
    print("Miglior rendimento totale: ", best_return)

    print("\nRisultati del backtest per la migliore combinazione:")
    print(best_portfolio.stats())

    fig = best_portfolio.plot()
    fig.update_layout(title="Strategia MMA Ottimizzata su BTC-USD")
    fig.show()


if __name__ == "__main__":
    main3()

