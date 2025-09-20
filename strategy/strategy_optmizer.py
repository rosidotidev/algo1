
import numpy as np
import itertools
from stock.my_yfinance import MyYFinance
import stock.candle_signal_vec as scs
import stock.indicators_signal_vec as si
import strategy.xx_trades_bt as trades
from strategy.ticker_stategy_repo import TickerStrategyRepo
from stock.strategy_repo import StrategyRepo
import json
import pandas as pd
from backtrader_util import bu

def get_best_strategies(
    df: pd.DataFrame,
    top_n: int = 5,
    min_trades: int = 6,
    sort_by: str = "Win Rate [%]"
) -> pd.DataFrame:
    """
    Estrae le migliori strategie da un dataframe di risultati di backtest.

    Args:
        df (pd.DataFrame): dataframe con colonne ['ticker', 'strategy', 'params', 'Win Rate [%]', '# Trades', 'Return [%]']
        top_n (int): numero di strategie migliori da estrarre.
        min_trades (int): filtro minimo sul numero di trades.
        sort_by (str): metrica su cui ordinare (default = 'Win Rate [%]').

    Returns:
        pd.DataFrame: sottoinsieme con le top strategie.
    """

    # Assicura che params sia un dict
    df = df.copy()
    if df["params"].dtype == object:
        try:
            df["params"] = df["params"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
        except Exception:
            pass

    # Applica filtro sui trades
    df_filtered = df[df["# Trades"] >= min_trades]

    # Ordina per la metrica scelta
    df_sorted = df_filtered.sort_values(by=sort_by, ascending=False)

    # Restituisci solo le top_n
    return df_sorted.head(top_n)

def test_donchian_breakout_with_ma_filter(tick="TPR",strat=si.donchian_breakout_with_ma_filter):
    ticker = tick
    func = strat
    price = MyYFinance.fetch_by_period(ticker, period='3y')

    # lookback=20, ma_window=50
    lbw_window = np.arange(10, 20, 2)
    ma_window = np.arange(25, 55, 1)


    all_portfolios = {}
    results_list = []
    tsp = TickerStrategyRepo("../../data/")

    strategy = tsp.get_by_ticker_and_strategy(ticker, func.__name__)
    # Iterate through all combinations of fast and slow windows
    for lw, mw in itertools.product(lbw_window, ma_window):

        # strategy=tsp.get_by_ticker_and_strategy(ticker,scs.retracement_tf_vectorized.__name__)
        tsp.update_ticker_strategy(strategy["ticker"], strategy["strategy_func"],
                                   {"lookback": int(lw), "ma_window": int(mw)})
        sl_w = 0.05
        tp_w = 0.08
        res = trades.run_x_backtest_DaxPattern_vec(f"../../data/{ticker}.csv", slperc=sl_w, tpperc=tp_w,
                                                   capital_allocation=1, show_plot=False,
                                                   target_strategy=func, add_indicators=True)

        all_portfolios[(lw, mw)] = res
        win_rate = res["Win Rate [%]"]
        n_trades = res["# Trades"]
        ret = res["Return [%]"]
        max_drawdown = res["Max. Drawdown [%]"]
        # lookback_weeks=1, entry_day=1
        results_list.append({
            "ticker": ticker,
            "strategy": func.__name__,
            "Win Rate [%]": float(win_rate),
            "# Trades": float(n_trades),
            "Return [%]": float(ret),
            "Max. Drawdown [%]": float(max_drawdown),
            "params": {
                "lookback": int(lw),
                "ma_window": int(mw)
            },
        })

        print(f'[{lw},{mw}] win rate {win_rate}, n trades {n_trades}, ret {ret}')

    df_results = pd.DataFrame(results_list)
    # Converto la colonna "params" in JSON string così resta leggibile nel CSV
    df_results["params"] = df_results["params"].apply(json.dumps)

    df_results.to_csv(
        f"../../optimization/{ticker}_{func.__name__}_backtest_results.csv",
        index=False
    )

    best_return = -np.inf
    best_params = None
    for params, portfolio in all_portfolios.items():
        current_return = float(portfolio["Win Rate [%]"])
        if current_return > best_return and float(portfolio["# Trades"]) > 6:
            best_return = current_return
            best_params = params

    tsp.update_ticker_strategy(ticker, func.__name__,
                               {"lookback_weeks": int(best_params[0]), "entry_day": int(best_params[1])})

    best_portfolio = all_portfolios[best_params]
    print(f' best parameters {best_params}')
    print(best_portfolio)

    # Esempio di utilizzo
    # df_results = pd.read_csv("...")  # se già salvato
    top_n = get_best_strategies(df_results, top_n=15)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    print(top_n)




def test_weekly_breakout_vectorized():
    ticker = "UCG.MI"
    func = si.weekly_breakout_vectorized
    price = MyYFinance.fetch_by_period(ticker, period='3y')

    # lookback_weeks=1, entry_day=1
    lbw_window = np.arange(1, 5, 1)
    ed_window = np.arange(1, 4, 1)


    all_portfolios = {}
    results_list = []
    tsp = TickerStrategyRepo("../../data/")

    strategy = tsp.get_by_ticker_and_strategy(ticker, func.__name__)
    # Iterate through all combinations of fast and slow windows
    for lw, ew in itertools.product(lbw_window, ed_window):

        # strategy=tsp.get_by_ticker_and_strategy(ticker,scs.retracement_tf_vectorized.__name__)
        tsp.update_ticker_strategy(strategy["ticker"], strategy["strategy_func"],
                                   {"lookback_weeks": int(lw), "entry_day": int(ew)})
        sl_w = 0.05
        tp_w = 0.08
        res = trades.run_x_backtest_DaxPattern_vec(f"../../data/{ticker}.csv", slperc=sl_w, tpperc=tp_w,
                                                   capital_allocation=1, show_plot=False,
                                                   target_strategy=func, add_indicators=True)

        all_portfolios[(lw, ew)] = res
        win_rate = res["Win Rate [%]"]
        n_trades = res["# Trades"]
        ret = res["Return [%]"]
        max_drawdown = res["Max. Drawdown [%]"]
        # lookback_weeks=1, entry_day=1
        results_list.append({
            "ticker": ticker,
            "strategy": func.__name__,
            "Win Rate [%]": float(win_rate),
            "# Trades": float(n_trades),
            "Return [%]": float(ret),
            "Max. Drawdown [%]": float(max_drawdown),
            "params": {
                "lookback_weeks": int(lw),
                "entry_day": int(ew)
            },
        })

        print(f'[{lw},{ew}] win rate {win_rate}, n trades {n_trades}, ret {ret}')

    df_results = pd.DataFrame(results_list)
    # Converto la colonna "params" in JSON string così resta leggibile nel CSV
    df_results["params"] = df_results["params"].apply(json.dumps)

    df_results.to_csv(
        f"../../optimization/{ticker}_{func.__name__}_backtest_results.csv",
        index=False
    )

    best_return = -np.inf
    best_params = None
    for params, portfolio in all_portfolios.items():
        current_return = float(portfolio["Win Rate [%]"])
        if current_return > best_return and float(portfolio["# Trades"]) > 6:
            best_return = current_return
            best_params = params

    tsp.update_ticker_strategy(ticker, func.__name__,
                               {"lookback_weeks": int(best_params[0]), "entry_day": int(best_params[1])})

    best_portfolio = all_portfolios[best_params]
    print(f' best parameters {best_params}')
    print(best_portfolio)

    # Esempio di utilizzo
    # df_results = pd.read_csv("...")  # se già salvato
    top_n = get_best_strategies(df_results, top_n=15)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    print(top_n)


def test_retracement_rev_vectorized(strat=scs.retracement_rev_vectorized):

    ticker="SNOW"
    func=strat
    price = MyYFinance.fetch_by_period(ticker, period='3y')

    # price.dropna(inplace=True)
    # Rimuovi le righe con prezzo di chiusura <= 0
    # price = price[price['Close'] > 0]

    s_window = np.arange(2, 10, 1)
    m_window = np.arange(5, 20, 2)
    l_window = np.arange(15, 40, 5)

    all_portfolios = {}
    results_list = []
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
            max_drawdown=res["Max. Drawdown [%]"]

            results_list.append({
                "ticker": ticker,
                "strategy": func.__name__,
                "Win Rate [%]": float(win_rate),
                "# Trades": float(n_trades),
                "Return [%]": float(ret),
                "Max. Drawdown [%]": float(max_drawdown),
                "params": {
                    "short": int(s_w),
                    "medium": int(m_w),
                    "long": int(l_w)
                },
            })

            print(f'[{s_w},{m_w},{l_w}] win rate {win_rate}, n trades {n_trades}, ret {ret}')

    df_results = pd.DataFrame(results_list)
    # Converto la colonna "params" in JSON string così resta leggibile nel CSV
    df_results["params"] = df_results["params"].apply(json.dumps)

    df_results.to_csv(
        f"../../optimization/{ticker}_{func.__name__}_backtest_results.csv",
        index=False
    )

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

    # Esempio di utilizzo
    # df_results = pd.read_csv("...")  # se già salvato
    top_n = get_best_strategies(df_results, top_n=15)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    print(top_n)



# Encoder custom per serializzare numpy types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


def optimize_strategy(
    ticker,
    func,
    param_space,
    sl=0.05,
    tp=0.08,
    min_trades=6,
    score_func=None,
    param_filter=None,
    base_path="../"
):
    """
    Run grid search optimization for a given strategy.

    Args:
        ticker (str): Ticker symbol.
        func (callable): Strategy function to test.
        param_space (dict): Dictionary of parameter ranges.
        sl (float): Stop loss percentage.
        tp (float): Take profit percentage.
        min_trades (int): Minimum number of trades required to consider a portfolio.
        add_indicators (bool): Whether to add indicators during backtest.
        score_func (callable): Function that takes a result dict and returns a score.
                               Defaults to Win Rate [%].
        param_filter (callable): Function that takes a params dict and returns True/False.
                                 If False, the combination is skipped. Default = None.

    Returns:
        (DataFrame, dict): DataFrame of results, best portfolio dict.
    """
    print(f" ticker {ticker} func {func}")
    tsp = TickerStrategyRepo(f"{base_path}data")
    bu.cache["context"]["TickerStrategyRepo"] = tsp
    strategy = tsp.get_by_ticker_and_strategy(ticker, func.__name__)
    add_indicators:bool=StrategyRepo.get_add_indicators_flag(func)
    all_portfolios = {}
    results_list = []

    if score_func is None:
        score_func = lambda r: float(r["Win Rate [%]"])

    keys, values = zip(*param_space.items())

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))

        # skip combo se esiste un filtro e ritorna False
        if param_filter is not None and not param_filter(params):
            continue

        # --- IMPORTANT: update repo so strategy reads the current params ---
        # convert numpy ints -> Python int to avoid JSON errors in repo.save()
        params_for_repo = {
            k: int(v) if isinstance(v, (np.integer,)) else (float(v) if isinstance(v, (np.floating,)) else v)
            for k, v in params.items()
        }

        # update the repo (this preserves original behavior where each run writes params)
        tsp.update_ticker_strategy(strategy["ticker"], strategy["strategy_func"], params_for_repo)

        res = trades.run_x_backtest_DaxPattern_vec(
            f"{base_path}data/{ticker}.csv",
            slperc=sl,
            tpperc=tp,
            capital_allocation=1,
            show_plot=False,
            target_strategy=func,
            add_indicators=add_indicators
        )

        all_portfolios[combo] = res
        # convertiamo i valori numpy a Python nativi per il json
        results_list.append({
            "ticker": ticker,
            "strategy": func.__name__,
            "Win Rate [%]": float(res["Win Rate [%]"]),
            "# Trades": float(res["# Trades"]),
            "Return [%]": float(res["Return [%]"]),
            "Max. Drawdown [%]": float(res["Max. Drawdown [%]"]),
            "params": {k: int(v) if isinstance(v, np.integer) else v for k, v in params.items()}
        })

    # salva CSV usando NpEncoder per sicurezza
    df_results = pd.DataFrame(results_list)
    df_results["params"] = df_results["params"].apply(lambda p: json.dumps(p, cls=NpEncoder))
    #df_results.to_csv(
    #    f"{base_path}optimization/{ticker}_{func.__name__}_backtest_results.csv",
    #    index=False
    #)

    # selezione del migliore
    best_score, best_params = -np.inf, None
    for combo, portfolio in all_portfolios.items():
        if float(portfolio["# Trades"]) <= min_trades:
            continue
        score = score_func(portfolio)
        if score > best_score:
            best_score, best_params = score, combo

    if best_params is not None:
        #best_params_dict = dict(zip(keys, [int(v) if isinstance(v, np.integer) else v for v in best_params]))
        #tsp.update_ticker_strategy(ticker, func.__name__, best_params_dict)
        best_portfolio = all_portfolios[best_params]
    else:
        best_portfolio = None
    return df_results, best_portfolio

def test_generic():
    df_results, best = optimize_strategy(
        ticker="ASML",
        func=scs.inverted_filled_bar_strategy,
        param_space={
            "ratio": np.arange(0.8, 0.98, 0.01),
        },
        base_path="../../",
        #param_filter=lambda p: p["short"] < p["medium"] < p["long"]

    )
    # Esempio di utilizzo
    # df_results = pd.read_csv("...")  # se già salvato
    top_n = get_best_strategies(df_results, top_n=15)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    print(top_n)

if __name__ == "__main__":
    #test_retracement_rev_vectorized(strat=scs.retracement_rev_vectorized)
    #test_retracement_rev_vectorized(strat=scs.retracement_tf_vectorized)
    #test_weekly_breakout_vectorized()
    #test_donchian_breakout_with_ma_filter(strat=si.donchian_breakout_with_ma_filter)
    #test_donchian_breakout_with_ma_filter(tick="BHP",strat=si.donchian_inv_with_ma_filter)
    test_generic()

