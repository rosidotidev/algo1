"""
portfolio_manager.py

Simple CSV-based portfolio tracker for real (not backtested) trades.
Stores everything in data/portfolio.csv  —  no database.

Schema:
  type,id,ticker,action,amount,exposure,leverage,entry_date,exit_date,fee,note,status
  cash,0,,,,,,,,0.0,initial_cash,
  trade,1,AAPL,Buy,100.0,200.0,2,2026-06-02,,0.0,,Open

actions:
  - Buy  = Enter Long (open)
  - Sell = Enter Short (open)

Close is done via close_trades() which creates the opposite action
and matches FIFO against open positions.
"""

import os
import pandas as pd
from datetime import date

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORTFOLIO_FILE = os.path.join(BASE_DIR, "data", "portfolio.csv")

COLS = [
    "type", "id", "ticker", "action",
    "amount", "exposure", "leverage",
    "entry_date", "exit_date", "fee", "pl_amount", "pl_pct", "note", "status",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _next_id(df: pd.DataFrame) -> int:
    if df.empty or "id" not in df.columns:
        return 1
    existing = df["id"].dropna()
    return int(existing.max() + 1) if not existing.empty else 1


def _load() -> pd.DataFrame:
    if not os.path.exists(PORTFOLIO_FILE):
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(PORTFOLIO_FILE, dtype={"id": int})
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    return df


def _save(df: pd.DataFrame):
    df.to_csv(PORTFOLIO_FILE, index=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def init_portfolio(initial_cash: float = 0.0):
    """Create portfolio.csv with initial cash entry if it doesn't exist."""
    if os.path.exists(PORTFOLIO_FILE):
        return
    cash_row = {c: None for c in COLS}
    cash_row.update({
        "type": "cash", "id": 0, "amount": initial_cash,
        "fee": 0.0, "note": "initial_cash",
    })
    df = pd.concat([pd.DataFrame(columns=COLS), pd.DataFrame([cash_row])], ignore_index=True)
    _save(df)


def _get_cash_amount(df: pd.DataFrame) -> float:
    cash_rows = df[df["type"] == "cash"]
    if cash_rows.empty:
        return 10_000.0
    val = cash_rows.iloc[0].get("amount", None)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 10_000.0
    return float(val)


def get_initial_cash() -> float:
    return _get_cash_amount(_load())


def set_initial_cash(amount: float):
    df = _load()
    mask = df["type"] == "cash"
    if not mask.any():
        init_portfolio(amount)
        return
    df.loc[mask, "amount"] = amount
    _save(df)


def add_trade(ticker: str, action: str, amount: float,
              leverage: int = 1, entry_date=None,
              fee: float = 0.0, note: str = ""):
    """
    Open a position (Buy = Long, Sell = Short).
    NEVER matches anything — this is pure entry.
    """
    df = _load()
    if entry_date is None:
        entry_date = date.today().isoformat()

    action_lower = action.strip().lower()
    ticker_upper = ticker.strip().upper()
    exposure = round(amount * leverage, 2)

    row = {c: None for c in COLS}
    row.update({
        "type": "trade",
        "id": _next_id(df),
        "ticker": ticker_upper,
        "action": "Buy" if action_lower == "buy" else "Sell",
        "amount": amount,
        "exposure": exposure,
        "leverage": leverage,
        "entry_date": entry_date,
        "fee": fee,
        "note": note,
        "status": "Open",
    })
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save(df)
    return row["id"]


def add_cash_operation(amount: float, entry_date=None, note: str = ""):
    """
    Record a manual cash operation (deposit/withdrawal).

    Parameters
    ----------
    amount : float       —  positive = deposit (cash in), negative = withdrawal (cash out)
    entry_date : str     —  "YYYY-MM-DD" or None (defaults to today)
    note : str           —  description of the operation
    """
    df = _load()
    if entry_date is None:
        entry_date = date.today().isoformat()

    row = {c: None for c in COLS}
    row.update({
        "type": "cash_op",
        "id": _next_id(df),
        "ticker": None,
        "action": "Deposit" if amount >= 0 else "Withdrawal",
        "amount": amount,
        "exposure": 0.0,
        "leverage": 1,
        "entry_date": entry_date,
        "fee": 0.0,
        "note": note or f"{'Deposit' if amount >= 0 else 'Withdrawal'} of €{abs(amount):.2f}",
        "status": "Closed",
    })
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save(df)
    return row["id"]


def close_trades(ticker: str, direction: str, amount_received: float,
                 fee: float = 0.0, note: str = ""):
    """
    Close open positions of a given direction for a ticker (FIFO).

    Parameters
    ----------
    ticker : str         —  e.g. "AAPL"
    direction : str      —  "LONG" or "SHORT"
    amount_received : float
        For LONG: total € you received from closing (cash in).
        For SHORT: total € you spent to buy back (cash out).
    fee : float          —  broker commission
    note : str           —  optional comment

    Behaviour
    ---------
    1. Finds open positions of the matching direction for the ticker (FIFO).
    2. Matches amount_received against their invested amounts.
    3. Creates a closing trade with the opposite action and same leverage.
    4. Closes / reduces each matched position.
    """
    df = _load()
    entry_date = date.today().isoformat()
    ticker_upper = ticker.strip().upper()
    direction_lower = direction.strip().lower()

    # LONG → open action is "Buy", closing action is "Sell"
    # SHORT → open action is "Sell", closing action is "Buy"
    if direction_lower == "long":
        open_action = "Buy"
        close_action = "Sell"
    elif direction_lower == "short":
        open_action = "Sell"
        close_action = "Buy"
    else:
        return False

    open_positions = df[
        (df["ticker"] == ticker_upper)
        & (df["action"] == open_action)
        & (df["exit_date"].isna())
    ].sort_values("id")

    if open_positions.empty:
        return False

    # Close the oldest open position (FIFO)
    idx = open_positions.index[0]

    pos_id = int(df.at[idx, "id"])
    pos_amount = float(df.at[idx, "amount"])
    pos_lev = int(df.at[idx, "leverage"])
    pos_exposure = float(df.at[idx, "exposure"])

    # amount_received is the ACTUAL cash flow — can be > or < pos_amount
    close_exposure = round(amount_received * pos_lev, 2)

    # Get open position's fee for P&L calculation
    pos_fee = float(df.at[idx, "fee"])

    # P&L calculation net of fees
    # For LONG: Sell(closing) - Buy(open) - fees
    # For SHORT: Sell(open) - Buy(closing) - fees
    # Both = (sell_amount - buy_amount) - (open_fee + close_fee)
    if close_action == "Sell":
        # Closing a Long: Sell is the close
        pnl_base = amount_received - pos_amount
    else:
        # Closing a Short: Buy is the close, open was Sell
        pnl_base = pos_amount - amount_received

    total_fees = pos_fee + fee
    pl_amount = round(pnl_base - total_fees, 2)
    pl_pct = round((pnl_base - total_fees) / pos_amount, 4) if pos_amount != 0 else 0.0

    # Create the closing trade with P&L
    close_row = {c: None for c in COLS}
    close_row.update({
        "type": "trade",
        "id": _next_id(df),
        "ticker": ticker_upper,
        "action": close_action,
        "amount": amount_received,
        "exposure": close_exposure,
        "leverage": pos_lev,
        "entry_date": entry_date,
        "exit_date": entry_date,
        "fee": fee,
        "pl_amount": pl_amount,
        "pl_pct": pl_pct,
        "note": note or f"close of #{pos_id}",
        "status": "Closed",
    })
    df = pd.concat([df, pd.DataFrame([close_row])], ignore_index=True)

    # Mark the opening trade with an exit_date but keep it "Open" for record-keeping
    df.at[idx, "exit_date"] = entry_date

    _save(df)
    return True


def delete_trade(trade_id: int) -> bool:
    """Remove a trade by id (for corrections)."""
    df = _load()
    before = len(df)
    df = df[~((df["type"] == "trade") & (df["id"] == trade_id))]
    if len(df) < before:
        _save(df)
        return True
    return False


def get_holdings() -> pd.DataFrame:
    """
    Current open positions (both Long and Short).

    Returns
    -------
    DataFrame with columns:
        Ticker, Direction, Invested (€), Leverage, Exposure (€)
    """
    df = _load()
    open_pos = df[
        (df["type"] == "trade")
        & (df["exit_date"].isna())
    ].copy()

    if open_pos.empty:
        return pd.DataFrame()

    def _direction(action):
        return "LONG" if action == "Buy" else "SHORT"

    open_pos["Direction"] = open_pos["action"].apply(_direction)

    grouped = open_pos.groupby(["ticker", "Direction"]).agg(
        Invested=("amount", "sum"),
        Exposure=("exposure", "sum"),
    ).reset_index()

    # Leverage from first row of each group
    lev_map = open_pos.groupby(["ticker", "Direction"]).first()["leverage"].to_dict()
    grouped["Leverage"] = grouped.apply(
        lambda r: int(lev_map.get((r["ticker"], r["Direction"]), 1)),
        axis=1,
    )

    grouped = grouped.round(2)
    grouped.columns = ["Ticker", "Direction", "Invested (€)", "Exposure (€)", "Leverage"]
    return grouped


def get_trade_history() -> pd.DataFrame:
    """
    All recorded entries (trades + cash ops), chronologically.
    P&L columns are read from the persisted CSV (computed at close time).
    """
    df = _load()
    entries = df[df["type"].isin(["trade", "cash_op"])].copy()
    if entries.empty:
        return pd.DataFrame()

    entries = entries.sort_values(["entry_date", "id"])
    cols_out = [
        "id", "ticker", "action", "amount", "leverage",
        "exposure", "entry_date", "exit_date", "fee",
        "pl_amount", "pl_pct", "status", "note",
    ]
    out = entries[cols_out].reset_index(drop=True)
    # Rename for display
    out.columns = [
        "id", "ticker", "action", "amount", "leverage",
        "exposure", "entry_date", "exit_date", "fee",
        "P&L (€)", "P&L (%)", "status", "note",
    ]
    # Clean NaN from P&L columns for display
    out["P&L (€)"] = out["P&L (€)"].apply(lambda x: round(x, 2) if pd.notna(x) else None)
    out["P&L (%)"] = out["P&L (%)"].apply(lambda x: round(x, 4) if pd.notna(x) else None)
    # Round other numeric columns
    for c in ["amount", "exposure", "fee"]:
        out[c] = out[c].apply(lambda x: round(x, 2) if pd.notna(x) else x)
    return out


def get_cash_balance() -> float:
    """
    Current cash balance = initial_cash + manual_ops - outflow(buys) + inflow(sells).
    Buy = money out (open long OR close short).
    Sell = money in (open short OR close long).
    """
    df = _load()
    cash = _get_cash_amount(df)

    # Manual cash operations (deposits / withdrawals)
    cash_ops = df[df["type"] == "cash_op"]
    if not cash_ops.empty:
        cash += cash_ops["amount"].sum()

    trades = df[df["type"] == "trade"].copy()
    if trades.empty:
        return round(cash, 2)

    buys = trades[trades["action"].fillna("").str.lower() == "buy"]
    sells = trades[trades["action"].fillna("").str.lower() == "sell"]

    outflow = (buys["amount"] + buys["fee"]).sum()
    inflow = (sells["amount"] - sells["fee"]).sum()

    return round(cash - outflow + inflow, 2)


def get_realized_pnl() -> float:
    """
    Total realized P&L across all closed trades (net of fees).
    Uses the persisted pl_amount column.
    """
    df = _load()
    trades = df[df["type"] == "trade"]
    if trades.empty:
        return 0.0
    pl_sum = trades["pl_amount"].dropna().sum()
    return round(float(pl_sum), 2)


def get_portfolio_summary() -> dict:
    """
    Full portfolio snapshot.
    """
    cash = get_cash_balance()
    holdings = get_holdings()

    exposure = holdings["Exposure (€)"].sum() if not holdings.empty else 0.0
    realized = get_realized_pnl()

    return {
        "cash_balance": cash,
        "positions_exposure": round(exposure, 2),
        "portfolio_total": round(cash + exposure, 2),
        "realized_pnl": realized,
    }


def get_advanced_stats(exclude_deposits: bool = False) -> dict:
    """
    Advanced portfolio statistics.

    Parameters
    ----------
    exclude_deposits : bool
        If True, excludes deposits/withdrawals from ROI base calculation.

    Returns
    -------
    dict with keys:
        pnl_month, pnl_year, pnl_12m, win_rate, avg_trade,
        total_deposits, roi, best_trade, worst_trade
    """
    df = _load()
    trades = df[df["type"] == "trade"].copy()
    cash_ops = df[df["type"] == "cash_op"].copy()

    total_deposits = cash_ops["amount"].sum() if not cash_ops.empty else 0.0

    closed = trades[trades["pl_amount"].notna()]
    total_pnl = closed["pl_amount"].sum() if not closed.empty else 0.0

    now = pd.Timestamp.now()
    current_year = now.year
    current_month = now.month
    twelve_months_ago = now - pd.DateOffset(months=12)

    def _filter_by_date(months_offset=0, year=None, month=None):
        if closed.empty:
            return 0.0
        dates = pd.to_datetime(closed["entry_date"], errors="coerce")
        if year is not None and month is not None:
            mask = (dates.dt.year == year) & (dates.dt.month == month)
        elif year is not None:
            mask = dates.dt.year == year
        else:
            mask = dates >= twelve_months_ago
        return closed.loc[mask, "pl_amount"].sum()

    pnl_month = _filter_by_date(year=current_year, month=current_month)
    pnl_year = _filter_by_date(year=current_year)
    pnl_12m = _filter_by_date()

    n_closed = len(closed)
    n_wins = (closed["pl_amount"] > 0).sum() if n_closed > 0 else 0
    win_rate = round(n_wins / n_closed * 100, 2) if n_closed > 0 else 0.0
    avg_trade = round(total_pnl / n_closed, 2) if n_closed > 0 else 0.0

    initial_cash = _get_cash_amount(df)
    if exclude_deposits:
        roi_base = initial_cash
    else:
        roi_base = initial_cash + total_deposits
    roi = round(total_pnl / max(1, abs(roi_base)) * 100, 2) if roi_base != 0 else 0.0

    best_trade = round(float(closed["pl_amount"].max()), 2) if n_closed > 0 else 0.0
    worst_trade = round(float(closed["pl_amount"].min()), 2) if n_closed > 0 else 0.0

    return {
        "pnl_month": round(float(pnl_month), 2),
        "pnl_year": round(float(pnl_year), 2),
        "pnl_12m": round(float(pnl_12m), 2),
        "win_rate": win_rate,
        "avg_trade": avg_trade,
        "total_deposits": round(float(total_deposits), 2),
        "roi": roi,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
    }