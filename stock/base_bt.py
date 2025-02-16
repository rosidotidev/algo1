from backtesting import Backtest, Strategy
import pandas as pd
# Abstract Strategy class containing shared methods
class BaseStrategyBT(Strategy):
    capital_allocation=1  # Portion of capital to allocate per trade (10%)
    debug_enabled= True
    df=None
    last_postion=0
    ctx=None

    def track_enter_long(self):
        self.ctx["Last Action"]=2

    def track_enter_short(self):
        self.ctx["Last Action"]=1

    def track_close_long(self):
        self.ctx["Last Action"]=-2

    def track_close_short(self):
        self.ctx["Last Action"]=-1

    def track_no_action(self):
        self.ctx["Last Action"]=0

    def calculate_size(self):
        """Calculate the number of shares to buy based on the allocated portion of capital."""
        capital_to_invest = self.equity * self.capital_allocation
        price = self.data.Close[-1]  # Current price of the asset
        if price > 0:  # Avoid division by zero
            size = int(capital_to_invest // price)  # Calculate size, rounded down
            return size
        return 0

    def current_df(self):
        current_date = self.data.index[-1]
        current_df=None
        # Check if the date exists in the DataFrame
        if current_date in self.df.index:
            current_df=self.df.loc[current_date]
        return current_df
    def get_total_signal(self):
        current_date = self.data.index[-1]
        total_signal = 0
        # Check if the date exists in the DataFrame
        if current_date in self.df.index:
            # Get the corresponding TotalSignal value
            total_signal = self.df.loc[current_date, 'TotalSignal']
        return total_signal

    def notify_trade(self, trade):
        """
        Prints trade entry and exit details during the backtest.
        """
        if self.params.debug_enabled:
            if trade.justopened:
                print(f"\n📌 Trade Opened - {self.datetime.date(0)}")
                print(f"   Type: {'LONG' if trade.size > 0 else 'SHORT'}")
                print(f"   Entry Price: {trade.price:.2f}")
                print(f"   Position Size: {trade.size}")

            if trade.isclosed:
                exit_price = self.data.close[0]  # Corrected exit price
                print(f"\n✅ Trade Closed - {self.datetime.date(0)}")
                print(f"   Entry Price: {trade.price:.2f}")  # Still shows the original entry price
                print(f"   Exit Price: {exit_price:.2f}")  # The actual price at the moment of closing
                print(f"   PnL: {trade.pnl:.2f}")
                print(f"   Gross PnL: {trade.pnlcomm:.2f} (after commissions)")
