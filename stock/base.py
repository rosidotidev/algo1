import backtrader as bt
import pandas as pd
# Abstract Strategy class containing shared methods
class BaseStrategy(bt.Strategy):
    params = (
        ('capital_allocation', 0.1),  # Portion of capital to allocate per trade (10%)
        ('debug_enabled', True),
        ('df', None),  # ADX period
    )

    def __init__(self):
        self.last_position = 0

    def track_enter_long(self):
        self.last_position=2
    def track_enter_short(self):
        self.last_position=1
    def track_close_long(self):
        self.last_position=-2
    def track_close_short(self):
        self.last_position=-1
    def track_no_action(self):
        self.last_position=0

    def calculate_size(self):
        """Calculate the number of shares to buy based on the allocated portion of capital."""
        capital_to_invest = self.broker.get_cash() * self.params.capital_allocation
        price = self.data.close[0]  # Current price of the asset
        if price > 0:  # Avoid division by zero
            size = int(capital_to_invest // price)  # Calculate size, rounded down
            return size
        return 0

    def current_df(self):
        current_date = self.datetime.datetime(0)
        current_df=None
        # Check if the date exists in the DataFrame
        if current_date in self.params.df.index:
            current_df=self.params.df.loc[current_date]
        return current_df
    def get_total_signal(self):
        current_date = self.datetime.datetime(0)
        total_signal = 0
        # Check if the date exists in the DataFrame
        if current_date in self.params.df.index:
            # Get the corresponding TotalSignal value
            total_signal = self.params.df.loc[current_date, 'TotalSignal']
        return total_signal

    def stop(self):
        cash = self.broker.get_cash()  # Capitale non investito
        value = self.broker.get_value()  # Valore totale del portafoglio
        invested = value - cash  # Capitale investito

        print(f"Capitale non investito (cash): {cash:.2f}")
        print(f"Capitale investito: {invested:.2f}")
        print(f"Valore totale del portafoglio: {value:.2f}")

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
