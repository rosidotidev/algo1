from stock.x_backtesting_bt import XBacktestingBT
from strategy.ticker_strategy import TickerStrategy
from typing import Optional
import stock.ticker as ti

class XStrategyBT(XBacktestingBT):

    ts: Optional[TickerStrategy] = None

    def init(self):
        self.built_signals=False
        super().init()

    def current_df(self):
        current_date = self.data.index[-1]
        current_df=None
        if 'TotalSignal' not in self.df.columns:
            self.df.loc[:, 'TotalSignal'] = 0
        if not self.built_signals:
            self.df.loc[:, 'TotalSignal'] = self.ts.run(self.df)
            self.built_signals=True

        # Check if the date exists in the DataFrame
        if current_date in self.df.index:
            current_df=self.df.loc[current_date]
        return current_df




