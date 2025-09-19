from stock.x_backtesting_bt import XBacktestingBT
from strategy.ticker_strategy import TickerStrategy
from typing import Optional
import stock.ticker as ti

class XStrategyBT(XBacktestingBT):

    ts: Optional[TickerStrategy] = None

    def init(self):
        self.built_signals=False
        super().init()
    def  _current_df(self,current_date):
        current_df = None
        if current_date in self.df.index:
            current_df=self.df.loc[current_date]
        return current_df

    def _get_current_date(self):
        return self.data.index[-1]

    def current_df_(self):

        if 'TotalSignal' not in self.df.columns:
            self.df.loc[:, 'TotalSignal'] = 0
        if not self.built_signals:
            self.df.loc[:, 'TotalSignal'] = self.ts.run(self.df)
            self.built_signals=True
        current_date = self._get_current_date()
        # Check if the date exists in the DataFrame
        current_df=self._current_df(current_date)
        return current_df

    def current_df(self):

        if 'TotalSignal' not in self.df.columns:
            self.df.loc[:, 'TotalSignal'] = 0
        if not self.built_signals:
            self.df.loc[:, 'TotalSignal'] = self.ts.run(self.df)
            self.built_signals=True
            self._cursor = 0

        current_df = self.df.iloc[self._cursor + 1]
        self._cursor += 1
        return current_df







