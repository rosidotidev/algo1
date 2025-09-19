from stock.base_bt import BaseStrategyBT

class XBacktestingBT(BaseStrategyBT):
    slperc= 0.05
    tpperc= 1.00
    def init(self):
        self.long_stop_price = None  # Stop Loss
        self.short_stop_price = None
        self.long_tp_price= None
        self.short_tp_price= None


    def stop_loss_long_check(self):
        stop_l = self.data.Close[-1] < self.long_stop_price
        return stop_l

    def take_profit_long_check(self):
        take_p = self.data.Close[-1] > self.long_tp_price
        return take_p

    def take_profit_short_check(self):
        take_p = self.data.Close[-1] < self.short_tp_price
        return take_p

    def stop_loss_short_check(self):
        stop_l = self.data.Close[-1] > self.short_stop_price
        return stop_l

    def next(self):
        # Get the current date in the format of the DataFrame index
        #total_signal=self.get_total_signal()
        total_signal=self.current_df()["TotalSignal"]

        #size = self.calculate_size()  # Calculate size dynamically based on capital allocation
        self.track_no_action()
        if not (self.position.is_short or self.position.is_long):

            if total_signal == 2:
                self.buy()
                self.track_enter_long()
                self.long_stop_price=self.data.Close[-1] * (1 - self.slperc)
                self.long_tp_price = self.data.Close[-1] * (1 + self.tpperc)
            elif total_signal == 1:
                self.sell()
                self.track_enter_short()
                self.short_stop_price = self.data.Close[-1] * (1 + self.slperc)
                self.short_tp_price = self.data.Close[-1] * (1 - self.tpperc)
        else:
            if self.position.is_long and ((total_signal==1 or total_signal==-2)
                    or self.stop_loss_long_check() or self.take_profit_long_check()):
                self.position.close()
                self.track_close_long()
            elif self.position.is_short and ((total_signal==2 or total_signal==-1)
                    or self.stop_loss_short_check() or self.take_profit_short_check()):
                self.position.close()
                self.track_close_short()