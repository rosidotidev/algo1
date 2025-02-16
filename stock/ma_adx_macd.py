import backtrader as bt
from stock.base import BaseStrategy

# Strategy with Moving Average Crossover, ADX, and MACD
class MovingAverageAdxMacdStrategy(BaseStrategy):
    params = (
        ('sma_fast', 12),          # Fast SMA period
        ('sma_slow', 26),          # Slow SMA period
        ('adx_period', 14),        # ADX period
        ('adx_threshold', 25),     # ADX threshold for trend strength
        ('macd_fast', 12),         # MACD fast period
        ('macd_slow', 26),         # MACD slow period
        ('macd_signal', 9),        # MACD signal period
    )

    def __init__(self):
        # Fast and Slow SMA
        self.sma_fast = bt.indicators.SMA(self.data.close, period=self.params.sma_fast)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=self.params.sma_slow)

        # ADX
        self.adx = bt.indicators.ADX(self.data, period=self.params.adx_period)

        # MACD
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )
    def next(self):
        # Controlla se abbiamo una posizione aperta
        if not self.position:  # Nessuna posizione aperta
            # Condizioni per l'acquisto (Long)
            if (self.sma_fast > self.sma_slow and  # Golden Cross
                self.macd.macd > self.macd.signal and  # MACD bullish crossover
                self.adx > self.p.adx_threshold):  # ADX forte
                self.buy()  # Apri una posizione long

            # Condizioni per la vendita (Short)
            elif (self.sma_fast < self.sma_slow and  # Death Cross
                  self.macd.macd < self.macd.signal and  # MACD bearish crossover
                  self.adx > self.p.adx_threshold):  # ADX forte
                self.sell()  # Apri una posizione short

        else:  # Abbiamo una posizione aperta
            if self.position.size > 0:  # Posizione long aperta
                # Condizioni per chiudere la posizione long
                if self.macd.macd < self.macd.signal:  # MACD incrocia al ribasso
                    self.close()  # Chiudi la posizione long

            elif self.position.size < 0:  # Posizione short aperta
                # Condizioni per chiudere la posizione short
                if self.macd.macd > self.macd.signal:  # MACD incrocia al rialzo
                    self.close()  # Chiudi la posizione short
    def next1(self):
        # If no position is open
        if not self.position:
            size = self.calculate_size()  # Calculate size dynamically based on capital allocation

            if size > 0 and self.adx[0] > self.params.adx_threshold:  # Ensure ADX indicates a strong trend
                # Long Entry: Fast SMA > Slow SMA and MACD > Signal
                if self.sma_fast[0] > self.sma_slow[0] and self.macd.macd > self.macd.signal:
                    self.buy(size=size)  # Open a long position

                # Short Entry: Fast SMA < Slow SMA and MACD < Signal
                elif self.sma_fast[0] < self.sma_slow[0] and self.macd.macd < self.macd.signal:
                    self.sell(size=size)  # Open a short position

        else:
            # Long Exit: Fast SMA crosses below Slow SMA or MACD < Signal
            if self.position.size > 0:  # Long position
                if self.sma_fast[0] < self.sma_slow[0] or self.macd.macd < self.macd.signal:
                    self.close()  # Close the long position

            # Short Exit: Fast SMA crosses above Slow SMA or MACD > Signal
            elif self.position.size < 0:  # Short position
                if self.sma_fast[0] > self.sma_slow[0] or self.macd.macd > self.macd.signal:
                    self.close()  # Close the short position

def run_backtest_MovingAverageAdxMacdStrategy(data_path,sma_fast=10, sma_slow=50, adx_period=14, adx_threshold=25, macd_fast=12, macd_slow=26, macd_signal=9, capital_allocation=0.1):
    # Initialize Cerebro
    cerebro = bt.Cerebro()

    # Load historical data
    data = bt.feeds.GenericCSVData(
        dataname=data_path,  # Replace with your CSV file path
        dtformat='%Y-%m-%d %H:%M:%S%z',  # Date format in the CSV
        timeframe=bt.TimeFrame.Days,  # Timeframe for the data
        compression=1
    )
    cerebro.adddata(data)

    # Add the strategy with external parameters
    cerebro.addstrategy(
        MovingAverageAdxMacdStrategy,
        sma_fast=sma_fast,
        sma_slow=sma_slow,
        adx_period=adx_period,
        adx_threshold=adx_threshold,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        capital_allocation=capital_allocation
    )

    # Set initial capital
    cerebro.broker.setcash(10000.0)

    # Run the backtest
    print("Starting Portfolio Value: {:.2f}".format(cerebro.broker.getvalue()))
    cerebro.run()
    print("Final Portfolio Value: {:.2f}".format(cerebro.broker.getvalue()))

    # Plot the results
    #cerebro.plot()
    return cerebro.broker.getvalue()
def run_backtest_for_all_tickers(tickers_file, data_directory):
    """Runs backtests for all tickers in tickers.txt and determines the best performer."""

    # Read tickers from file
    with open(tickers_file, "r") as f:
        tickers = [line.strip() for line in f.readlines()]

    # Dictionary to store final portfolio values
    results = {}

    for ticker in tickers:
        print(ticker)
        data_path = f"{data_directory}/{ticker}.csv"  # Path to CSV file
        try:
            final_value =run_backtest_MovingAverageAdxMacdStrategy(data_path,sma_fast=10, sma_slow=50, adx_period=14, adx_threshold=25, macd_fast=12, macd_slow=26, macd_signal=9, capital_allocation=1)
            results[ticker] = final_value
        except Exception as e:
            print(f"Error running backtest for {ticker}: {e}")

    # Find the best-performing ticker
    if results:
        best_ticker = max(results, key=results.get)
        print("\nBest performing ticker:", best_ticker)
        print("Final Portfolio Value:", results[best_ticker])
        print(results)

# Execute the backtest
if __name__ == "__main__":
    # Example: pass custom parameters here
    run_backtest_for_all_tickers('../../data/tickers.txt', '../../data')