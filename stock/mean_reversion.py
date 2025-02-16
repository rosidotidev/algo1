import backtrader as bt
from stock.base import BaseStrategy


# Mean-Reversion Strategy with Bollinger Bands, RSI, and CCI
class MeanReversionStrategy(BaseStrategy):
    params = (
        ('bollinger_period', 20),  # Bollinger Bands period
        ('bollinger_dev', 2),  # Bollinger Bands standard deviation
        ('rsi_period', 14),  # RSI period
        ('cci_period', 14),  # CCI period
    )

    def __init__(self):
        # Bollinger Bands
        self.bollinger = bt.indicators.BollingerBands(
            self.data.close, period=self.params.bollinger_period, devfactor=self.params.bollinger_dev
        )

        # RSI
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

        # CCI
        self.cci = bt.indicators.CCI(self.data, period=self.params.cci_period)

    def next(self):
        # If no position is open
        if not self.position:
            size = self.calculate_size()  # Calculate size dynamically based on capital allocation

            if size > 0:
                # Long Entry: Price at lower Bollinger Band and RSI < 30
                if self.data.close[0] <= self.bollinger.lines.bot and self.rsi[0] < 30:
                    self.buy(size=size)  # Open a long position

                # Short Entry: Price at upper Bollinger Band and RSI > 70
                elif self.data.close[0] >= self.bollinger.lines.top and self.rsi[0] > 70:
                    self.sell(size=size)  # Open a short position

        else:
            # Long Exit: RSI > 50 or price near Bollinger upper band
            if self.position.size > 0:
                if self.rsi[0] > 50 or self.data.close[0] >= self.bollinger.lines.top:
                    self.close()  # Close the long position

            # Short Exit: RSI < 50 or price near Bollinger lower band
            elif self.position.size < 0:
                if self.rsi[0] < 50 or self.data.close[0] <= self.bollinger.lines.bot:
                    self.close()  # Close the short position


def run_backtest_MeanReversionStrategy(data_path,bollinger_period=20, bollinger_dev=2, rsi_period=14, cci_period=14, capital_allocation=0.1):
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
        MeanReversionStrategy,
        bollinger_period=bollinger_period,
        bollinger_dev=bollinger_dev,
        rsi_period=rsi_period,
        cci_period=cci_period,
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
            final_value =run_backtest_MeanReversionStrategy(data_path,bollinger_period=20, bollinger_dev=2, rsi_period=14, cci_period=14, capital_allocation=1)
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
    run_backtest_for_all_tickers("../../data/tickers.txt","../../data")