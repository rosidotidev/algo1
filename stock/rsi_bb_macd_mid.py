import backtrader as bt
from stock.base import BaseStrategy
from backtrader_util import bu


# Strategy with RSI, Bollinger Bands, and MACD tailored for bullish and bearish trends
class RsiBollingerMacdStrategy(BaseStrategy):
    params = (
        ('rsi_period', 14),        # RSI period
        ('bollinger_period', 20),  # Bollinger Bands period
        ('bollinger_dev', 2),      # Bollinger Bands standard deviation
        ('macd_fast', 12),         # MACD fast period
        ('macd_slow', 26),         # MACD slow period
        ('macd_signal', 9),        # MACD signal period
    )

    def __init__(self):
        # RSI
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

        # Bollinger Bands
        self.bollinger = bt.indicators.BollingerBands(
            self.data.close, period=self.params.bollinger_period, devfactor=self.params.bollinger_dev
        )

        # MACD
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )

    def next(self):
        # If no position is open
        if not self.position:
            size = self.calculate_size()  # Calculate size dynamically based on capital allocation

            if size > 0:  # Ensure we can buy at least one unit
                # Long Entry: RSI < 30, price near or above Bollinger middle band, MACD > Signal
                if self.rsi < 50 and self.data.close >= self.bollinger.lines.bot and self.macd.macd > self.macd.signal:
                    self.buy(size=size)  # Open a long position

                # Short Entry: RSI > 50, price near or below Bollinger middle band, MACD < Signal
                elif self.rsi > 70 and self.data.close <= self.bollinger.lines.mid and self.macd.macd < self.macd.signal:
                    self.sell(size=size)  # Open a short position

        else:
            # Long Exit: RSI > 70 or price near Bollinger upper band
            if self.position.size > 0:  # Long position
                if self.rsi > 70 or self.data.close >= self.bollinger.lines.top:
                    self.close()  # Close the long position

            # Short Exit: RSI < 30 or price near Bollinger lower band
            elif self.position.size < 0:  # Short position
                if self.rsi < 50 or self.data.close <= self.bollinger.lines.bot:
                    self.close()  # Close the short position

def run_backtest_RsiBollingerMacdStrategy(data_path,rsi_period=14, bollinger_period=20, bollinger_dev=2, macd_fast=12, macd_slow=26, macd_signal=9, capital_allocation=0.1):
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
    bu.add_analyzers_to_cerebro(cerebro)
    # Add the strategy with external parameters
    cerebro.addstrategy(
        RsiBollingerMacdStrategy,
        rsi_period=rsi_period,
        bollinger_period=bollinger_period,
        bollinger_dev=bollinger_dev,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        capital_allocation=capital_allocation
    )

    # Set initial capital
    cerebro.broker.setcash(10000.0)

    # Run the backtest
    print("Starting Portfolio Value: {:.2f}".format(cerebro.broker.getvalue()))
    results=cerebro.run()
    bu.print_backtest_report(results)
    print("Final Portfolio Value: {:.2f}".format(cerebro.broker.getvalue()))
    #cerebro.plot()
    return cerebro.broker.getvalue()
    # Plot the results



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
            final_value = run_backtest_RsiBollingerMacdStrategy(data_path,rsi_period=14, bollinger_period=20, bollinger_dev=2, macd_fast=12, macd_slow=26, macd_signal=9, capital_allocation=1)
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
    #run_backtest_RsiBollingerMacdStrategy('../../data/DOGE-USD.csv',rsi_period=10, bollinger_period=15, bollinger_dev=2, macd_fast=10, macd_slow=20, macd_signal=8, capital_allocation=1)
    run_backtest_for_all_tickers('../../data/tickers.txt','../../data/')