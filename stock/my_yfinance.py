import requests
from datetime import datetime, timedelta, timezone
import time
import pandas as pd

class MyYFinance:
    BASE_URL = "https://query2.finance.yahoo.com/v8/finance/chart/"

    @staticmethod
    def _to_unix_timestamp(date_str: str) -> int:
        dt = datetime.strptime(date_str, "%Y%m%d")
        return int(time.mktime(dt.timetuple()))

    @staticmethod
    def fetch_data(symbol: str, start_date: str, end_date: str, interval: str = "1d"):
        period1 = MyYFinance._to_unix_timestamp(start_date)
        period2 = MyYFinance._to_unix_timestamp(end_date)

        url = f"{MyYFinance.BASE_URL}{symbol}"
        params = {
            "period1": period1,
            "period2": period2,
            "interval": interval,
            "events": "history"
        }

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "DNT": "1"
        }

        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return MyYFinance._parse_response(response.json())

    @staticmethod
    def _parse_response(data):
        result = data['chart']['result'][0]
        timestamps = result.get('timestamp', [])
        quote = result['indicators']['quote'][0]
        adjclose = result['indicators']['adjclose'][0]['adjclose']

        records = []
        for i, ts in enumerate(timestamps):
            records.append({
                "Date": datetime.fromtimestamp(ts, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S%z"),
                "Open": quote['open'][i],
                "High": quote['high'][i],
                "Low": quote['low'][i],
                "Close": quote['close'][i],
                "Volume": quote['volume'][i],
                "Adj_close": adjclose[i]
            })
        df = pd.DataFrame(records)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df

    @staticmethod
    def fetch_by_period(symbol: str, period: str = "1y", interval: str = "1d"):
        today = datetime.today()
        if period.endswith("y"):
            years = int(period[:-1])
            start = today - timedelta(days=365 * years)
        elif period.endswith("mo"):
            months = int(period[:-2])
            start = today - timedelta(days=30 * months)
        elif period.endswith("d"):
            days = int(period[:-1])
            start = today - timedelta(days=days)
        else:
            raise ValueError("Unsupported period format. Use '1y', '6mo', '30d', etc.")

        start_str = start.strftime("%Y%m%d")
        end_str = today.strftime("%Y%m%d")

        return MyYFinance.fetch_data(symbol, start_str, end_str, interval)


if __name__ == "__main__":
    # https://github.com/ranaroussi/yfinance/issues/2422


    data = MyYFinance.fetch_data("AAPL", start_date="20201130", end_date="20201201")

    data= MyYFinance.fetch_by_period("MSFT",period="3y")

    print(data)