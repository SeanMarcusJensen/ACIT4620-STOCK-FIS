import yfinance as yf
import pandas as pd


def download_history(ticker: str, **kwargs) -> pd.DataFrame:
    """Download stock history data from Yahoo Finance.
    Limitations:
        For '1m' interval is:
            - The maximum amount of data that can be downloaded is 7 days.
            - Maximum last 30 days.
        For '5m' - '30m' interval is:
            - The maximum amount of data that can be downloaded is 7 days.
            - Maximum last 60 days.

    You can get data for '60m' interval for ever.

    Args:
        ticker (str): Ticker of the stock.
        **kwargs: Keyword arguments for yfinance.Ticker.history

    Returns:
        pd.DataFrame: Stock history data.
    """
    ticker = yf.Ticker(ticker)
    data = ticker.history(**kwargs)
    return data


if __name__ == '__main__':
    data = download_history('AAPL', start='2023-10-01',
                            end=None, period='1d', interval='30m')

    print(data.head())
