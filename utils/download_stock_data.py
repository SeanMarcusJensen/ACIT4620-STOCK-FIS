import os
import pandas as pd
import yfinance as yf


def save(data: pd.DataFrame, ticker_name: str, interval: str | None = None) -> pd.DataFrame:
    """Save stock data to a file.
    Args:
        data (pd.DataFrame): Stock data.
        ticker_name(str): Name of the ticker.
    """
    BASE_PATH = 'data/'
    folder = os.path.join(BASE_PATH, ticker_name, interval) if interval else os.path.join(
        BASE_PATH, ticker_name)
    file_name = os.path.join(folder, f'{ticker_name}.csv')

    if not os.path.exists(folder):
        os.makedirs(folder)

    if os.path.exists(file_name):
        data.to_csv(os.path.join(
            folder, f'{ticker_name}.csv'), mode='a', header=False, index=False)
        data = pd.read_csv(file_name, index_col='Datetime')
    else:
        data.to_csv(os.path.join(
            folder, f'{ticker_name}.csv'), index=False)
    return data


def download_stock_data(ticker_name: str, **kwargs) -> pd.DataFrame:
    DAY_LIMIT = 60
    DAYS = 7
    INTERVAL = kwargs.get('interval', '5m')
    PERIOD = kwargs.get('period', '1d')

    if os.path.exists(os.path.join('data', ticker_name, INTERVAL)):
        return pd.read_csv(os.path.join('data', ticker_name, INTERVAL, f'{ticker_name}.csv'), index_col='Datetime')

    if PERIOD == '1d':
        start = pd.Timestamp.now() - pd.Timedelta(days=DAY_LIMIT)
        end = pd.Timestamp.now()
        current_time = start

        data = pd.DataFrame()

        while current_time <= end:
            end_time = current_time + pd.Timedelta(days=DAYS)
            stock = yf.Ticker(ticker_name)
            data = stock.history(
                start=current_time,
                end=end_time,
                period=PERIOD,
                interval=INTERVAL)
            data.index.names = ['Datetime']
            data.reset_index(inplace=True)
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            data = save(data, ticker_name, INTERVAL)
            current_time = end_time
        return data
    else:
        stock = yf.Ticker(ticker_name)
        data = stock.history(period=PERIOD, interval=INTERVAL)
        data.index.names = ['Datetime']
        data.reset_index(inplace=True)
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data = save(data, ticker_name, INTERVAL)
        return data


if __name__ == '__main__':
    stocks = ['AAPL']
    for stock in stocks:
        download_stock_data(stock)
